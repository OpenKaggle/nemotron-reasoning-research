from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError as exc:  # pragma: no cover - import-time guard for CLI users.
    raise SystemExit(
        "adapter_svd_soup.py requires torch and safetensors. "
        "Try running it with the project virtualenv: .venv/bin/python scripts/adapter_svd_soup.py ..."
    ) from exc


LORA_A_MARKER = ".lora_A."
LORA_B_MARKER = ".lora_B."


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    source: Path
    weight: float


@dataclass
class ResolvedAdapter:
    spec: AdapterSpec
    config_path: Path
    tensor_path: Path
    config: dict[str, Any]
    tmpdir: tempfile.TemporaryDirectory[str] | None = None

    def cleanup(self) -> None:
        if self.tmpdir is not None:
            self.tmpdir.cleanup()


@dataclass(frozen=True)
class Pair:
    module: str
    a_key: str
    b_key: str


def parse_adapter_spec(value: str) -> AdapterSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--adapter must look like name=path:weight")
    name, rest = value.split("=", 1)
    path_text, sep, weight_text = rest.rpartition(":")
    if not name or not sep or not path_text or not weight_text:
        raise argparse.ArgumentTypeError("--adapter must look like name=path:weight")
    try:
        weight = float(weight_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid adapter weight: {weight_text}") from exc
    return AdapterSpec(name=name, source=Path(path_text), weight=weight)


def resolve_adapter(spec: AdapterSpec) -> ResolvedAdapter:
    path = spec.source.expanduser().resolve()
    tmpdir: tempfile.TemporaryDirectory[str] | None = None

    if path.is_file() and path.suffix == ".zip":
        tmpdir = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(path) as zf:
            members = {Path(name).name: name for name in zf.namelist() if not name.endswith("/")}
            missing = {"adapter_config.json", "adapter_model.safetensors"} - set(members)
            if missing:
                raise FileNotFoundError(f"{path} is missing {sorted(missing)}")
            zf.extract(members["adapter_config.json"], tmpdir.name)
            zf.extract(members["adapter_model.safetensors"], tmpdir.name)
            config_path = Path(tmpdir.name) / members["adapter_config.json"]
            tensor_path = Path(tmpdir.name) / members["adapter_model.safetensors"]
    elif path.is_dir():
        config_path = path / "adapter_config.json"
        tensor_path = path / "adapter_model.safetensors"
    else:
        config_path = path.with_name("adapter_config.json")
        tensor_path = path

    if not config_path.exists() or not tensor_path.exists():
        if tmpdir is not None:
            tmpdir.cleanup()
        raise FileNotFoundError(f"Could not resolve adapter_config.json and adapter_model.safetensors from {path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    return ResolvedAdapter(spec=spec, config_path=config_path, tensor_path=tensor_path, config=config, tmpdir=tmpdir)


def compile_patterns(values: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(value) for value in values]


def module_from_a_key(a_key: str) -> str:
    return a_key.split(LORA_A_MARKER, 1)[0]


def b_key_from_a_key(a_key: str) -> str:
    return a_key.replace(LORA_A_MARKER, LORA_B_MARKER)


def include_pair(module: str, includes: list[re.Pattern[str]], excludes: list[re.Pattern[str]]) -> bool:
    if includes and not any(pattern.search(module) for pattern in includes):
        return False
    return not any(pattern.search(module) for pattern in excludes)


def tensor_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            try:
                shapes[key] = tuple(handle.get_slice(key).get_shape())
            except Exception:
                shapes[key] = tuple(handle.get_tensor(key).shape)
    return shapes


def anchor_pairs(shapes: dict[str, tuple[int, ...]], includes: list[re.Pattern[str]], excludes: list[re.Pattern[str]]) -> list[Pair]:
    pairs: list[Pair] = []
    for a_key in sorted(key for key in shapes if LORA_A_MARKER in key):
        b_key = b_key_from_a_key(a_key)
        if b_key not in shapes:
            continue
        module = module_from_a_key(a_key)
        if include_pair(module, includes, excludes):
            pairs.append(Pair(module=module, a_key=a_key, b_key=b_key))
    return pairs


def lora_scale(config: dict[str, Any], module: str, rank: int) -> float:
    alpha = pattern_value(config.get("alpha_pattern"), module)
    if alpha is None:
        alpha = config.get("lora_alpha", rank)
    return float(alpha) / float(rank)


def pattern_value(pattern: Any, module: str) -> Any:
    if not isinstance(pattern, dict):
        return None
    if module in pattern:
        return pattern[module]
    suffix_matches = [(key, value) for key, value in pattern.items() if module.endswith(str(key))]
    if suffix_matches:
        return max(suffix_matches, key=lambda item: len(str(item[0])))[1]
    return None


def rank_for_pair(shapes: dict[str, tuple[int, ...]], pair: Pair) -> int:
    a_shape = shapes[pair.a_key]
    b_shape = shapes[pair.b_key]
    if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[0] != b_shape[1]:
        raise ValueError(f"Invalid LoRA pair shapes for {pair.module}: A={a_shape}, B={b_shape}")
    return int(a_shape[0])


def classify_pair(
    pair: Pair,
    anchor_shape: tuple[tuple[int, ...], tuple[int, ...]],
    adapters: list[ResolvedAdapter],
    all_shapes: dict[str, dict[str, tuple[int, ...]]],
) -> tuple[str, list[str]]:
    missing_or_mismatched: list[str] = []
    anchor_delta_shape = delta_shape(anchor_shape[0], anchor_shape[1])
    for adapter in adapters:
        shapes = all_shapes[adapter.spec.name]
        if pair.a_key not in shapes or pair.b_key not in shapes:
            missing_or_mismatched.append(adapter.spec.name)
            continue
        if delta_shape(shapes[pair.a_key], shapes[pair.b_key]) != anchor_delta_shape:
            missing_or_mismatched.append(adapter.spec.name)
    if missing_or_mismatched:
        return "incompatible", missing_or_mismatched
    return "compatible", []


def delta_shape(a_shape: tuple[int, ...], b_shape: tuple[int, ...]) -> tuple[int, int] | None:
    if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[0] != b_shape[1]:
        return None
    return int(b_shape[0]), int(a_shape[1])


def summarize(
    adapters: list[ResolvedAdapter],
    anchor: ResolvedAdapter,
    pairs: list[Pair],
    all_shapes: dict[str, dict[str, tuple[int, ...]]],
    shape_policy: str,
    target_rank: int,
    target_alpha: float,
) -> dict[str, Any]:
    by_action = {"svd": 0, "anchor_copy": 0, "skipped": 0, "error": 0}
    modules: list[dict[str, Any]] = []
    errors: list[str] = []

    for pair in pairs:
        anchor_shape = (all_shapes[anchor.spec.name][pair.a_key], all_shapes[anchor.spec.name][pair.b_key])
        status, offenders = classify_pair(pair, anchor_shape, adapters, all_shapes)
        if status == "compatible":
            action = "svd"
        elif shape_policy == "anchor-copy":
            action = "anchor_copy"
        elif shape_policy == "common-only":
            action = "skipped"
        else:
            action = "error"
            errors.append(f"{pair.module}: incompatible adapters={offenders}")
        by_action[action] += 1
        modules.append(
            {
                "module": pair.module,
                "a_key": pair.a_key,
                "b_key": pair.b_key,
                "anchor_a_shape": list(anchor_shape[0]),
                "anchor_b_shape": list(anchor_shape[1]),
                "status": status,
                "action": action,
                "incompatible_adapters": offenders,
            }
        )

    return {
        "adapters": [
            {"name": item.spec.name, "source": str(item.spec.source), "weight": item.spec.weight}
            for item in adapters
        ],
        "anchor": anchor.spec.name,
        "target_rank": target_rank,
        "target_alpha": target_alpha,
        "target_scale": target_alpha / target_rank,
        "shape_policy": shape_policy,
        "num_anchor_pairs": len(pairs),
        "actions": by_action,
        "errors": errors,
        "modules": modules,
    }


def qr_svd_factor(
    left_blocks: list[torch.Tensor],
    right_blocks: list[torch.Tensor],
    target_rank: int,
    target_scale: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    left = torch.cat(left_blocks, dim=1).float()
    right = torch.cat(right_blocks, dim=0).float()
    max_rank = min(left.shape[0], right.shape[1], left.shape[1])
    if target_rank <= 0 or target_rank > max_rank:
        raise ValueError(
            f"target rank {target_rank} is invalid for delta shape {(left.shape[0], right.shape[1])} "
            f"and concatenated rank {left.shape[1]}"
        )

    q_left, r_left = torch.linalg.qr(left, mode="reduced")
    q_right, r_right = torch.linalg.qr(right.T, mode="reduced")
    core = torch.matmul(r_left, r_right.T)
    u_core, s, vh_core = torch.linalg.svd(core, full_matrices=False)

    u = torch.matmul(q_left, u_core[:, :target_rank])
    vh = torch.matmul(vh_core[:target_rank, :], q_right.T)
    sqrt_s = torch.sqrt(s[:target_rank].clamp_min(0))
    b = u * sqrt_s.unsqueeze(0)
    a = sqrt_s.unsqueeze(1) * vh

    if target_scale == 0:
        raise ValueError("target alpha / target rank scale must be non-zero")
    b = b / target_scale
    return a.to(dtype=dtype).contiguous(), b.to(dtype=dtype).contiguous()


def merge_pair(pair: Pair, adapters: list[ResolvedAdapter], target_rank: int, target_alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    target_scale = float(target_alpha) / float(target_rank)
    out_dtype: torch.dtype | None = None
    left_blocks: list[torch.Tensor] = []
    right_blocks: list[torch.Tensor] = []

    for adapter in adapters:
        with safe_open(adapter.tensor_path, framework="pt", device="cpu") as handle:
            a = handle.get_tensor(pair.a_key)
            b = handle.get_tensor(pair.b_key)
        rank = int(a.shape[0])
        scale = lora_scale(adapter.config, pair.module, rank)
        left_blocks.append(b.float() * (float(adapter.spec.weight) * scale))
        right_blocks.append(a.float())
        if out_dtype is None:
            out_dtype = a.dtype

    if not left_blocks or out_dtype is None:
        raise ValueError(f"No compatible adapters for {pair.module}")
    return qr_svd_factor(left_blocks, right_blocks, target_rank, target_scale, out_dtype)


def load_all_tensors(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def output_config(
    anchor_config: dict[str, Any],
    manifest: dict[str, Any],
    all_shapes: dict[str, dict[str, tuple[int, ...]]],
    target_rank: int,
    target_alpha: float,
) -> dict[str, Any]:
    config = dict(anchor_config)
    config["r"] = target_rank
    config["lora_alpha"] = target_alpha
    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, float] = {}
    anchor_name = str(manifest["anchor"])
    for item in manifest["modules"]:
        if item["action"] != "anchor_copy":
            continue
        module = item["module"]
        rank = rank_for_pair(all_shapes[anchor_name], Pair(module=module, a_key=item["a_key"], b_key=item["b_key"]))
        alpha = lora_scale(anchor_config, module, rank) * rank
        if rank != target_rank:
            rank_pattern[module] = rank
        if float(alpha) != float(target_alpha):
            alpha_pattern[module] = alpha
    config["rank_pattern"] = rank_pattern
    config["alpha_pattern"] = alpha_pattern
    return config


def write_zip(output_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            zf.write(output_dir / name, name)


def maybe_write_manifest(manifest: dict[str, Any], manifest_json: str | None) -> None:
    if manifest_json is None:
        return
    text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    if manifest_json == "-":
        print(text, end="")
    else:
        Path(manifest_json).write_text(text, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a delta-space LoRA SVD soup from PEFT adapters.")
    parser.add_argument("--adapter", action="append", type=parse_adapter_spec, required=True, help="name=path:weight")
    parser.add_argument("--anchor", required=True, help="Adapter name whose key set/config anchor the output.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory.")
    parser.add_argument("--target-rank", type=int, default=None)
    parser.add_argument("--target-alpha", type=float, default=None)
    parser.add_argument("--shape-policy", choices=("strict", "anchor-copy", "common-only"), default="strict")
    parser.add_argument("--include-module", action="append", default=[], help="Regex matched against module names.")
    parser.add_argument("--exclude-module", action="append", default=[], help="Regex matched against module names.")
    parser.add_argument("--dry-run", action="store_true", help="Only report compatibility; do not write adapter weights.")
    parser.add_argument("--zip", action="store_true", help="Also write submission.zip containing only adapter files.")
    parser.add_argument(
        "--manifest-json",
        nargs="?",
        const="-",
        default=None,
        help="Write manifest JSON to this path, or stdout when passed without a value.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    names = [spec.name for spec in args.adapter]
    if len(set(names)) != len(names):
        raise SystemExit(f"Adapter names must be unique: {names}")
    if args.anchor not in set(names):
        raise SystemExit(f"--anchor {args.anchor!r} is not one of {sorted(names)}")

    adapters = [resolve_adapter(spec) for spec in args.adapter]
    try:
        by_name = {adapter.spec.name: adapter for adapter in adapters}
        anchor = by_name[args.anchor]
        target_rank = args.target_rank if args.target_rank is not None else int(anchor.config.get("r") or 0)
        target_alpha = args.target_alpha if args.target_alpha is not None else float(anchor.config.get("lora_alpha") or target_rank)
        if target_rank <= 0:
            raise SystemExit("--target-rank must be positive or anchor config must contain positive r")

        includes = compile_patterns(args.include_module)
        excludes = compile_patterns(args.exclude_module)
        all_shapes = {adapter.spec.name: tensor_shapes(adapter.tensor_path) for adapter in adapters}
        pairs = anchor_pairs(all_shapes[anchor.spec.name], includes, excludes)
        manifest = summarize(adapters, anchor, pairs, all_shapes, args.shape_policy, target_rank, target_alpha)

        if manifest["errors"]:
            maybe_write_manifest(manifest, args.manifest_json)
            print(json.dumps({key: manifest[key] for key in ("num_anchor_pairs", "actions", "errors")}, indent=2))
            raise SystemExit("Shape incompatibilities found under --shape-policy strict")

        if args.dry_run:
            print(json.dumps({key: manifest[key] for key in ("num_anchor_pairs", "actions", "errors")}, indent=2))
            maybe_write_manifest(manifest, args.manifest_json)
            return

        args.output.mkdir(parents=True, exist_ok=True)
        tensors = load_all_tensors(anchor.tensor_path)
        for item in manifest["modules"]:
            action = item["action"]
            if action == "svd":
                pair = Pair(module=item["module"], a_key=item["a_key"], b_key=item["b_key"])
                tensors[pair.a_key], tensors[pair.b_key] = merge_pair(pair, adapters, target_rank, target_alpha)
            elif action == "anchor_copy":
                continue
            elif action == "skipped":
                tensors.pop(item["a_key"], None)
                tensors.pop(item["b_key"], None)

        (args.output / "adapter_config.json").write_text(
            json.dumps(
                output_config(anchor.config, manifest, all_shapes, target_rank, target_alpha),
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        save_file(tensors, args.output / "adapter_model.safetensors")

        manifest["output"] = str(args.output)
        (args.output / "merge_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if args.zip:
            write_zip(args.output, args.output / "submission.zip")
        maybe_write_manifest(manifest, args.manifest_json)
        print(json.dumps({key: manifest[key] for key in ("num_anchor_pairs", "actions", "errors")}, indent=2))
    finally:
        for adapter in adapters:
            adapter.cleanup()


if __name__ == "__main__":
    main()
