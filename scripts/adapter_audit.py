from __future__ import annotations

import argparse
import json
import math
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]


def resolve_adapter(path: Path) -> tuple[Path, Path, tempfile.TemporaryDirectory[str] | None]:
    """Return (config_path, safetensors_path, tmpdir)."""
    path = path.resolve()
    tmpdir: tempfile.TemporaryDirectory[str] | None = None
    if path.is_file() and path.suffix == ".zip":
        tmpdir = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(path) as zf:
            members = {Path(name).name: name for name in zf.namelist()}
            if "adapter_config.json" not in members or "adapter_model.safetensors" not in members:
                raise FileNotFoundError(f"{path} is missing adapter_config.json or adapter_model.safetensors")
            zf.extract(members["adapter_config.json"], tmpdir.name)
            zf.extract(members["adapter_model.safetensors"], tmpdir.name)
            config_path = Path(tmpdir.name) / members["adapter_config.json"]
            tensor_path = Path(tmpdir.name) / members["adapter_model.safetensors"]
        return config_path, tensor_path, tmpdir

    if path.is_dir():
        config_path = path / "adapter_config.json"
        tensor_path = path / "adapter_model.safetensors"
    else:
        config_path = path.with_name("adapter_config.json")
        tensor_path = path

    if not config_path.exists() or not tensor_path.exists():
        raise FileNotFoundError(f"Could not resolve adapter files from {path}")
    return config_path, tensor_path, None


def key_category(key: str) -> str:
    if ".experts." in key:
        if ".up_proj.lora_A." in key:
            return "experts.up_proj.lora_A"
        if ".up_proj.lora_B." in key:
            return "experts.up_proj.lora_B"
        if ".down_proj.lora_A." in key:
            return "experts.down_proj.lora_A"
        if ".down_proj.lora_B." in key:
            return "experts.down_proj.lora_B"
        return "experts.other"
    for name in ("in_proj", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "lm_head"):
        if f".{name}." in key:
            side = "lora_A" if ".lora_A." in key else "lora_B" if ".lora_B." in key else "other"
            return f"{name}.{side}"
    return "other"


def tensor_cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    if left.shape != right.shape:
        return None
    a = left.detach().flatten().float()
    b = right.detach().flatten().float()
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if denom.item() == 0:
        return 1.0 if torch.equal(a, b) else 0.0
    value = torch.dot(a, b) / denom
    if not torch.isfinite(value):
        return None
    return float(value.item())


def summarize_adapter(config_path: Path, tensor_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    with safe_open(tensor_path, framework="pt") as handle:
        keys = list(handle.keys())
        categories = Counter(key_category(key) for key in keys)
        shapes = Counter()
        for key in keys:
            try:
                shape = tuple(handle.get_slice(key).get_shape())
            except Exception:
                shape = tuple(handle.get_tensor(key).shape)
            shapes[str(shape)] += 1
    return {
        "config": {
            "base_model_name_or_path": config.get("base_model_name_or_path"),
            "r": config.get("r"),
            "lora_alpha": config.get("lora_alpha"),
            "target_modules": sorted(config.get("target_modules") or []),
            "has_target_parameters": "target_parameters" in config,
            "rank_pattern": config.get("rank_pattern"),
            "alpha_pattern": config.get("alpha_pattern"),
        },
        "num_tensors": len(keys),
        "categories": dict(sorted(categories.items())),
        "top_shapes": shapes.most_common(12),
    }


def compare_adapters(reference_tensor: Path, candidate_tensor: Path, categories: Iterable[str] | None = None) -> dict:
    wanted = set(categories or [])
    by_category: dict[str, dict[str, float | int | None]] = defaultdict(
        lambda: {
            "count": 0,
            "matched": 0,
            "missing": 0,
            "shape_mismatch": 0,
            "mean_cos": 0.0,
            "min_cos": None,
            "max_cos": None,
            "mean_abs_delta": 0.0,
        }
    )
    with safe_open(reference_tensor, framework="pt") as ref, safe_open(candidate_tensor, framework="pt") as cand:
        cand_keys = set(cand.keys())
        for key in ref.keys():
            category = key_category(key)
            if wanted and category not in wanted:
                continue
            bucket = by_category[category]
            bucket["count"] += 1
            if key not in cand_keys:
                bucket["missing"] += 1
                continue
            left = ref.get_tensor(key)
            right = cand.get_tensor(key)
            cos = tensor_cosine(left, right)
            if cos is None:
                bucket["shape_mismatch"] += 1
                continue
            delta = (right.float() - left.float()).abs().mean().item()
            bucket["matched"] += 1
            bucket["mean_cos"] += cos
            bucket["mean_abs_delta"] += delta
            bucket["min_cos"] = cos if bucket["min_cos"] is None else min(float(bucket["min_cos"]), cos)
            bucket["max_cos"] = cos if bucket["max_cos"] is None else max(float(bucket["max_cos"]), cos)

    summary = {}
    for category, item in sorted(by_category.items()):
        matched = int(item["matched"])
        summary[category] = dict(item)
        if matched:
            summary[category]["mean_cos"] = float(item["mean_cos"]) / matched
            summary[category]["mean_abs_delta"] = float(item["mean_abs_delta"]) / matched
        else:
            summary[category]["mean_cos"] = math.nan
            summary[category]["mean_abs_delta"] = math.nan
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--category", action="append", default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    cand_config, cand_tensor, cand_tmp = resolve_adapter(args.candidate)
    ref_tmp = None
    try:
        result = {"candidate": summarize_adapter(cand_config, cand_tensor)}
        if args.reference:
            ref_config, ref_tensor, ref_tmp = resolve_adapter(args.reference)
            result["reference"] = summarize_adapter(ref_config, ref_tensor)
            result["comparison"] = compare_adapters(ref_tensor, cand_tensor, args.category)

        text = json.dumps(result, indent=2, ensure_ascii=False)
        print(text)
        if args.json:
            args.json.write_text(text + "\n", encoding="utf-8")
    finally:
        if cand_tmp is not None:
            cand_tmp.cleanup()
        if ref_tmp is not None:
            ref_tmp.cleanup()


if __name__ == "__main__":
    main()
