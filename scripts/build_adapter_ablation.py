from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from adapter_audit import key_category, resolve_adapter


def load_tensors(path: Path) -> dict:
    with safe_open(path, framework="pt") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def write_zip(output_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            zf.write(output_dir / name, name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path, help="Adapter carrying SFT deltas")
    parser.add_argument("--reference", required=True, type=Path, help="Baseline adapter to start from")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--keep-category", action="append", default=[])
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()

    keep_categories = set(args.keep_category)
    if not keep_categories:
        raise ValueError("At least one --keep-category is required")

    src_config, src_tensor, src_tmp = resolve_adapter(args.source)
    ref_config, ref_tensor, ref_tmp = resolve_adapter(args.reference)
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        out_config = args.output / "adapter_config.json"
        out_tensor = args.output / "adapter_model.safetensors"

        shutil.copy2(ref_config, out_config)
        tensors = load_tensors(ref_tensor)
        source_tensors = load_tensors(src_tensor)

        copied = {}
        missing = []
        for key, value in source_tensors.items():
            category = key_category(key)
            if category not in keep_categories:
                continue
            if key not in tensors:
                missing.append(key)
                continue
            tensors[key] = value
            copied[category] = copied.get(category, 0) + 1

        save_file(tensors, out_tensor)

        manifest = {
            "source": str(args.source),
            "reference": str(args.reference),
            "keep_categories": sorted(keep_categories),
            "copied": dict(sorted(copied.items())),
            "missing": missing,
            "num_tensors": len(tensors),
        }
        (args.output / "ablation_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        if args.zip:
            write_zip(args.output, args.output / "submission.zip")

        print(json.dumps(manifest, indent=2))
    finally:
        if src_tmp is not None:
            src_tmp.cleanup()
        if ref_tmp is not None:
            ref_tmp.cleanup()


if __name__ == "__main__":
    main()
