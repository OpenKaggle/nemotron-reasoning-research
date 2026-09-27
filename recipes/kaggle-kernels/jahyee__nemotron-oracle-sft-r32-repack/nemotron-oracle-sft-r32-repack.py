"""Repack the Unsloth-trained adapter into a Kienngx/PEFT-compatible adapter,
with configurable Kienngx warm-start splicing for MoE-expert LoRA tensors.

Why this script exists
----------------------
Trained submission `jahyee/nemotron-oracle-sft-r32` v3 scored 0.50 vs the
0.86 Kienngx baseline. After two failed repacks (0.50 / 0.51 / 0.51) we
compared safetensors values element-wise against the Kienngx adapter that
was used as warm-start:

  category                                count   mean_cos
  experts.down_proj.lora_A                 2944   0.9915  (preserved)
  experts.down_proj.lora_B                 2944   0.0019  (randomized!)
  experts.up_proj.lora_A                   2944   0.1496  (collapsed)
  experts.up_proj.lora_B                   2944   0.9603  (preserved)
  in_proj.lora_{A,B}                         23   0.96-0.99
  out_proj.lora_{A,B}                        23   ~1.0000
  lm_head.lora_{A,B}                          1   ~0.99

The MoE-expert LoRA tensors `experts.up_proj.lora_A` and
`experts.down_proj.lora_B` no longer reflect the Kienngx warm start, while
their lora_A/lora_B counterparts were preserved. The likely cause is the
Unsloth + PEFT combination of `target_modules` (per-expert submodules) and
`target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]`
in adapter_config: the parameter-level LoRA path either initialised those
two keys from scratch or merged them poorly back into the per-expert
layout on save.

This kernel therefore:
- reads the v3 submission.zip,
- rewrites LoRA keys from Unsloth's `backbone.*` namespace into Kienngx's
  `model.*` / `lm_head.*` layout,
- drops `lm_head.base_layer.weight` (Kienngx adapters don't carry it),
- splices configurable Kienngx MoE tensors back over the trained values,
- rewrites adapter_config to Kienngx's baseline layout,
- emits the final submission.zip at /kaggle/working.

RESTORE_MODE:
- `all_moe` (default): restore every `*.experts.*` LoRA tensor from Kienngx,
  keeping only non-MoE SFT deltas. This is the next queued ablation because
  partial MoE restoration scored 0.84 while full SFT scored 0.50/0.52.
- `broken_moe`: restore only `experts.*.up_proj.lora_A` and
  `experts.*.down_proj.lora_B`, matching submission 53279382.
- `none`: key/config rewrite only, matching the 0.51 repack behavior.

Fallback safety: if Kienngx adapter cannot be located, the script still
performs the key-rewrite repack and emits a zip (matching the previous
repack v2 behaviour, ~0.51).
"""

import json
import os
import shutil
import zipfile

from safetensors import safe_open
from safetensors.torch import save_file

BASE_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
INPUT_ROOTS = ("/kaggle/input/kernels", "/kaggle/input")
RESTORE_MODE = os.environ.get("RESTORE_MODE", "all_moe")

# Keys whose cosine similarity to the Kienngx warm-start was ~0 in V3.
# We restore these from the original Kienngx adapter.
BROKEN_MOE_SUFFIXES = (
    ".up_proj.lora_A.weight",
    ".down_proj.lora_B.weight",
)


def find_v3_zip() -> str:
    for root in INPUT_ROOTS:
        for dirpath, _dirs, files in os.walk(root):
            if (
                "submission.zip" in files
                and "nemotron-oracle-sft-r32" in dirpath
                and "repack" not in dirpath
            ):
                return os.path.join(dirpath, "submission.zip")
    raise FileNotFoundError(f"v3 submission.zip not found under {INPUT_ROOTS}")


def find_kienngx_safetensors() -> str | None:
    for root in INPUT_ROOTS:
        for dirpath, _dirs, files in os.walk(root):
            if "adapter_model.safetensors" not in files:
                continue
            if "tinker-adapter" in dirpath and "nano-30b-trained" in dirpath:
                return os.path.join(dirpath, "adapter_model.safetensors")
    return None


def rename_v3_key(key: str) -> str:
    return (
        key
        .replace("base_model.model.backbone.layers.", "base_model.model.model.layers.")
        .replace("base_model.model.backbone.lm_head.", "base_model.model.lm_head.")
    )


def should_restore_key(key: str) -> bool:
    if RESTORE_MODE == "none":
        return False
    if ".experts." not in key:
        return False
    if RESTORE_MODE in {"all_moe", "all_experts"}:
        return ".lora_A." in key or ".lora_B." in key
    if RESTORE_MODE == "broken_moe":
        return any(key.endswith(suffix) for suffix in BROKEN_MOE_SUFFIXES)
    raise ValueError(f"Unknown RESTORE_MODE={RESTORE_MODE!r}")


def main() -> None:
    print(f"RESTORE_MODE={RESTORE_MODE}")
    v3_zip = find_v3_zip()
    print(f"V3 zip: {v3_zip} size={os.path.getsize(v3_zip):,}")

    kg_st = find_kienngx_safetensors()
    if kg_st:
        print(f"Kienngx warm-start: {kg_st} size={os.path.getsize(kg_st):,}")
    else:
        print("WARNING: Kienngx adapter not found; falling back to key-rewrite-only repack")

    extract_dir = "/kaggle/working/_extract"
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(v3_zip, "r") as zf:
        zf.extractall(extract_dir)
    print(f"Extracted: {os.listdir(extract_dir)}")

    # ----- 1. Rewrite adapter_config.json -----
    config_path = os.path.join(extract_dir, "adapter_config.json")
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["base_model_name_or_path"] = BASE_MODEL_NAME
    cfg.setdefault("rank_pattern", {})["in_proj"] = 32
    cfg.setdefault("alpha_pattern", {})["in_proj"] = 32
    cfg["auto_mapping"] = None
    cfg.pop("target_parameters", None)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print("adapter_config.json rewritten (base_model_name, rank/alpha_pattern, cleared auto_mapping/target_parameters)")

    # ----- 2. Build the rewritten tensor dict from V3 -----
    src_st = os.path.join(extract_dir, "adapter_model.safetensors")
    tensors = {}
    skipped_base_layer = 0
    with safe_open(src_st, framework="pt") as h:
        for old_key in h.keys():
            if old_key.endswith("lm_head.base_layer.weight"):
                skipped_base_layer += 1
                continue
            tensors[rename_v3_key(old_key)] = h.get_tensor(old_key)
    print(f"V3 key rewrite: kept {len(tensors)}, dropped {skipped_base_layer} (lm_head.base_layer)")

    n_lm = sum(1 for k in tensors if k.startswith("base_model.model.lm_head."))
    n_layers = sum(1 for k in tensors if k.startswith("base_model.model.model.layers."))
    print(f"  lm_head keys: {n_lm}; model.layers keys: {n_layers}")
    assert n_lm > 0 and n_layers > 0, "V3 key rewrite failed"

    # ----- 3. Splice Kienngx's MoE expert tensors over the selected V3 keys -----
    restored = 0
    skipped_missing = 0
    if kg_st:
        with safe_open(kg_st, framework="pt") as h:
            kg_keys = set(h.keys())
            for k in list(tensors.keys()):
                if not should_restore_key(k):
                    continue
                if k not in kg_keys:
                    skipped_missing += 1
                    continue
                tensors[k] = h.get_tensor(k)
                restored += 1
        print(
            f"Restored {restored} MoE-expert LoRA tensors from Kienngx "
            f"(mode={RESTORE_MODE}, skipped {skipped_missing} missing)"
        )
        if restored == 0:
            print("WARNING: 0 tensors restored; expected nonzero for all restore modes except none")

    save_file(tensors, src_st)
    print(f"Wrote rewritten safetensors: {os.path.getsize(src_st):,} bytes")

    # ----- 4. Repack into submission.zip at /kaggle/working -----
    output_zip = "/kaggle/working/submission.zip"
    if os.path.exists(output_zip):
        os.remove(output_zip)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name in sorted(os.listdir(extract_dir)):
            zf.write(os.path.join(extract_dir, name), name)
            print(f"  + {name}")
    print(f"Wrote {output_zip} size={os.path.getsize(output_zip):,}")

    shutil.rmtree(extract_dir, ignore_errors=True)
    print("Cleaned up _extract")


if __name__ == "__main__":
    main()
