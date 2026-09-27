# Repack the trained Kienngx-recipe adapter into a single submission.zip
# so Kaggle's submission service can find it (parent kernel has 33k+ output
# files which causes "Did not find provided Notebook Output File" 400).
import os
import json
import zipfile
import shutil
from pathlib import Path

# Parent kernel mount path (script kernel: dashes -> underscores).
# Kaggle kernel_source OUTPUT files are mounted under /kaggle/input/ (not
# under /kaggle/usr/lib/notebooks/, which only has the kernel SOURCE code).
print("Probing /kaggle/input layout:")
for _root in ["/kaggle/input", "/kaggle/input/kernels"]:
    try:
        print(f"  {_root} -> {os.listdir(_root)[:20]}")
    except Exception as _e:
        print(f"  {_root} -> ERR {_e}")

def find_adapter_dir():
    for root in ["/kaggle/input/kernels", "/kaggle/input"]:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            if "adapter_config.json" in files and "adapter_model.safetensors" in files:
                return dirpath
    return None

ROOT = find_adapter_dir()
if ROOT is None:
    raise RuntimeError("could not find adapter dir with both adapter_config.json + adapter_model.safetensors under /kaggle/input")
print(f"adapter root = {ROOT}")

# Files we need for the LoRA submission. The evaluator only needs these two
# root-level files; tokenizer/README files make the package fail our audit.
NEEDED = [
    "adapter_config.json",
    "adapter_model.safetensors",
]

# Stage files into a clean working dir.
STAGE = "/kaggle/working/adapter_stage"
os.makedirs(STAGE, exist_ok=True)
present = []
missing = []
for name in NEEDED:
    src = os.path.join(ROOT, name)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(STAGE, name))
        present.append(name)
    else:
        missing.append(name)
print(f"staged: {present}")
if missing:
    print(f"missing (non-fatal if just README/templates): {missing}")
if "adapter_config.json" not in present or "adapter_model.safetensors" not in present:
    raise RuntimeError("required adapter files missing")

# Defensive: ensure base_model_name_or_path is the public HF id.
cfg_path = os.path.join(STAGE, "adapter_config.json")
with open(cfg_path) as f:
    cfg = json.load(f)
orig = cfg.get("base_model_name_or_path", "")
if orig.startswith("/") or "kagglehub" in orig:
    cfg["base_model_name_or_path"] = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"rewrote base_model_name_or_path: {orig} -> {cfg['base_model_name_or_path']}")
else:
    print(f"base_model_name_or_path already public: {orig}")

# Build submission.zip with files at root.
ZIP_PATH = "/kaggle/working/submission.zip"
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for name in present:
        zf.write(os.path.join(STAGE, name), name)
sz_mb = os.path.getsize(ZIP_PATH) / 1024 / 1024
print(f"submission.zip = {sz_mb:.1f} MB at {ZIP_PATH}")
with zipfile.ZipFile(ZIP_PATH) as zf:
    contents = zf.namelist()
    print(f"zip contents: {contents}")
    assert sorted(contents) == sorted(NEEDED), f"unexpected zip contents: {contents}"

# Remove the stage dir so Kaggle output contains only submission.zip.
shutil.rmtree(STAGE, ignore_errors=True)
print("done — output should contain only submission.zip")
