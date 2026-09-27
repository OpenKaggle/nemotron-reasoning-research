# # Nemotron Huikang 0.87 Submit


# %% cell 1
import json
import shutil
import zipfile
from pathlib import Path


BASE_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
WORKING = Path("/kaggle/working")


def main() -> None:
    input_root = Path("/kaggle/input")
    zip_candidates = sorted(
        input_root.rglob("submission.zip"),
        key=lambda p: ("agi-for-medal" not in str(p), len(str(p))),
    )
    if not zip_candidates:
        raise FileNotFoundError("No submission.zip found under /kaggle/input")

    source_zip = zip_candidates[0]
    print("Using source zip:", source_zip)
    with zipfile.ZipFile(source_zip, "r") as zf:
        names = set(zf.namelist())
        print("Zip entries:", sorted(names))
        required = {"adapter_config.json", "adapter_model.safetensors"}
        missing = required - names
        if missing:
            raise ValueError(f"Missing required root files: {sorted(missing)}")
        config = json.loads(zf.read("adapter_config.json"))

    print("peft_type:", config.get("peft_type"))
    print("rank:", config.get("r"))
    print("base_model:", config.get("base_model_name_or_path"))
    assert str(config.get("peft_type", "")).upper() == "LORA"
    assert int(config.get("r", 999)) <= 32
    assert config.get("base_model_name_or_path") == BASE_MODEL_NAME

    out_zip = WORKING / "submission.zip"
    shutil.copy2(source_zip, out_zip)
    print("Wrote", out_zip, "size", out_zip.stat().st_size)


main()
