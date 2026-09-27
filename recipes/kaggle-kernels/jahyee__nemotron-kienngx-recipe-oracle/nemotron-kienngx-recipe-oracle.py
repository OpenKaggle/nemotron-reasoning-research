# Nemotron Kienngx-recipe SFT — fork of kienngx/nvidia-nemotron-training-cot-labels
# Phase 1a: DATA_MODE="kienngx"  (1200 cot-labels, 2 epochs)  — pure repro of Kienngx 0.86
# Phase 1b: DATA_MODE="oracle"   (9056 oracle traces, 1 epoch)
# Phase 1c: DATA_MODE="mix"      (1200 + 9056, 1 epoch)
# Phase 2 : DATA_MODE="mix_plus" (+kh0a solver CoT, set later)
DATA_MODE = "kienngx"

# ===== Cell 1: offline pip install (verbatim from Kienngx) =====
import subprocess, sys, os
from pathlib import Path

def resolve_python_path(target_dir):
    for pth_file in Path(target_dir).glob("*.pth"):
        with pth_file.open() as fp:
            relpath = fp.read()
            rel_pack_path = (pth_file.parent / relpath)
            if rel_pack_path.exists():
                print(f"append {rel_pack_path}")
                sys.path.append(str(rel_pack_path))

# Resolve dataset / kernel_source mount paths. Notebook and script kernel modes
# differ on whether they use dashes or underscores and whether they prefix with
# `datasets/<owner>/`. Probe and pick the one that exists.
def _first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

offline_dir = _first_existing(
    "/kaggle/input/nvidia-nemotron-offline-packages/offline_packages",
    "/kaggle/input/datasets/dennisfong/nvidia-nemotron-offline-packages/offline_packages",
)
target_dir = "/kaggle/working/packages"
os.makedirs(target_dir, exist_ok=True)

NVUTIL = _first_existing(
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/",
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/",
)
if NVUTIL:
    if NVUTIL not in sys.path:
        sys.path.insert(0, NVUTIL)
    print(f"NVUTIL={NVUTIL}")
    resolve_python_path(NVUTIL)
else:
    print("WARN: NVUTIL not found")
    try:
        print(os.listdir("/kaggle/usr/lib/notebooks/ryanholbrook"))
    except Exception as _e:
        print(f"  listdir err: {_e}")

if offline_dir and os.path.exists(offline_dir):
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "--no-index",
        "--find-links", offline_dir,
        "--target", target_dir,
        "datasets", "trl",
    ])
    print(f"Installed from offline packages: {offline_dir}")

sys.path.append(target_dir)
resolve_python_path(target_dir)

import datasets  # noqa: F401
try:
    import cutlass  # noqa: F401
    print("cutlass import OK")
except Exception as _e:
    print(f"cutlass import skipped: {_e}")

# ===== Cell 3: env + imports (verbatim) =====
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json, stat, shutil, gc, zipfile  # noqa: E402
import polars as pl  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import kagglehub  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import LoraConfig, get_peft_model, TaskType  # noqa: E402
from trl import SFTTrainer, SFTConfig  # noqa: E402

# ===== Cell 5: triton + rmsnorm fix (verbatim) =====
def _pure_rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5,
                     group_size=None, norm_before_gate=True, upcast=True):
    dtype = x.dtype
    if upcast:
        x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    out = x_normed * weight.float()
    if bias is not None:
        out = out + bias.float()
    if z is not None:
        out = out * F.silu(z.float())
    return out.to(dtype)

for name, mod in list(sys.modules.items()):
    if hasattr(mod, "rmsnorm_fn"):
        mod.rmsnorm_fn = _pure_rmsnorm_fn

src = os.path.join(NVUTIL, "triton/backends/nvidia/bin/ptxas-blackwell") if NVUTIL else ""
dst = "/tmp/ptxas-blackwell"
print(f"ptxas-fix: NVUTIL={NVUTIL}  src_exists={os.path.exists(src) if src else False}  src={src}")
sys.stdout.flush()
if src and os.path.exists(src):
    shutil.copy2(src, dst)
    os.chmod(dst, os.stat(dst).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    try:
        import triton.backends.nvidia as nv_backend
        src_bin = os.path.join(os.path.dirname(nv_backend.__file__), "bin")
        dst_bin = "/tmp/triton_nvidia_bin"
        shutil.copytree(src_bin, dst_bin, dirs_exist_ok=True)
        for f in os.listdir(dst_bin):
            fp = os.path.join(dst_bin, f)
            if os.path.isfile(fp):
                os.chmod(fp, os.stat(fp).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        nv_backend.__file__ = os.path.join(dst_bin, "..", "__init__.py")
    except Exception as _e:
        print(f"nv_backend redirect err: {_e}")
    os.environ["TRITON_PTXAS_PATH"] = dst
    os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = dst
    print(f"Triton ptxas fix applied: src={src} dst={dst}")
else:
    print(f"WARN: ptxas-blackwell not found at {src}; triton may fail at first train step")

# Build a real NvidiaTool pointing at our writable /tmp/ptxas-blackwell so Triton
# subprocesses our executable copy AND gets the binary's real version string
# (avoids the "PTX 8.0 doesn't support sm_120a" mismatch the fake stub caused).
try:
    import triton.backends.nvidia.compiler as nv_compiler
    import triton.knobs as _triton_knobs

    _real = _triton_knobs.NvidiaTool.from_path(dst)
    if _real is None:
        # Fallback: build a stub object with the right shape; bump version so
        # Triton picks PTX ISA >= 8.7 (compatible with sm_120a Blackwell).
        class _FakePtxas:
            def __init__(self, path, version="12.8"):
                self.path = path
                self.version = version
        _real = _FakePtxas(dst)
        print(f"NvidiaTool.from_path returned None, using FakePtxas(version=12.8)")
    else:
        print(f"NvidiaTool.from_path -> path={_real.path} version={_real.version}")

    # Patch at class level so descriptor lookups return our real tool.
    try:
        type(_triton_knobs.nvidia).ptxas_blackwell = _real
        type(_triton_knobs.nvidia).ptxas = _real
    except Exception as _e2:
        print(f"  class-level patch err: {_e2}")
    _triton_knobs.nvidia.ptxas_blackwell = _real
    _triton_knobs.nvidia.ptxas = _real
    nv_compiler.get_ptxas = lambda arch: _real
    # Keep Kienngx's get_ptxas_version stub as a harmless second layer.
    nv_compiler.get_ptxas_version = lambda arch: getattr(_real, "version", "12.8")
    print(f"monkey-patched knobs.nvidia.ptxas_blackwell -> {_real.path}")
except Exception as _e:
    print(f"knobs monkey-patch err: {_e}")

# ===== Cell 7: hyperparams (Kienngx recipe; epochs depend on data size) =====
LORA_RANK = 32
MAX_SEQ_LEN = 2048
BATCH_SIZE = 1
GRAD_ACCUM = 4
LR = 5e-5
OUTPUT_DIR = "/kaggle/working/adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if DATA_MODE == "kienngx":
    SUBSAMPLE_SIZE = 1200
    NUM_EPOCHS = 2
elif DATA_MODE == "oracle":
    SUBSAMPLE_SIZE = None
    NUM_EPOCHS = 1
elif DATA_MODE == "mix":
    SUBSAMPLE_SIZE = None
    NUM_EPOCHS = 1
else:
    raise ValueError(f"Unknown DATA_MODE: {DATA_MODE}")

MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
print(f"MODEL_PATH = {MODEL_PATH}")

KIENNGX_CSV = "/kaggle/input/datasets/kienngx/nemotron-30b-competition-trainingdata-cot-labels/final_Nemotron_training_data.csv"
ORACLE_JSONL = "/kaggle/input/datasets/jahyee/nemotron-oracle-reasoning-traces/oracle_reasoning_traces.jsonl"


def load_kienngx(n=None):
    df = pl.read_csv(KIENNGX_CSV)
    if n is not None:
        df = df.sample(n=n, seed=42)
    pdf = df.to_pandas()
    # Kienngx schema: prompt, answer, generated_cot
    pdf["completion"] = None
    return pdf


def load_oracle():
    rows = []
    with open(ORACLE_JSONL) as f:
        for line in f:
            rec = json.loads(line)
            rows.append({
                "prompt": rec["prompt"],
                "answer": str(rec["answer"]),
                "generated_cot": None,
                "completion": rec["completion"],
            })
    return pd.DataFrame(rows)


if DATA_MODE == "kienngx":
    train_pdf = load_kienngx(n=SUBSAMPLE_SIZE)
elif DATA_MODE == "oracle":
    train_pdf = load_oracle()
elif DATA_MODE == "mix":
    df_k = load_kienngx(n=None)  # use full Kienngx (~5K-ish? actually CSV has more)
    df_o = load_oracle()
    train_pdf = pd.concat([df_k, df_o], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

print(f"DATA_MODE={DATA_MODE}  rows={len(train_pdf)}")
hf_dataset = Dataset.from_pandas(train_pdf)

# ===== Cell 9: tokenizer + prompt builder =====
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def build_training_text(example):
    prompt = example["prompt"]
    answer = example["answer"]
    cot = example.get("generated_cot")
    completion = example.get("completion")

    user_msg = prompt + "\nPut your final answer inside \\boxed{}."

    # If we have a pre-built completion (oracle), use it verbatim.
    # Otherwise build "{cot}\n\n\\boxed{{answer}}" like Kienngx.
    if completion:
        assistant_msg = completion
    else:
        assistant_msg = f"{cot}\n\n\\boxed{{{answer}}}"

    try:
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        text = (
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
        )
    return {"text": text}


hf_dataset = hf_dataset.map(build_training_text, remove_columns=hf_dataset.column_names)

# ===== Cell 11: model + LoRA + train (verbatim except for derived NUM_EPOCHS) =====
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={"": 0},
    trust_remote_code=True,
    dtype=torch.bfloat16,
)
model.gradient_checkpointing_enable()

for name, mod in sys.modules.items():
    if "modeling_nemotron_h" in name:
        mod.is_fast_path_available = False
        print(f"Patched {name}: is_fast_path_available = False")

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

import triton.backends.nvidia.compiler as nv_compiler  # noqa: E402
os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = "/tmp/ptxas-blackwell"
nv_compiler.get_ptxas_version = lambda arch: "12.0"

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    logging_steps=30,
    bf16=True,
    max_grad_norm=1.0,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    save_strategy="no",
    report_to="none",
    dataset_text_field="text",
    max_length=MAX_SEQ_LEN,
    packing=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

trainer = SFTTrainer(
    model=model,
    train_dataset=hf_dataset,
    processing_class=tokenizer,
    args=training_args,
)

print(f"Training start: DATA_MODE={DATA_MODE}, {len(hf_dataset)} samples, {NUM_EPOCHS} epochs")
trainer.train()
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Adapter saved to {OUTPUT_DIR}:")
for f in os.listdir(OUTPUT_DIR):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"  {f} ({size / 1024:.1f} KB)")

# Defensive: rewrite local kagglehub cache path -> public HF repo id so the
# Kaggle evaluator can resolve the base model. Kienngx didn't do this and
# still scored 0.86, but we've previously seen 0.50 runs traced to this.
_cfg_path = os.path.join(OUTPUT_DIR, "adapter_config.json")
if os.path.exists(_cfg_path):
    with open(_cfg_path) as f:
        _cfg = json.load(f)
    _orig = _cfg.get("base_model_name_or_path", "")
    if _orig.startswith("/"):
        _cfg["base_model_name_or_path"] = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
        with open(_cfg_path, "w") as f:
            json.dump(_cfg, f, indent=2)
        print(f"Rewrote base_model_name_or_path: {_orig} -> {_cfg['base_model_name_or_path']}")
    else:
        print(f"base_model_name_or_path already public: {_orig}")

# ===== Cell 14: submission zip (verbatim) =====
zip_path = "/kaggle/working/submission.zip"
print(f"Packaging files from {OUTPUT_DIR}...")
SUBMISSION_FILES = ["adapter_config.json", "adapter_model.safetensors"]
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in SUBMISSION_FILES:
        fpath = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(fpath):
            raise AssertionError(f"CRITICAL: {fname} missing from adapter output")
        zf.write(fpath, fname)

print(f"Created {zip_path} ({os.path.getsize(zip_path) / 1024 / 1024:.1f} MB)")

with zipfile.ZipFile(zip_path, "r") as zf:
    zip_contents = zf.namelist()
    print(f"Zip Contents: {zip_contents}")
    if sorted(zip_contents) != sorted(SUBMISSION_FILES):
        raise AssertionError(f"CRITICAL: unexpected submission.zip contents: {zip_contents}")

print("submission.zip ready")
