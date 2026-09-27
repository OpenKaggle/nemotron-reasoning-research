# # Nemotron oracle SFT pipeline


# %% cell 2
# ── Shared config ─────────────────────────────────────────────────────
LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0

MAX_SEQ_LEN = 8192
NUM_STEPS = 40
BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4
LEARNING_RATE = 5e-6
DIAGNOSTIC_ONLY = False  # set True only in the separate diagnostic kernel to audit load mapping and exit
RESET_WEIGHTS = (
    False  # if True, skip loading pretrained adapter; train from fresh LoRA init
)
IN_PROJ_ONLY = True
MOE_TIE_WEIGHTS = False  # Tinker-style tying mean-broadcasts the warm-start across 128 experts at init;
# this destroyed the per-expert structure of the Kienngx warm-start adapter (verified empirically:
# experts.up_proj.lora_A cos_sim collapsed to 0.15, experts.down_proj.lora_B to 0.002), which is what
# caused submissions 53249867/53264508/53276955 to score ~0.50 vs the 0.86 Kienngx baseline.
ORIGINAL_PROBLEMS_ONLY = (
    False  # if True, filter examples to only problem_ids listed in train.csv
)
SHUFFLE_DATASET = True

BASE_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
TRACE_DATASET_SLUG = "jahyee/nemotron-oracle-reasoning-traces"
PRETRAINED_ADAPTER_MODEL = (
    "kienngx/nemotron-nano-30b-trained/Triton/tinker-adapter/1"
)
PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

KAGGLE_DATASET = "jahyee/nemotron-oracle-sft-r32-inproj-lr5e-6-s40-output"
MINUTES = 60

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "in_proj",
    "out_proj",
    "lm_head",
]


# %% cell 3
import os

IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
IS_MODAL_WORKER = "MODAL_TASK_ID" in os.environ
IS_MODAL_LAUNCHER = not IS_KAGGLE and not IS_MODAL_WORKER


# %% cell 4
# ── Env-specific install (Kaggle only; Modal image has packages pre-installed) ──
if IS_KAGGLE:
    import subprocess

    subprocess.run(
        "pip install -q --no-index --find-links /kaggle/input/datasets/mayukh18/nemotron-packages/packages "
        "unsloth trl peft transformers datasets accelerate bitsandbytes",
        shell=True,
        check=True,
    )
    subprocess.run(
        "pip install -q /kaggle/input/datasets/mayukh18/nemotron-packages/causal_conv1d-1.6.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
        shell=True,
        check=True,
    )
    subprocess.run(
        "pip install -q /kaggle/input/datasets/mayukh18/nemotron-packages/mamba_ssm-2.3.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
        shell=True,
        check=True,
    )
    for _wd in ["/kaggle/input/datasets/llkh0a/rtx-wheels/wheels"]:
        if os.path.isdir(_wd):
            subprocess.run(
                [
                    "pip",
                    "install",
                    "-q",
                    "--no-index",
                    "--find-links",
                    _wd,
                    "protobuf==6.33.5",
                    "sentencepiece",
                    "safetensors",
                    "huggingface_hub",
                ],
                check=False,
            )
    subprocess.run("rm -rf /kaggle/tmp/*", shell=True, check=True)


# %% cell 5
def adapter_state_dict_key_candidates(adapter_key: str) -> list[str]:
    """Map submission-style LoRA keys onto Unsloth's backbone + default adapter names."""
    variants: list[str] = []

    def add(key: str) -> None:
        if key not in variants:
            variants.append(key)

    add(adapter_key)
    add(
        adapter_key.replace(".lora_A.weight", ".lora_A.default.weight").replace(
            ".lora_B.weight", ".lora_B.default.weight"
        )
    )
    for key in list(variants):
        if ".model.model." in key:
            add(key.replace(".model.model.", ".model.backbone."))
        if ".model.lm_head." in key:
            add(key.replace(".model.lm_head.", ".model.backbone.lm_head."))
        if ".model.model.lm_head." in key:
            add(key.replace(".model.model.lm_head.", ".model.backbone.lm_head."))
    for key in list(variants):
        add(
            key.replace(".lora_A.weight", ".lora_A.default.weight").replace(
                ".lora_B.weight", ".lora_B.default.weight"
            )
        )
    return variants


def adapter_key_category(key: str) -> str:
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
    for name in (
        "in_proj",
        "out_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ):
        if f".{name}." in key:
            side = "lora_A" if ".lora_A." in key else "lora_B" if ".lora_B." in key else "other"
            return f"{name}.{side}"
    return "other"


def run_training() -> None:
    """Full training flow. Runs on Kaggle at module level or inside Modal container via train_remote()."""
    import gc
    import json
    import math
    import random
    import subprocess
    import sys
    import time
    from collections import defaultdict

    from unsloth import FastLanguageModel

    import torch
    from cut_cross_entropy import linear_cross_entropy
    from peft import LoraConfig
    from peft.tuners.lora import Linear as LoraLinear

    # ── Env-specific paths + adapter source ──────────────────────────
    if IS_KAGGLE:
        import kagglehub

        TRACE_PATH = "/kaggle/input/datasets/jahyee/nemotron-oracle-reasoning-traces/oracle_reasoning_traces.jsonl"
        if not os.path.isfile(TRACE_PATH):
            _trace_candidates = sorted(
                os.path.join(root, "oracle_reasoning_traces.jsonl")
                for root, _, files in os.walk("/kaggle/input")
                if "oracle_reasoning_traces.jsonl" in files
            )
            assert _trace_candidates, "No oracle_reasoning_traces.jsonl found under /kaggle/input"
            TRACE_PATH = _trace_candidates[0]
        print(f"Using oracle trace corpus: {TRACE_PATH}")
        TRAIN_CSV_PATH = "/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/train.csv"
        ADAPTER_SRC = (
            "/kaggle/input/models/kienngx/nemotron-nano-30b-trained/triton/tinker-adapter/1"
        )
        if not RESET_WEIGHTS:
            if not os.path.isfile(os.path.join(ADAPTER_SRC, "adapter_model.safetensors")):
                _adapter_dirs = sorted(
                    os.path.join(root, "adapter_model.safetensors")
                    for root, _, files in os.walk("/kaggle/input")
                    if "adapter_model.safetensors" in files
                    and "tinker-adapter" in root
                )
                assert _adapter_dirs, (
                    f"No adapter_model.safetensors found for {PRETRAINED_ADAPTER_MODEL} "
                    "under /kaggle/input"
                )
                ADAPTER_SRC = os.path.dirname(_adapter_dirs[0])
            print(f"Loading pretrained adapter from: {ADAPTER_SRC}")
        MODEL_PATH = kagglehub.model_download(
            "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
        )
    else:  # IS_MODAL_WORKER
        MODEL_PATH = "unsloth/Nemotron-3-Nano-30B-A3B"
        TRACE_PATH = "/data/oracle_reasoning_traces.jsonl"
        TRAIN_CSV_PATH = "/data/train.csv"
        ADAPTER_SRC = "/merged/weights"
        OUTPUT_DIR = "/output/weights"

    # ── GPU + kernel sanity check (runs on both Kaggle and Modal worker) ──
    import causal_conv1d
    import mamba_ssm

    cc = torch.cuda.get_device_capability(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}, sm_{cc[0] * 10 + cc[1]}")
    print(f"torch={torch.__version__}, cuda={torch.version.cuda}")
    print(
        f"mamba_ssm={mamba_ssm.__version__}, causal_conv1d={causal_conv1d.__version__}"
    )
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    if IS_MODAL_WORKER:
        assert cc == (12, 0), (
            f"Expected sm_120 (RTX PRO 6000), got sm_{cc[0] * 10 + cc[1]}"
        )
    from causal_conv1d import causal_conv1d_fn

    _x = torch.randn(1, 256, 32, device="cuda", dtype=torch.bfloat16)
    _w = torch.randn(256, 4, device="cuda", dtype=torch.bfloat16)
    causal_conv1d_fn(_x, _w, None, activation="silu")
    print("causal_conv1d CUDA kernel: OK")

    # Clear stale HF modules cache (Modal-only; bug: persists across runs)
    if IS_MODAL_WORKER:
        import shutil as _shutil

        hf_modules = os.path.join(
            os.environ.get("HF_HOME", "/root/.cache/huggingface"), "modules"
        )
        if os.path.exists(hf_modules):
            _shutil.rmtree(hf_modules)

    # ── Load corpus into `examples` list ─────────────────────────────
    examples: list[dict] = []

    from transformers import AutoTokenizer

    chat_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )
    family_counts: dict[str, int] = {}
    method_counts: dict[str, int] = {}
    oracle_count = 0
    with open(TRACE_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            messages = [{"role": "user", "content": rec["prompt"] + PROMPT_SUFFIX}]
            prompt_ids = chat_tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            completion_text = (
                rec["completion"].rstrip()
                + "\n</think>\n\\boxed{"
                + rec["answer"]
                + "}<|im_end|>"
            )
            completion_ids = chat_tokenizer.encode(
                completion_text,
                add_special_tokens=False,
            )
            tokens = prompt_ids + completion_ids
            mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
            if len(tokens) > MAX_SEQ_LEN:
                tokens = tokens[:MAX_SEQ_LEN]
                mask = mask[:MAX_SEQ_LEN]
            if not any(mask) or len(tokens) < 2:
                continue
            examples.append(
                {
                    "problem_id": rec["id"],
                    "task_type": rec["task_type"],
                    "method": rec["method"],
                    "oracle": bool(rec.get("oracle", False)),
                    "tokens": tokens[:-1],
                    "targets": tokens[1:],
                    "weights": [float(m) for m in mask[1:]],
                }
            )
            family_counts[rec["task_type"]] = family_counts.get(rec["task_type"], 0) + 1
            method_counts[rec["method"]] = method_counts.get(rec["method"], 0) + 1
            oracle_count += int(bool(rec.get("oracle", False)))
    print(f"Loaded oracle trace corpus from {TRACE_PATH}")
    print(f"Trace family counts: {family_counts}")
    print(f"Top methods: {sorted(method_counts.items(), key=lambda kv: kv[1], reverse=True)[:12]}")
    print(f"Gold-conditioned oracle traces: {oracle_count}")

    if ORIGINAL_PROBLEMS_ONLY:
        import csv

        with open(TRAIN_CSV_PATH) as f:
            original_ids = {row["id"] for row in csv.DictReader(f)}
        before = len(examples)
        examples = [e for e in examples if e["problem_id"] in original_ids]
        print(
            f"ORIGINAL_PROBLEMS_ONLY=True: filtered {before} → {len(examples)} examples "
            f"using {len(original_ids)} ids from {TRAIN_CSV_PATH}"
        )

    total_unmasked = sum(sum(e["weights"]) for e in examples)
    total_tokens = sum(len(e["tokens"]) for e in examples)
    print(
        f"Loaded {len(examples)} examples, {total_tokens:,} tokens "
        f"(unmasked={total_unmasked:,.0f})"
    )

    # ── Load base model ──────────────────────────────────────────────
    gc.collect()
    torch.cuda.empty_cache()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        unsloth_force_compile=True,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )
    if IS_MODAL_WORKER:
        hf_cache_vol.commit()  # noqa: F821 — defined at module level on non-Kaggle

    # ── Wrap in LoRA ─────────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    FastLanguageModel.for_training(model)

    # ── Patch Mamba CUDA fast path ───────────────────────────────────
    nemotron_mod = None
    for _name, _m in sys.modules.items():
        if "modeling_nemotron_h" in _name and hasattr(_m, "is_fast_path_available"):
            nemotron_mod = _m
            break
    assert nemotron_mod is not None, "Could not find modeling_nemotron_h module"
    print(f"is_fast_path_available was: {nemotron_mod.is_fast_path_available}")
    nemotron_mod.is_fast_path_available = True  # type: ignore[attr-defined]
    print("Patched is_fast_path_available = True")

    # ── Manually add lm_head LoRA (Unsloth drops it for MoE) ─────────
    _causal_lm = model
    while hasattr(_causal_lm, "model"):
        _causal_lm = _causal_lm.model
    _lm_head = _causal_lm.lm_head
    if not isinstance(_lm_head, LoraLinear):
        _cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT)
        model.base_model._create_and_replace(
            _cfg,
            "default",
            target=_lm_head,
            target_name="lm_head",
            parent=_causal_lm,
        )
        print("Manually added LoRA to lm_head")
    else:
        print("lm_head already has LoRA")

    # ── Cast LoRA params to fp32 (base model stays bf16 except MoE router) ──
    for name, param in model.named_parameters():
        if ".lora_" in name:
            param.data = param.data.to(torch.float32)

    for name, param in model.named_parameters():
        if ".lora_" in name:
            assert param.dtype == torch.float32, (
                f"LoRA param {name} expected fp32, got {param.dtype}"
            )
            continue

        is_router = (
            ".mixer.gate." in name
        )  # NemotronHTopkRouter.weight + e_score_correction_bias
        # Nemotron-H loads the MoE router (`mixer.gate`) in fp32 on purpose.
        # Ref: transformers/src/transformers/models/nemotron_h/modeling_nemotron_h.py
        #
        #   class NemotronHPreTrainedModel(PreTrainedModel):
        #       _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
        #
        #   class NemotronHTopkRouter(nn.Module):
        #       def __init__(self, config):
        #           self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        #           self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts))
        #       def forward(self, hidden_states):
        #           router_logits = F.linear(
        #               hidden_states.type(torch.float32),
        #               self.weight.type(torch.float32),
        #           )
        #           return router_logits
        #
        # The per-forward fp32 cast on `self.weight` plus the strict list entry
        # mean the gate weight is promoted to fp32 at load time.
        if is_router:
            assert param.dtype == torch.float32, (
                f"param {name} expected fp32, got {param.dtype}"
            )
            continue

        assert param.dtype == torch.bfloat16, (
            f"param {name} expected bf16, got {param.dtype}"
        )
        continue

    print("Verified: LoRA params fp32, base params bf16 (MoE router fp32)")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {trainable:,} trainable / {total:,} total parameters")

    # ── Patch forward with Cut Cross-Entropy ─────────────────────────
    _base = model
    while hasattr(_base, "model"):
        _base = _base.model

    def _patched_causal_forward(
        input_ids=None, attention_mask=None, labels=None, **kwargs
    ):
        backbone_out = _base.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **{
                k: v
                for k, v in kwargs.items()
                if k in ("position_ids", "past_key_values", "use_cache")
            },
        )
        hidden_states = backbone_out[0]
        lm_head = _base.lm_head
        base_w = lm_head.base_layer.weight
        lora_A = lm_head.lora_A["default"].weight
        lora_B = lm_head.lora_B["default"].weight
        scaling = lm_head.scaling["default"]
        lm_weight = base_w + scaling * lora_B @ lora_A
        if labels is not None:
            per_token_ce = linear_cross_entropy(
                hidden_states, lm_weight, labels, reduction="none"
            )
            loss = per_token_ce.mean()
        else:
            per_token_ce = None
            loss = None
        model._cached_per_token_ce = per_token_ce  # type: ignore[attr-defined]
        return loss

    _base.forward = _patched_causal_forward
    print("Patched CausalLM.forward with CCE (no logits materialization)")

    # ── Load adapter weights (unless RESET_WEIGHTS) ──────────────────
    if RESET_WEIGHTS:
        print(
            "RESET_WEIGHTS=True — skipping pretrained adapter load; using fresh LoRA init"
        )
        loaded = 0
        adapter_weights: dict = {}
    else:
        print(f"Loading adapter from {ADAPTER_SRC}...")
        from peft import load_peft_weights

        adapter_weights = load_peft_weights(ADAPTER_SRC)

        model_sd = model.state_dict()
        new_sd: dict = {}
        loaded = 0
        missing_keys: list[str] = []
        target_sources: dict[str, list[str]] = defaultdict(list)
        for ak, av in adapter_weights.items():
            matched_key = None
            for candidate in adapter_state_dict_key_candidates(ak):
                if candidate in model_sd:
                    matched_key = candidate
                    break
            if matched_key is None:
                missing_keys.append(ak)
                continue
            new_sd[matched_key] = av
            target_sources[matched_key].append(ak)
            loaded += 1

        duplicate_targets = {
            key: sources for key, sources in target_sources.items() if len(sources) > 1
        }
        print(
            f"  Adapter load targets: {len(target_sources)} unique model keys "
            f"for {loaded} source tensors"
        )
        if duplicate_targets:
            sample = list(duplicate_targets.items())[:5]
            print("  ERROR: multiple adapter tensors map to the same model key:")
            for target_key, sources in sample:
                print(f"    {target_key} <= {sources[:4]} ... total={len(sources)}")
            raise AssertionError(
                f"{len(duplicate_targets)} duplicate adapter load targets; "
                "warm-start would be overwritten before training"
            )

        model.load_state_dict(new_sd, strict=False)
        if missing_keys:
            print(
                f"  WARNING: {len(missing_keys)} adapter keys had no model match; "
                f"sample: {missing_keys[:5]}"
            )
        assert loaded == len(adapter_weights), (
            f"Not all adapter weights loaded: {loaded}/{len(adapter_weights)}"
        )
        print(f"  Loaded {loaded}/{len(adapter_weights)} weights into model")

        diagnostic_report = {
            "config": {
                "diagnostic_only": DIAGNOSTIC_ONLY,
                "reset_weights": RESET_WEIGHTS,
                "in_proj_only": IN_PROJ_ONLY,
                "moe_tie_weights": MOE_TIE_WEIGHTS,
                "num_steps": NUM_STEPS,
                "batch_size": BATCH_SIZE,
                "micro_batch_size": MICRO_BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
            },
            "adapter_source": ADAPTER_SRC,
            "loaded": loaded,
            "source_tensors": len(adapter_weights),
            "unique_model_targets": len(target_sources),
            "missing_keys": missing_keys,
        }

        if DIAGNOSTIC_ONLY:
            load_compare = defaultdict(
                lambda: {
                    "count": 0,
                    "matched": 0,
                    "missing": 0,
                    "shape_mismatch": 0,
                    "mean_cos_sum": 0.0,
                    "mean_abs_delta_sum": 0.0,
                    "min_cos": None,
                    "max_cos": None,
                }
            )
            model_sd_after = model.state_dict()
            with torch.no_grad():
                for target_key, source_value in new_sd.items():
                    source_key = target_sources[target_key][0]
                    category = adapter_key_category(source_key)
                    bucket = load_compare[category]
                    bucket["count"] += 1
                    current_value = model_sd_after.get(target_key)
                    if current_value is None:
                        bucket["missing"] += 1
                        continue
                    if tuple(current_value.shape) != tuple(source_value.shape):
                        bucket["shape_mismatch"] += 1
                        continue
                    source_flat = source_value.detach().float().reshape(-1).cpu()
                    current_flat = current_value.detach().float().reshape(-1).cpu()
                    denom = torch.linalg.vector_norm(source_flat) * torch.linalg.vector_norm(current_flat)
                    if denom.item() == 0:
                        cos = 1.0 if torch.equal(source_flat, current_flat) else 0.0
                    else:
                        cos_t = torch.dot(source_flat, current_flat) / denom
                        cos = float(cos_t.item()) if torch.isfinite(cos_t) else 0.0
                    delta = (current_flat - source_flat).abs().mean().item()
                    bucket["matched"] += 1
                    bucket["mean_cos_sum"] += cos
                    bucket["mean_abs_delta_sum"] += delta
                    bucket["min_cos"] = cos if bucket["min_cos"] is None else min(bucket["min_cos"], cos)
                    bucket["max_cos"] = cos if bucket["max_cos"] is None else max(bucket["max_cos"], cos)

            category_compare = {}
            print("  Exact post-load tensor comparison vs adapter source:")
            for category, bucket in sorted(load_compare.items()):
                matched = max(bucket["matched"], 1)
                item = {
                    "count": bucket["count"],
                    "matched": bucket["matched"],
                    "missing": bucket["missing"],
                    "shape_mismatch": bucket["shape_mismatch"],
                    "mean_cos": bucket["mean_cos_sum"] / matched,
                    "mean_abs_delta": bucket["mean_abs_delta_sum"] / matched,
                    "min_cos": bucket["min_cos"],
                    "max_cos": bucket["max_cos"],
                }
                category_compare[category] = item
                print(
                    f"    {category}: matched={item['matched']}/{item['count']}, "
                    f"missing={item['missing']}, shape_mismatch={item['shape_mismatch']}, "
                    f"mean_cos={item['mean_cos']:.6f}, "
                    f"mean_abs_delta={item['mean_abs_delta']:.8f}"
                )
                assert item["missing"] == 0, f"{category} has missing post-load tensors"
                assert item["shape_mismatch"] == 0, f"{category} has shape-mismatched post-load tensors"
                assert item["mean_cos"] > 0.999, f"{category} changed during load"
                assert item["mean_abs_delta"] < 1e-8, f"{category} changed during load"
            diagnostic_report["category_load_compare"] = category_compare

        moe_loaded = defaultdict(lambda: {"count": 0, "mean_abs": 0.0, "slice_std": 0.0})
        for name, param in model.named_parameters():
            if ".experts." not in name or ".lora_" not in name:
                continue
            if ".up_proj." in name and ".lora_A." in name:
                category = "experts.up_proj.lora_A"
            elif ".up_proj." in name and ".lora_B." in name:
                category = "experts.up_proj.lora_B"
            elif ".down_proj." in name and ".lora_A." in name:
                category = "experts.down_proj.lora_A"
            elif ".down_proj." in name and ".lora_B." in name:
                category = "experts.down_proj.lora_B"
            else:
                category = "experts.other"
            data = param.detach().float()
            stats = moe_loaded[category]
            stats["count"] += 1
            stats["mean_abs"] += data.abs().mean().item()
            if data.dim() >= 3 and data.shape[0] > 1:
                flat = data.reshape(data.shape[0], -1)
                stats["slice_std"] += flat.abs().mean(dim=1).std().item()

        print("  Loaded MoE LoRA audit before training:")
        moe_loaded_summary = {}
        for category, stats in sorted(moe_loaded.items()):
            count = stats["count"]
            mean_abs = stats["mean_abs"] / max(count, 1)
            slice_std = stats["slice_std"] / max(count, 1)
            moe_loaded_summary[category] = {
                "params": count,
                "mean_abs": mean_abs,
                "slice_mean_abs_std": slice_std,
            }
            print(
                f"    {category}: params={count}, "
                f"mean_abs={mean_abs:.6f}, slice_mean_abs_std={slice_std:.6f}"
            )
            assert mean_abs > 1e-6, f"{category} appears effectively zero after warm-start load"

        if DIAGNOSTIC_ONLY:
            diagnostic_report["loaded_moe_lora_stats"] = moe_loaded_summary
            report_path = (
                "/kaggle/working/diagnostic_report.json"
                if IS_KAGGLE
                else os.path.join(OUTPUT_DIR, "diagnostic_report.json")
            )
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(diagnostic_report, f, indent=2)
            print(f"DIAGNOSTIC_ONLY=True: wrote {report_path}; exiting before training")
            return

    # ── Freeze all LoRA params except in_proj (if IN_PROJ_ONLY) ──
    print(f"{IN_PROJ_ONLY=}")
    if IN_PROJ_ONLY:
        for name, param in model.named_parameters():
            if param.requires_grad and ".in_proj." not in name:
                param.requires_grad = False
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  {trainable_params:,} trainable / {frozen_params:,} frozen")

    # ── MoE tied-weight params (Tinker convention) ───────────────────
    # Tinker ties whichever LoRA side touches the hidden dim:
    #   gate_up_proj / up_proj / w1 / gate_proj  -> tie A (input/hidden side)
    #   down_proj / w2                           -> tie B (output/hidden side)
    # We keep Unsloth's batched [num_experts, ...] tensor layout; "tying" means
    # all 128 expert slices are kept identical. Saving the adapter naturally
    # emits 128 per-expert copies, so submission.zip is untied downstream.
    moe_tied_params: list[torch.Tensor] = []
    if MOE_TIE_WEIGHTS:
        w1_proj_names = ("gate_up_proj", "up_proj", "gate_proj", ".w1.")
        w2_proj_names = ("down_proj", ".w2.")
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if ".experts." not in name or ".lora_" not in name:
                continue
            is_w1 = any(p in name for p in w1_proj_names)
            is_w2 = any(p in name for p in w2_proj_names)
            is_A = ".lora_A." in name
            is_B = ".lora_B." in name
            should_tie = (is_w1 and is_A) or (is_w2 and is_B)
            if not should_tie:
                continue
            if param.dim() < 2 or param.shape[0] <= 1:
                continue
            moe_tied_params.append(param)

        def _tie_param_init() -> None:
            """Make all 128 expert slices identical (mean-and-broadcast)."""
            with torch.no_grad():
                for p in moe_tied_params:
                    mean = p.data.mean(dim=0, keepdim=True)
                    p.data.copy_(mean.expand_as(p.data))

        def _tie_grads() -> None:
            # Sum (not mean) across the expert dim: if W is the shared LoRA factor
            # and each expert uses a copy W_i = W, chain rule gives
            # dL/dW = sum_i dL/dW_i. Inactive experts contribute 0 and router
            # weights are already baked into active g_i, so there's no
            # double-counting. Summing keeps all 128 slices identical after each
            # AdamW step and reproduces the true shared-weight update; mean would
            # be off by a 1/128 lr rescale (and not exactly equivalent under
            # AdamW's eps/weight-decay).
            with torch.no_grad():
                for p in moe_tied_params:
                    if p.grad is None:
                        continue
                    grad_sum = p.grad.sum(dim=0, keepdim=True)
                    p.grad.copy_(grad_sum.expand_as(p.grad))

        print(f"MoE weight tying: {len(moe_tied_params)} params identified for tying")
        if moe_tied_params:
            print(f"  example shapes: {[tuple(p.shape) for p in moe_tied_params[:3]]}")
        _tie_param_init()  # start from a tied state
    else:

        def _tie_grads() -> None:
            pass

    # ── Training loop ────────────────────────────────────────────────
    gc.collect()
    torch.cuda.empty_cache()

    device = next(model.parameters()).device
    optimizer: torch.optim.AdamW | None = None

    indices = list(range(len(examples)))
    if SHUFFLE_DATASET:
        rng = random.Random(0)
        rng.shuffle(indices)
        print(f"SHUFFLE_DATASET=True: shuffled {len(indices)} examples (seed=0)")
    else:
        print(f"SHUFFLE_DATASET=False: keeping corpus order ({len(indices)} examples)")

    training_log: list[str] = []

    def _log(msg: str) -> None:
        print(msg, flush=True)
        training_log.append(msg)

    max_steps = len(examples) // BATCH_SIZE
    num_steps = NUM_STEPS
    if num_steps > max_steps:
        _log(
            f"WARNING: NUM_STEPS={NUM_STEPS} exceeds max_steps={max_steps} "
            f"({len(examples)} examples // {BATCH_SIZE} batch). Clamping to {max_steps}."
        )
        num_steps = max_steps

    _log(
        f"Training: {num_steps} steps, batch_size={BATCH_SIZE}, "
        f"micro_batch_size={MICRO_BATCH_SIZE}, lr={LEARNING_RATE}"
    )

    step = 0
    for batch_start in range(0, len(indices), BATCH_SIZE):
        if step >= num_steps:
            break
        batch_indices = indices[batch_start : batch_start + BATCH_SIZE]
        batch = [examples[i] for i in batch_indices]
        batch_tokens = [e["tokens"] for e in batch]
        batch_targets = [e["targets"] for e in batch]
        batch_weights = [e["weights"] for e in batch]

        n = len(batch)
        n_accum = math.ceil(n / MICRO_BATCH_SIZE)
        total_loss_sum = 0.0
        total_weight_sum = 0.0

        for mb_start in range(0, n, MICRO_BATCH_SIZE):
            mb_end = min(mb_start + MICRO_BATCH_SIZE, n)
            mb_toks = batch_tokens[mb_start:mb_end]
            mb_tgts = batch_targets[mb_start:mb_end]
            mb_wts = batch_weights[mb_start:mb_end]

            n_micro = len(mb_toks)
            max_len = max(len(t) for t in mb_toks)
            total_len = sum(len(t) for t in mb_toks)

            padded_input = torch.zeros(
                n_micro, max_len, dtype=torch.long, device=device
            )
            padded_targets = torch.zeros(
                n_micro, max_len, dtype=torch.long, device=device
            )
            padded_weights = torch.zeros(
                n_micro, max_len, dtype=torch.float32, device=device
            )
            attention_mask = torch.zeros(
                n_micro, max_len, dtype=torch.long, device=device
            )
            for i in range(n_micro):
                seq_len = len(mb_toks[i])
                padded_input[i, :seq_len] = torch.tensor(mb_toks[i], dtype=torch.long)
                padded_targets[i, :seq_len] = torch.tensor(mb_tgts[i], dtype=torch.long)
                padded_weights[i, :seq_len] = torch.tensor(
                    mb_wts[i], dtype=torch.float32
                )
                attention_mask[i, :seq_len] = 1

            t0 = time.time()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model(
                    input_ids=padded_input,
                    attention_mask=attention_mask,
                    labels=padded_targets,
                    use_cache=False,
                )
                per_token_ce = model._cached_per_token_ce  # type: ignore[attr-defined]
                weighted_loss = per_token_ce * padded_weights
                weight_sum_t = padded_weights.sum()
                loss_sum_t = weighted_loss.sum()
                loss = (
                    loss_sum_t / weight_sum_t if weight_sum_t > 0 else loss_sum_t * 0.0
                )

            (loss / n_accum).backward()
            total_loss_sum += loss_sum_t.item()
            total_weight_sum += weight_sum_t.item()
            del loss, per_token_ce, weighted_loss

            t_end = time.time()
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            mem_gb = torch.cuda.memory_allocated() / 1e9
            mb_idx = mb_start // MICRO_BATCH_SIZE
            print(
                f"    micro-batch {mb_idx}: {n_micro} seqs, max_len={max_len}, "
                f"total_len={total_len}, wall={t_end - t0:.1f}s, "
                f"peak={peak_gb:.1f}GB, mem={mem_gb:.1f}GB"
            )

        if optimizer is None:
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=LEARNING_RATE,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.0,
            )
        lr = LEARNING_RATE * (1 - step / num_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        _tie_grads()  # average MoE expert grads before clip+step so Adam stays in sync
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1e9
        )
        optimizer.step()
        optimizer.zero_grad()
        loss_mean = total_loss_sum / total_weight_sum if total_weight_sum > 0 else 0
        step += 1
        _log(
            f"  step {step}/{num_steps}: "
            f"loss:mean={loss_mean:.6f}, grad_norm={grad_norm:.4f}, lr={lr:.2e}"
        )

    print(
        f"\nTraining complete. Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
    )

    # ── Save adapter + rename lm_head keys (identical on both sides) ──
    from safetensors.torch import load_file, save_file

    save_dir = "." if IS_KAGGLE else OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)
    for _f in os.listdir(save_dir):
        if _f.startswith("adapter"):
            os.remove(os.path.join(save_dir, _f))
    model.save_pretrained(save_dir)
    # Rewrite LoRA keys from Unsloth's `backbone` namespace into the public
    # Kienngx/PEFT layout (`model.layers.*` + `lm_head.*`). Mismatched prefixes
    # cause the evaluator's vLLM to silently skip every LoRA weight, which
    # collapses the score to a near-baseline ~0.5 (verified empirically with
    # submissions 53249867 / 53264508).
    st_path = os.path.join(save_dir, "adapter_model.safetensors")
    tensors = load_file(st_path)
    renamed = {}
    skipped = 0
    for k, v in tensors.items():
        # Kienngx adapters do not ship lm_head.base_layer; drop it so the keys
        # match exactly.
        if k.endswith("lm_head.base_layer.weight"):
            skipped += 1
            continue
        new_k = (
            k.replace("base_model.model.backbone.layers.", "base_model.model.model.layers.")
             .replace("base_model.model.backbone.lm_head.", "base_model.model.lm_head.")
        )
        renamed[new_k] = v
    print(
        f"Adapter key rewrite: kept {len(renamed)}, dropped {skipped} (lm_head.base_layer)"
    )
    save_file(renamed, st_path)

    # Rewrite adapter_config to the Kienngx baseline layout: HF base model
    # name, an explicit rank/alpha pattern for in_proj, no PEFT auto_mapping,
    # and no target_parameters field (vLLM does not need it for inference).
    cfg_path = os.path.join(save_dir, "adapter_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf-8") as _cf:
            _cfg = json.load(_cf)
        _cfg["base_model_name_or_path"] = BASE_MODEL_NAME
        _cfg.setdefault("rank_pattern", {})["in_proj"] = LORA_RANK
        _cfg.setdefault("alpha_pattern", {})["in_proj"] = LORA_ALPHA
        _cfg["auto_mapping"] = None
        _cfg.pop("target_parameters", None)
        with open(cfg_path, "w", encoding="utf-8") as _cf:
            json.dump(_cfg, _cf, indent=2)
        print(
            f"Rewrote adapter_config: base_model_name_or_path={BASE_MODEL_NAME}, "
            f"rank_pattern.in_proj={LORA_RANK}, alpha_pattern.in_proj={LORA_ALPHA}"
        )

    # ── Clean unsloth compiled cache (runs on both) ──────────────────
    _ucache = "unsloth_compiled_cache"
    if os.path.isdir(_ucache):
        import shutil as _sh

        _sh.rmtree(_ucache)

    # ── Package & ship (divergent) ───────────────────────────────────
    if IS_KAGGLE:
        import zipfile

        adapter_files = [f for f in os.listdir(save_dir) if f.startswith("adapter")]
        SUBMISSION_ZIP = "submission.zip"
        with zipfile.ZipFile(SUBMISSION_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in adapter_files:
                zf.write(os.path.join(save_dir, fname), fname)
        for fname in adapter_files:
            os.remove(os.path.join(save_dir, fname))
        print(f"Wrote {SUBMISSION_ZIP}")
    else:  # IS_MODAL_WORKER
        import shutil
        import tempfile

        with open(os.path.join(save_dir, "training_log.txt"), "w") as f:
            f.write("\n".join(training_log) + "\n")
        output_vol.commit()  # noqa: F821 — defined at module level on non-Kaggle

        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)
        with open(os.path.join(kaggle_dir, "access_token"), "w") as f:
            f.write(os.environ["KAGGLE_API_TOKEN"])
        upload_dir = tempfile.mkdtemp()
        for fname in os.listdir(save_dir):
            shutil.copy(os.path.join(save_dir, fname), upload_dir)
        metadata = {"id": KAGGLE_DATASET, "title": KAGGLE_DATASET.split("/")[1]}
        with open(os.path.join(upload_dir, "dataset-metadata.json"), "w") as f:
            json.dump(metadata, f)
        print(f"Uploading to Kaggle {KAGGLE_DATASET}...")
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "version",
                "-p",
                upload_dir,
                "-m",
                "post-finetuned adapter + compiled wheels",
            ],
            check=True,
        )
        print("Kaggle upload complete.")
    print("Training complete.")


# %% cell 6
# ── Modal glue: image, app, volumes, train_remote, main ──────────────
# Defined at module level on non-Kaggle so the worker's module import
# registers train_remote with the app. On Kaggle, skipped entirely
# (modal package is not installed there).
if not IS_KAGGLE:
    import modal

    train_image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.1-devel-ubuntu22.04",
            add_python="3.12",
        )
        .entrypoint([])
        .apt_install("git", "build-essential", "clang")
        .pip_install(
            "torch==2.10.0",
            extra_index_url="https://download.pytorch.org/whl/cu128",
        )
        .pip_install(
            "safetensors>=0.5.0",
            "transformers>=4.56.2",
            "accelerate>=1.0.0",
            "peft>=0.15.0",
            "bitsandbytes>=0.45.0",
            "huggingface_hub>=0.36.2",
            "hf-transfer>=0.1.9",
            "numpy",
            "pillow",
            "torchvision",
            "datasets",
            "sentencepiece",
            "xformers",
            "cut-cross-entropy>=25.1.0",
            "wheel",
            "setuptools",
            "trl",
            "kaggle>=1.6.0",
        )
        .run_commands(
            'python -c "import torch.utils.cpp_extension as e; p=e.__file__; '
            "t=open(p).read().replace('raise RuntimeError(CUDA_MISMATCH_MESSAGE', 'pass  # '); "
            "open(p,'w').write(t)\"",
            "TORCH_CUDA_ARCH_LIST='12.0' pip wheel --no-build-isolation --wheel-dir /wheels mamba_ssm==2.3.1 causal_conv1d==1.6.1",
            "pip install --no-deps /wheels/mamba_ssm-*.whl /wheels/causal_conv1d-*.whl",
            "pip install --no-deps 'unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo'",
            "pip install --no-deps 'unsloth[base] @ git+https://github.com/unslothai/unsloth'",
        )
        .pip_install("einops")
        .env({"HF_HOME": "/root/.cache/huggingface"})
    )

    hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
    merged_vol = modal.Volume.from_name("merged-adapter", create_if_missing=True)
    corpus_vol = modal.Volume.from_name("corpus-data", create_if_missing=True)
    output_vol = modal.Volume.from_name("post-finetune-output", create_if_missing=True)

    app = modal.App("post-finetune-pipeline")

    @app.function(
        image=train_image,
        gpu="RTX-PRO-6000",
        volumes={
            "/root/.cache/huggingface": hf_cache_vol,
            "/merged": merged_vol,
            "/data": corpus_vol,
            "/output": output_vol,
        },
        timeout=6 * 60 * MINUTES,
        secrets=[modal.Secret.from_local_environ(["KAGGLE_API_TOKEN"])],
    )
    def train_remote() -> None:
        run_training()

    if IS_MODAL_LAUNCHER:

        @app.local_entrypoint()
        def main() -> None:
            train_remote.remote()


# %% cell 7
# On Kaggle, trigger training directly after cells load.
# On Modal worker, Modal's runtime calls train_remote() which calls run_training().
# On Modal launcher, neither fires (main() submits the remote call instead).
if IS_KAGGLE:
    run_training()
