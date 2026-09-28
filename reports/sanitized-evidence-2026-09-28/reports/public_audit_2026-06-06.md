# Public High-Score Package / Adapter Recon - 2026-06-06

Scope: NVIDIA Nemotron Model Reasoning Challenge public notebooks, datasets, and model/adapters that look like they may exceed the current 0.86 public score. No submissions were made. Large weights were not downloaded except files already present locally from earlier work; new downloads were limited to source, metadata, README, configs, logs, and small CSV samples.

Kaggle CLI used with proxy cleared:

```sh
env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy <redacted-path> ...
```

## Summary Table

| Candidate | Object | Evidence Checked | Type | Verified Public Score / Record | Verdict |
|---|---:|---|---|---|---|
| `kuangyicheng/nemotron-087-training` | kernel | Pulled notebook source; output file list; public LB grep | Training notebook that warm-starts huikang adapter and trains 240 steps | Team `kuangyicheng`, rank 1124, score 0.86, last submission 2026-06-01 09:13:09 | Real training recipe, but no verifiable 0.87; title/source says 0.88/0.87-style. Needs output verification before trusting. |
| `penguin069/nemotron-dapo-05-06` | dataset | File list, metadata, README, adapter_config, training_args | Real large LoRA adapter dataset; 4.26 GB `adapter_model.safetensors` | Team `Penguin` / `penguin069`, rank 90, score 0.86, last submission 2026-06-05 18:10:09 | Real adapter, recent, potentially audit-worthy, but leaderboard says 0.86 not 0.87+. |
| `foysalemonshanto/nemotron-cot-tong-submission` | dataset | File list, metadata, adapter_config | Real large LoRA adapter dataset; 3.55 GB `adapter_model.safetensors` | Team `FOYSAL`, rank 226, score 0.86, last submission 2026-06-05 17:19:20 | Real adapter; no evidence above 0.86. |
| `foysalemonshanto/end-to-end-finetuning-for-lb-0-90` | kernel | Source pulled; output list; README only from output | Training notebook / claimed LB 0.90 title; output includes `submission.zip` | Same FOYSAL team score 0.86 | Title claim not leaderboard-verified. Treat as title bait unless output zip is separately validated. |
| `wethepeople918/sigilagi-nemotron-rank32-adapter` | dataset | File list, metadata, downloaded config | Config-only package; only `adapter_config.json` listed | Team `Nine1Eight` / `wethepeople918`, rank 692, score 0.86, last submission 2026-05-27 03:47:50 | Not a usable adapter as published; no weight file. |
| `wethepeople918/nemotron-rank32-corpus` | dataset | File list | Synthetic corpus, 42.6 MB aggregate plus per-problem JSONL | Same team 0.86 | Data, not adapter. May be useful for training-data ideas. |
| `wethepeople918/repairv4` | dataset | File list | Repair SFT data, README/manifests/jsonl | Same team 0.86 | Data, not adapter. |
| `profmansoor/nemotron-v34-path` | dataset | File list, metadata, README, status, config, trainer_state | Real large LoRA adapter/checkpoints, 3.48 GB submission adapter plus checkpoint | Team `Mansour A. S. Ataa` / `profmansoor`, rank 196, score 0.86, last submission 2026-06-01 12:59:40 | Real adapter, but self status/eval loss only; no 0.87 evidence. |
| `hammadfarooq470/agi-for-medal-0-87` | kernel / local cached output | Source, metadata, output file list, local cached zip/config/log | Real converted adapter: huikang v20 rank-32 SVD/block-topk conversion | Team `Reasoning Rockets` includes `hammadfarooq470`, rank 30, score 0.86, last submission 2026-06-04 16:14:36 | Real adapter, already cached locally; title says 0.87 but public LB is 0.86. Your cocoa/jahyee forks just re-emit this zip. |
| `cocoaai/nvidia-nemotron-huikang-0-87-svd-submit` | kernel | Local source/metadata; output file list | Repack/submitter for Hammad output | No cocoaAI score mapped from LB search; source is Hammad output | Wrapper only; not independent. |
| `johnjanson/agi-for-medal-0-87-is-possible` | kernel | Local source/metadata; output list | Training pipeline based on huikang/kienngx sources | No leaderboard hit by author found in candidate grep | Title is speculative; output resembles Hammad-style adapter package, not verified 0.87. |
| `atahalam/tonghuikang-0-87-nemotron-dataset` | dataset | Metadata, file list page, README, `augmentation.py` | Huikang public repo/training-data snapshot, not adapter | Dataset title references 0.87; no team score mapped here | Data/code resource. Useful context, not a direct submission adapter. |
| `llkh0a/nemotron-unsloth-sft-training-3-30-2` | kernel | Source, output list, small output files/log/train samples; leaderboard grep | Real training notebook/output; output includes adapter and submission; likely large actual files | Team `Kh0a` / `llkh0a`, rank 6, score 0.87, last submission 2026-06-05 16:20:43 | Strongest public high-score lead found. Notebook is old but author/team has verified 0.87. Needs careful output adapter audit if allowed. |
| `goodmeatday`, `refv00` (`NullSira`) | users | Kernel/dataset lists | No public kernels or datasets found by CLI | Rank 1 `NullSira`, score 0.89, members `goodmeatday,refv00` | True high-score team, but no public artifacts found in this pass. |
| Models search: `nemotron` | Kaggle models | Model list | Mostly official/base/known adapters: huikang, kienngx, mirrors, etc. No `0.87 nemotron` model found | N/A | No new 0.87 model object surfaced. |

## Detailed Notes

### `kuangyicheng/nemotron-087-training`

- Source was pulled to `reports/public_audit_2026-06-06/kuangyicheng__nemotron-087-training/nemotron-087-training.ipynb`.
- Notebook title cell says `Nemotron 0.88 (nb02 clone)`.
- Recipe:
  - Base model: `/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1`.
  - Warm-start adapter: `/kaggle/input/models/huikang/nemotron-adapter/transformers/default/27`.
  - Synthetic generators for bit/cipher/unit/etc.
  - LoRA train for `MAX_STEPS = 240`, `learning_rate = 2e-4`, `MAX_LENGTH = 6144`.
  - Saves `adapter_model.safetensors` and `adapter_config.json`, copies reference keys from huikang all-linear if present.
- Output file list via CLI lists `submission.zip`, `adapter_config.json`, `adapter_model.safetensors`, tokenizer files, README. CLI size display for kernel outputs is unreliable for large files in this challenge, so do not infer actual byte size from the 851/918-style values.
- Leaderboard grep: `kuangyicheng` public score 0.86, not 0.87.

### `penguin069/nemotron-dapo-05-06`

- Dataset file list:
  - `README.md` 5.5 KB
  - `adapter_config.json` 1.3 KB
  - `adapter_model.safetensors` 4,259,063,856 bytes
  - tokenizer files and `training_args.bin`
- Config summary:
  - LoRA r=32, alpha=32, dropout=0.0
  - target modules: `up_proj`, `in_proj`, `lm_head`, `o_proj`, `q_proj`, `v_proj`, `down_proj`, `out_proj`, plus one more
  - peft 0.19.1
- README is generic Hugging Face model card boilerplate, no LB evidence.
- Leaderboard: `penguin069` team score 0.86.

### `foysalemonshanto/nemotron-cot-tong-submission`

- Dataset file list:
  - `adapter_config.json` 580 bytes
  - `adapter_model.safetensors` 3,554,384,888 bytes
- Config summary:
  - Base model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
  - LoRA r=32, alpha=32, dropout=0.0
  - target modules include `down_proj`, `in_proj`, `k_proj`, `lm_head`, `o_proj`, `out_proj`, `q_proj`, `up_proj`.
- Leaderboard: FOYSAL team score 0.86.

### `foysalemonshanto/end-to-end-finetuning-for-lb-0-90`

- Kernel output list shows `README.md` and `submission.zip`.
- Source notebook is a real training script, not just a copy cell:
  - `LORA_RANK=32`, `NUM_STEPS=1000`, `LEARNING_RATE=2e-4`.
  - `RESET_WEIGHTS=True`.
  - Uses huikang/nemotron-data style corpus paths and Unsloth/FastLanguageModel.
- Pulled README from output; avoided pulling `submission.zip` after recognizing it may be large despite CLI output size display.
- Leaderboard contradicts title: FOYSAL team score 0.86.

### `wethepeople918/sigilagi-nemotron-rank32-adapter`

- Dataset file list only contains `adapter_config.json` 41,541 bytes.
- Config is a very large target-module list:
  - r=32, alpha=64, dropout=0.03
  - 778 target modules, mostly selected `backbone.layers.*.mixer.experts.*.(up_proj|down_proj)`.
- No `adapter_model.safetensors` exists in the file list, so it is not a loadable/publishable adapter package as-is.

### `profmansoor/nemotron-v34-path`

- Dataset file list includes:
  - `submission/adapter_model.safetensors` 3,479,065,680 bytes
  - `v30_best_adapter/adapter_model.safetensors` 3,479,065,680 bytes
  - `v34_runs/checkpoint-175/adapter_model.safetensors` 1,747,904,016 bytes
  - status/readme/config/trainer state files
- `V34_STATUS.json`:
  - `state`: `TRAINED_AND_PACKAGED`
  - start adapter: `kienngx/nemotron-nano-30b-trained/transformers/1800s-32/1`
- `V34_TRAINING_STATUS.txt`:
  - train rows 9436, eval rows 64
  - lr 5e-6, max_steps 180, max_length 2048
  - eval_loss 3.1979
- Leaderboard: profmansoor score 0.86.

### Hammad / Cocoa / Jahyee / JohnJanson 0.87 Family

- `hammadfarooq470/agi-for-medal-0-87` is the actual builder.
- It loads `huikang/nemotron-adapter/Transformers/default/20`, SVD/block-topk converts/fuses to rank 32, and emits `submission.zip`.
- Local cached files:
  - `submissions/hammadfarooq470__agi-for-medal-0-87/submission.zip` 3.0 GB
  - `adapter_candidates/block_topk_floor4_model/adapter_model.safetensors` 3.3 GB
  - target modules: `down_proj`, `in_proj`, `k_proj`, `lm_head`, `o_proj`, `out_proj`, `q_proj`, `up_proj`, `v_proj`
  - selected candidate `mean_fused_retained_energy = 0.9553816923314378`
- `cocoaai/nvidia-nemotron-huikang-0-87-svd-submit` and private `jahyee/nemotron-huikang-087-submit` are wrappers that validate/copy Hammad output.
- Leaderboard for Hammad team (`Reasoning Rockets`) is 0.86.

### `llkh0a/nemotron-unsloth-sft-training-3-30-2`

- This is the main positive lead:
  - Author/team `llkh0a` maps to `Kh0a`, verified public leaderboard score 0.87.
  - Kernel output includes adapter and `submission.zip`.
  - Source is a real Unsloth SFT training flow with custom traces and category sampling.
- Output list includes `adapter_config.json`, `adapter_model.safetensors`, tokenizer files, `submission.zip`, training CSVs, and Unsloth trainer files.
- Pulled only small outputs:
  - `formatted_train_dataset.csv` 15 MB
  - `train_sample.csv` 1.5 MB
  - `nemotron-unsloth-sft-training-3-30-2.log` 192 KB
- Log confirms real run:
  - Train 9500 rows
  - Data from `llkh0a/nvidia-nemotron-distiled-dataset`
  - `MAX_SEQ_LEN=3500`, epochs=2, batch=2, lr=1e-4
- Risk: The public 0.87 submission may not necessarily be this exact old notebook output; however, author/team score makes it the best public artifact to audit next.

## Leaderboard Evidence

Downloaded public leaderboard to:

`reports/public_audit_2026-06-06/leaderboard/nvidia-nemotron-model-reasoning-challenge.zip`

Key hits:

- Rank 1: `NullSira`, score 0.89, members `goodmeatday,refv00`; no public kernels/datasets found by CLI.
- Rank 6: `Kh0a`, score 0.87, member `llkh0a`.
- `Penguin` / `penguin069`: score 0.86.
- `Mansour A. S. Ataa` / `profmansoor`: score 0.86.
- `FOYSAL` / `foysalemonshanto,rokaiyasomapti`: score 0.86.
- `Reasoning Rockets` / includes `hammadfarooq470`: score 0.86.
- `kuangyicheng`: score 0.86.
- `Nine1Eight` / `wethepeople918`: score 0.86.
- `Jiayi Du` / `jahyee`: score 0.86.

## Next-Step Commands

Only run these if continuing audit. None submit.

### Best lead: `llkh0a` output config/log, then optional weight header

Pull config/tokenizer/readme only:

```sh
mkdir -p <redacted-path>
env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy <redacted-path> kernels output llkh0a/nemotron-unsloth-sft-training-3-30-2 \
  -p <redacted-path> \
  --file-pattern '^(README\.md|adapter_config\.json|special_tokens_map\.json|tokenizer_config\.json|chat_template\.jinja)$' -q
```

Risk: low. Small text/config files only.

If and only if a full adapter audit is approved, pull the adapter/submission. Expected several GB:

```sh
env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy <redacted-path> kernels output llkh0a/nemotron-unsloth-sft-training-3-30-2 \
  -p <redacted-path> \
  --file-pattern '^(adapter_model\.safetensors|adapter_config\.json|submission\.zip)$' -q
```

Risk: high disk/network, possible 3+ GB; also public 0.87 may not be this exact output.

### Penguin / FOYSAL / ProfMansoor true adapters

Inspect metadata/config only:

```sh
env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy <redacted-path> datasets files penguin069/nemotron-dapo-05-06 -v
env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy <redacted-path> datasets files foysalemonshanto/nemotron-cot-tong-submission -v
env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy <redacted-path> datasets files profmansoor/nemotron-v34-path -v
```

Risk: low. Lists only.

Full download is not recommended unless you want to reproduce their 0.86 behavior or compare tensor keys; each is 3.5-4.3 GB.

### Kuangyicheng source/log/output status

Pull source and check output list:

```sh
mkdir -p <redacted-path>
env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy <redacted-path> kernels pull kuangyicheng/nemotron-087-training \
  -p <redacted-path>
env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy <redacted-path> kernels files kuangyicheng/nemotron-087-training -v
```

Risk: low for source/list. Full output may be GB-scale and is not score-verified.

### Public leaderboard refresh

```sh
mkdir -p <redacted-path>
env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy <redacted-path> competitions leaderboard nvidia-nemotron-model-reasoning-challenge \
  --download -p <redacted-path> -q
```

Risk: low. Read-only.

