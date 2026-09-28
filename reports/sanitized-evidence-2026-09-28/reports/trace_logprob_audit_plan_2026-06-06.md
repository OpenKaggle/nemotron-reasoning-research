# Trace Logprob Audit Plan - 2026-06-06

Script: `scripts/trace_logprob_audit.py`

Purpose: reproduce the Huikang-style diagnostic loop: inspect per-trace minimum logprob, worst token, and trace-family/source tails before doing more training.

## Local Dry-Run

Local machine should be used for parsing and schema checks only.

```bash
<redacted-path> scripts/trace_logprob_audit.py \
  --source data/generated/oracle_reasoning_traces.jsonl \
  --source tmp_kaggle_recon/datasets/leevvin_bitmanip/2026-05-06_tong_reasoning_bit_manipulation.csv \
  --source reports/public_audit_2026-06-06/llkh0a__nemotron-unsloth-sft-training-3-30-2/output_small/formatted_train_dataset.csv \
  --source txt_dir:tmp_kaggle_recon/datasets/sybyrr_symbolic \
  --limit 200 \
  --dry-run \
  --no-tokenizer \
  --output reports/trace_logprob_dryrun.jsonl \
  --csv reports/trace_logprob_dryrun.csv \
  --summary-json reports/trace_logprob_dryrun_summary.json
```

Observed from initial dry-run:

- Local oracle traces parse cleanly.
- `leevvin` Tong bit traces are high-value but carry `multiple_boxed` on every sampled row.
- `sybyrr` symbolic beta traces also often carry `multiple_boxed`.
- `llkh0a` formatted traces are parseable but some prompts classify as `unknown`; keep this as a quality flag/audit concern.

Do not drop `multiple_boxed` by default; it is a diagnostic signal. Dropping it removes most of the high-value bit/symbolic public traces.

## Kaggle GPU Audit

Run the same script inside a Kaggle kernel with model and adapter mounted. Example shape:

```bash
python /kaggle/working/trace_logprob_audit.py \
  --source /kaggle/input/nemotron-oracle-reasoning-traces/oracle_reasoning_traces.jsonl \
  --source /kaggle/input/nemotron-bitmanip-traces-pub/2026-05-06_tong_reasoning_bit_manipulation.csv \
  --task-type bit_manipulation \
  --limit 256 \
  --model-path /kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1 \
  --adapter /kaggle/input/<kienngx-adapter-dataset-or-output>/submission.zip \
  --max-length 4096 \
  --batch-size 1 \
  --output /kaggle/working/trace_logprob_bit.jsonl \
  --csv /kaggle/working/trace_logprob_bit.csv \
  --summary-json /kaggle/working/trace_logprob_bit_summary.json
```

First audit batches:

1. `oracle_bit`: local oracle bit traces, `task_type=bit_manipulation`, limit 256.
2. `leevvin_tong_bit`: Tong bit traces, limit 256, keep `multiple_boxed`.
3. `bankoglu_bit`: compact hard bit traces, limit 256.
4. `sybyrr_symbolic`: symbolic beta txt traces, limit 128, require prompt/answer.
5. `llkh0a_formatted`: formatted train sample, limit 256.

## Output

JSONL row fields include:

- `id`, `source`, `task_type`, `method`, `answer`, `quality_flags`
- char/token lengths, truncation and prefix-mismatch flags
- if scored: `mean_logprob`, `min_logprob`, `worst_token_text`, `worst_context`

CSV is a flat subset. Summary JSON aggregates source/task/flags and, after scoring, worst rows.

## Interpretation

The first decision is not "which data has highest answer coverage." It is:

- Which trace style has the least damaging worst tokens?
- Are low-confidence tokens final boxed tokens, structural separators, bit/symbol glyphs, or arithmetic text?
- Does a public trace source improve hard-family content while worsening template stability?

Training should wait until this report identifies a concrete repair target.
