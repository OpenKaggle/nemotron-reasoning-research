# Continuation Checkpoint - 2026-06-06 19:20 CST

## Current State

- Workspace `<redacted-path> is not a git repository. Local/remote drift is tracked through reports, notebooks, datasets, kernels, and submissions.
- Competition submissions are unchanged. Best remains Kienngx `53215612` at public `0.86`; tiny in-proj `53315223` is also `0.86`. No submission quota was spent in this continuation.
- Latest leaderboard snapshot downloaded at `2026-06-06T11:12:57 UTC`: 3983 teams, our team rank `334`, score `0.86`, submission count `10`. Distribution: `0.89`: 1, `0.87`: 17, `0.86`: 1386, `0.85`: 485.

## What Changed Locally

- Pulled v6 content-tail audit outputs into `reports/remote_trace_logprob_audit_v6_content64/`.
- Generated v6 hard-tail views:
  - `reports/hard_tail_v6_64_content_summary.json`
  - `reports/hard_tail_v6_64_content_decision.md`
  - `reports/hard_tail_v6_64_content_nomultibox_summary.json`
  - `reports/hard_tail_v6_64_content_nomultibox_decision.md`
- Updated `scripts/bundle_trace_logprob_audit_kernel.py` so embedded manifests are zlib/base64 compressed. This fixed Kaggle 400 push rejection for the 384-row audit bundle.
- Built the current embedded 384-row audit manifest at:
  - `notebooks/jahyee__nemotron-trace-logprob-audit/trace_manifest_first_gpu_audit_slim384.jsonl`
  - copied into `trace_manifest_first_gpu_audit_slim64.jsonl` for the existing bundler default.
- Regenerated `notebooks/jahyee__nemotron-trace-logprob-audit/nemotron-trace-logprob-audit-bundled.py`; local and remote v7 source hash match:
  - `e8a49443a90e0b8705d95aeffa86a5e42afc3317fc30f6f6e4b7c73174ca3fb5`
- Updated:
  - `reports/research_summary.md`
  - `reports/vision_big_research_2026-06-06.md`

## v6 Decision

Standard content-tail v6 remains mixed:

- `bit_repair`: 0.491
- `symbolic_operator`: 0.461
- `formatting_repair`: 0.393
- chosen path: `inspect_manually`

No-multiple-boxed diagnostic also remains mixed:

- `bit_repair`: 0.355
- `symbolic_operator`: 0.307
- `formatting_repair`: 0.279
- chosen path: `inspect_manually`

Interpretation: v6 is not enough evidence to train. The correct next step is the v7 384-row audit.

## Remote v7 Status

- Kernel `jahyee/nemotron-trace-logprob-audit` version 7 was pushed successfully at about `2026-06-06 11:11 UTC`.
- `kernels pull -m` confirms the remote source matches local generated code.
- As of the latest check, `kernels files` still shows v6 output timestamps (`10:50 UTC`), `kernels logs` is empty, and `kernels status` returns Kaggle `500`. Treat this as pending/stale Kaggle status, not a model or script failure yet.

## Next Commands

Poll v7 output:

```bash
env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy \
<redacted-path> kernels files jahyee/nemotron-trace-logprob-audit
```

When v7 output timestamps update, pull and process:

```bash
rm -rf <redacted-path>
env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy \
<redacted-path> kernels output jahyee/nemotron-trace-logprob-audit \
  -p <redacted-path> -o

rm -rf reports/remote_trace_logprob_audit_v7_content384
mkdir -p reports/remote_trace_logprob_audit_v7_content384
cp <redacted-path> reports/remote_trace_logprob_audit_v7_content384/

<redacted-path> scripts/select_hard_tail_from_audit.py \
  --audit reports/remote_trace_logprob_audit_v7_content384/trace_logprob_audit.jsonl \
  --manifest reports/trace_manifest_first_gpu_audit.jsonl \
  --score-field content_min_logprob \
  --output reports/hard_tail_v7_384_content.jsonl \
  --output-csv reports/hard_tail_v7_384_content.csv \
  --summary-json reports/hard_tail_v7_384_content_summary.json \
  --holdout-output reports/hard_tail_v7_384_content_holdout.jsonl \
  --max-rows 128 \
  --per-task 24 \
  --drop-flag truncated

<redacted-path> scripts/decide_repair_path_from_hard_tail.py \
  --summary reports/hard_tail_v7_384_content_summary.json \
  --output-json reports/hard_tail_v7_384_content_decision.json \
  --output-md reports/hard_tail_v7_384_content_decision.md
```

## Priority After v7

1. Bit-dominant: build bit DPO/IPO pairs from verified Donald/Bankoglu/Tong/oracle positives versus Kishan failures.
2. Symbolic-dominant: build no-gold symbolic verifier/distill corpus; do not blind-SFT gold-conditioned traces.
3. Formatting-dominant: normalize trace format and re-audit before training.
4. Mixed: run Huikang `nemo_valid`/14-class audit and inspect `llkh0a` full adapter/output as a secondary branch.
