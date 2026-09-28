# Parallel Upload Sweep - 2026-06-07

## Goal

Use the full daily submission quota for `nvidia-nemotron-model-reasoning-challenge` as quickly as possible by launching a minimal closed-loop submission sweep from existing local candidate zips.

## Confirmed Remote State Before Launch

- Date: `2026-06-07`
- Competition: `nvidia-nemotron-model-reasoning-challenge`
- Previous latest completed submission remained from `2026-06-06`.
- `kaggle competitions submissions -c nvidia-nemotron-model-reasoning-challenge` showed no `2026-06-07` entries before launch.
- Historical best confirmed public score remained `0.86`.

## Launched Uploads

All commands were launched from:

`<local-kagglenvda>`

with:

`source .venv/bin/activate`

### Sweep 1

- Session id: `56600`
- File: `<local-kagglenvda>/artifacts/local_submission_sweep_2026-06-06/kienngx_tiny_inproj_alpha1_lmhead_alpha0p10/submission.zip`
- Message: `2026-06-07 sweep1 tiny in_proj alpha1 plus lm_head alpha0.10 probe`

### Sweep 2

- Session id: `36826`
- File: `<local-kagglenvda>/artifacts/local_submission_sweep_2026-06-06/kienngx_allmoe_inproj_alpha0p25/submission.zip`
- Message: `2026-06-07 sweep2 allmoe in_proj alpha0.25 quarter-scale probe`

### Sweep 3

- Session id: `91043`
- File: `<local-kagglenvda>/artifacts/local_submission_sweep_2026-06-06/kienngx_allmoe_inproj_alpha0p10_lmhead_alpha0p10/submission.zip`
- Message: `2026-06-07 sweep3 allmoe in_proj alpha0.10 plus lm_head alpha0.10 probe`

### Sweep 4

- Session id: `51356`
- File: `<local-kagglenvda>/artifacts/adapter_ablation_candidates/kienngx_plus_in_proj_only/submission.zip`
- Message: `2026-06-07 sweep4 kienngx plus in_proj only ablation full-strength`

### Sweep 5

- Session id: `86829`
- File: `<local-kagglenvda>/artifacts/adapter_ablation_candidates/kienngx_plus_in_proj_lm_head/submission.zip`
- Message: `2026-06-07 sweep5 kienngx plus in_proj and lm_head ablation full-strength`

## Current Status At Last Poll

- All five uploads were actively transferring bytes.
- None had yet reached the server response stage during the observed polling window.
- The bottleneck is local/network upload throughput, not packaging or CLI setup.

## Practical Next Check

When any upload completes or to re-check remote state:

```bash
source <local-kagglenvda>/.venv/bin/activate
kaggle competitions submissions -c nvidia-nemotron-model-reasoning-challenge
```

To poll a running upload session from this environment, inspect the corresponding session id listed above.
