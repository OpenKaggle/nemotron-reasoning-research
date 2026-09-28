# Minimal Remote Loop - 2026-06-07

## Remote State

- Workspace: `<local-kagglenvda>`
- Competition: `nvidia-nemotron-model-reasoning-challenge`
- Latest confirmed remote submission remains `53423077`.
- Best confirmed public score remains `0.86`.
- No new submission ref was created for the attempted `kienngx_tiny_inproj_alpha1_lmhead_alpha0p10` upload.

## 2026-06-06 Submitted Slots

The remote submissions table shows five completed submissions on `2026-06-06`:

| ref | description | public |
|---|---|---|
| 53423077 | tiny `in_proj` alpha1.5 | 0.86 |
| 53422621 | Huikang 087 wrapper | 0.85 |
| 53422620 | low-lr `in_proj` | 0.86 |
| 53422448 | tiny `in_proj` alpha0.5 | 0.85 |
| 53421959 | half-scale tiny `in_proj` | 0.86 |

Competition rules confirm: maximum five submissions per day.

## Attempted But Not Accepted

Candidate:

`<local-kagglenvda>/artifacts/local_submission_sweep_2026-06-06/kienngx_tiny_inproj_alpha1_lmhead_alpha0p10/submission.zip`

Message:

`local sweep 2026-06-07: Kienngx safe tiny in_proj alpha1 plus small lm_head alpha0.10 probe`

Result:

- Upload reached `100%`.
- Kaggle `CreateSubmission` returned `400 Client Error: Bad Request`.
- A follow-up remote submissions query showed no new ref.
- Root cause is most likely daily quota exhaustion, not a broken zip.

Package checks:

- Zip contains exactly `adapter_config.json` and `adapter_model.safetensors`.
- Zip size is about `3.05G`, matching prior accepted adapter-package scale.
- Manifest: known-neutral tiny `in_proj` alpha1 plus very small `lm_head` alpha0.10 probe.

## Next Submission When Quota Refreshes

First candidate to submit after daily quota refresh:

`<local-kagglenvda>/artifacts/local_submission_sweep_2026-06-06/kienngx_tiny_inproj_alpha1_lmhead_alpha0p10/submission.zip`

Command:

```bash
<local-kagglenvda>/.venv/bin/kaggle competitions submit \
  -c nvidia-nemotron-model-reasoning-challenge \
  -f <local-kagglenvda>/artifacts/local_submission_sweep_2026-06-06/kienngx_tiny_inproj_alpha1_lmhead_alpha0p10/submission.zip \
  -m "local sweep 2026-06-07: Kienngx safe tiny in_proj alpha1 plus small lm_head alpha0.10 probe"
```

Immediately after submit:

```bash
<local-kagglenvda>/.venv/bin/kaggle competitions submissions \
  -c nvidia-nemotron-model-reasoning-challenge
```

## Decision

Do not spend time uploading another 3GB candidate before the quota window refreshes. The minimal loop is closed for now: remote state checked, attempted candidate rejected by quota, and the next submission command is fixed.
