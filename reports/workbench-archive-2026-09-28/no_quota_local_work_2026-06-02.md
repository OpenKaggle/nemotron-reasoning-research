# No-Quota Local Work - 2026-06-02

## Current Remote State

- Team: `Jiayi Du` / `jahyee`
- Public rank from latest downloaded leaderboard: `348 / 3827` at 2026-06-02 16:00 UTC; ref `53293499` is reflected but did not change best score/rank.
- Best completed submission: `53215612`, public score `0.86`
- Completed submissions today include v3/v4 SFT/repack attempts; none beat `0.86`.
- Follow-up after this local-work pass: the all-MoE splice repack was pushed, audited, and submitted as Kaggle ref `53288769`; it completed with public score `0.85`.
- Final UTC-day slot was used on the conservative in-proj-only ablation, Kaggle ref `53293499`; it completed with public score `0.85`.
- Current UTC-day competition submissions used after `53293499`: 5 of 5. Do not submit again today unless quota unexpectedly remains and the user explicitly chooses to spend it.

## Local Work Completed

1. Fixed oracle report accounting.
   - `scripts/generate_oracle_traces.py` now treats reused `oracle=true` traces as oracle-only, not no-gold base.
   - Correct report values are back to:
     - no-gold base: `8364 / 9500 = 88.04%`
     - oracle training traces: `9056 / 9500 = 95.33%`

2. Rebuilt local trace package.
   - `artifacts/nemotron_oracle_training_package` was rebuilt after the report fix.
   - `oracle_reasoning_traces.jsonl` line count remains `9056`.

3. Added adapter audit tooling.
   - New script: `scripts/adapter_audit.py`
   - Supports adapter directories or `submission.zip` files.
   - Reports config fields, tensor categories, top shapes, missing/extra category shape, and optional cosine comparison against a reference adapter.
   - Sanity check saved at `reports/kienngx_self_moe_audit.json`; Kienngx-vs-Kienngx MoE category cosines are ~1.0.

4. Hardened the training kernel.
   - `notebooks/jahyee__nemotron-oracle-sft-r32/nemotron-oracle-sft-r32.py` now detects duplicate adapter load targets before training.
   - This should catch the suspected failure mode where many per-expert Kienngx tensors map into the same Unsloth batched parameter and silently overwrite each other.
   - The kernel also prints pre-training MoE LoRA stats (`mean_abs`, `slice_mean_abs_std`) after warm-start load.

5. Prepared all-MoE splice repack.
   - `notebooks/jahyee__nemotron-oracle-sft-r32-repack/nemotron-oracle-sft-r32-repack.py` now supports `RESTORE_MODE`.
   - Default `RESTORE_MODE=all_moe`: restore every `*.experts.*` LoRA tensor from Kienngx, keeping only non-MoE SFT deltas.
   - `RESTORE_MODE=broken_moe`: reproduce submission `53279382` partial splice behavior.
   - `RESTORE_MODE=none`: key/config rewrite only.

6. Prepared post-score diagnostics and ablations.
   - `notebooks/jahyee__nemotron-oracle-sft-r32-diagnostic` is a separate diagnostic kernel with `DIAGNOSTIC_ONLY=True`; it audits exact post-load tensor equality and exits before training.
   - Diagnostic kernel `jahyee/nemotron-oracle-sft-r32-diagnostic` version `1` was pushed, completed, and downloaded as `reports/diagnostic_load_mapping_v1.json`.
   - Diagnostic conclusion: Kienngx warm-start loads exactly (`12010/12010` source tensors into `12010` unique model targets), with no duplicate targets, missing tensors, shape mismatches, or value deltas. The v4 `0.52` failure is not explained by load mapping loss.
   - Added `scripts/build_adapter_ablation.py`.
   - Built and audited `artifacts/adapter_ablation_candidates/kienngx_plus_in_proj_lm_head/submission.zip`, which starts from Kienngx and keeps only `in_proj` + `lm_head` deltas from the all-MoE candidate. Audit: `reports/ablation_in_proj_lm_head_audit.json`.
   - Built and audited `artifacts/adapter_ablation_candidates/kienngx_plus_in_proj_only/submission.zip`, which starts from Kienngx and keeps only `in_proj.lora_A/B` deltas. Manifest: 46 tensors copied, 0 missing. Audit: `reports/ablation_in_proj_only_audit.json`.

## Validation Run

- Python compile check passed for:
  - `scripts/generate_oracle_traces.py`
  - `scripts/adapter_audit.py`
  - `scripts/build_training_package.py`
  - `scripts/reasoning_assets.py`
  - `notebooks/jahyee__nemotron-oracle-sft-r32/nemotron-oracle-sft-r32.py`
  - `notebooks/jahyee__nemotron-oracle-sft-r32-repack/nemotron-oracle-sft-r32-repack.py`

- Trace/package line counts:
  - `data/generated/reasoning_traces.jsonl`: `8364`
  - `data/generated/oracle_reasoning_traces.jsonl`: `9056`
  - `artifacts/nemotron_oracle_training_package/oracle_reasoning_traces.jsonl`: `9056`

## Next-Day Queue

1. Done: pushed and ran the repack kernel with default `RESTORE_MODE=all_moe`.
   - Kernel: `jahyee/nemotron-oracle-sft-r32-repack`
   - Version: `4`
   - Log: restored `11776` MoE-expert LoRA tensors from Kienngx with `0` missing.

2. Done: downloaded and audited output before submitting.

```bash
.venv/bin/python scripts/adapter_audit.py <temporary-path> \
  --reference submissions/kienngx_root \
  --json reports/all_moe_repack_v4_audit.json
```

3. Done: submitted the all-MoE splice after audit.
   - root zip layout is clean,
   - adapter config is Kienngx-compatible,
   - all MoE expert categories match Kienngx,
   - non-MoE deltas are intentional and auditable.
   - Submission ref: `53288769`
   - Current status: `COMPLETE`, public score `0.85`

4. Done: pushed and ran `notebooks/jahyee__nemotron-oracle-sft-r32-diagnostic`.
   - Kernel: `jahyee/nemotron-oracle-sft-r32-diagnostic`
   - Version: `1`
   - Result: load mapping passed exactly; prefer small-scope SFT over another all-module full run.

5. Done: spent the remaining UTC-day slot on the cleaner `kienngx_plus_in_proj_only` ablation.
   - Artifact: `artifacts/adapter_ablation_candidates/kienngx_plus_in_proj_only/submission.zip`
   - Submission ref: `53293499`
   - Current status: `COMPLETE`, public score `0.85`
   - Rationale: all tensors are identical/effectively identical to Kienngx except `in_proj.lora_A/B`, so the score will isolate whether that smallest SFT delta helps, is neutral, or hurts.
   - Conclusion: the isolated `in_proj` delta is net-negative relative to Kienngx (`0.85` vs `0.86`).

6. Latest leaderboard snapshot:
   - CSV timestamp: 2026-06-02 16:00 UTC
   - Total teams: 3827
   - `0.87`: 14 teams, ranks 1-14
   - `0.86`: 1289 teams, ranks 15-1303
   - Team `Jiayi Du` / `jahyee`: rank `348`, score `0.86`, submission count `9`

7. Do not resubmit known 0.84/0.85/0.86 packages.

8. Next direction: stop adjacent splice submissions and change the training recipe itself: lower LR, shorter schedule, validation-gated small-scope SFT, or explicit regularization back to Kienngx weights.

9. Prepared and pushed the next small-scope training kernel.
   - Local folder: `notebooks/jahyee__nemotron-oracle-sft-r32-inproj-lr5e-6-s40`
   - Kaggle kernel: `jahyee/nemotron-oracle-sft-r32-inproj-lr5e-6-s40`
   - Version: `1` pushed successfully at 2026-06-02 16:11 UTC.
   - URL: `https://www.kaggle.com/code/jahyee/nemotron-oracle-sft-r32-inproj-lr5e-6-s40`
   - Config: `IN_PROJ_ONLY=True`, `NUM_STEPS=40`, `LEARNING_RATE=5e-6`, `MOE_TIE_WEIGHTS=False`.
   - Local compile check passed.
   - Completed as Kaggle kernel version `1`; status confirmed `KernelWorkerStatus.COMPLETE`.
   - Downloaded output with `scripts/download_kernel_output_file.py`.
   - Because the stock Kaggle CLI repeatedly failed on SSL / non-resumable 3GB output download, added `scripts/download_kernel_output_file.py` for resumable single-file kernel-output downloads.
   - Preserved candidate artifact at `artifacts/training_candidates/nemotron_oracle_sft_r32_inproj_lr5e_6_s40_v1/submission.zip`; the temporary `<temporary-path>` copy was removed after preservation.
   - Zip integrity passed (`unzip -t`); root layout has only `adapter_model.safetensors` and `adapter_config.json`.
   - Audit: `reports/inproj_lr5e_6_s40_v1_audit.json`.
   - Audit conclusion: config and all 12,010 tensor categories match Kienngx structure; only `in_proj.lora_A/B` differ from Kienngx, with all other categories at `mean_abs_delta=0`.
   - New delta is much smaller than the failed `53293499` in-proj-only splice:
     - `in_proj.lora_A`: mean abs delta `7.97e-05`, 11.2% of old.
     - `in_proj.lora_B`: mean abs delta `7.71e-05`, 12.0% of old.
   - Do not submit its output today; current competition submissions are already 5/5 for the UTC day.

10. Next candidate after UTC reset / explicit decision:
   - `artifacts/training_candidates/nemotron_oracle_sft_r32_inproj_lr5e_6_s40_v1/submission.zip`
   - Rationale: clean tiny `in_proj`-only update; previous larger `in_proj` delta scored `0.85`, so this is a cautious neutral-or-slight probe rather than a high-confidence breakout.
   - Submitted after UTC reset as Kaggle ref `53315223` at 2026-06-03 05:31 UTC.
   - Current status: `PENDING` as of 2026-06-03 05:34 UTC.
   - Do not spend additional 2026-06-03 UTC submissions until this result completes.
