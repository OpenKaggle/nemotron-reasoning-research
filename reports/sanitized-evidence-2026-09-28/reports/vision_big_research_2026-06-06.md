# Vision Big Research Sweep - 2026-06-06

Scope: local/remote state audit, public package and discussion sweep, and next-step research inference for pushing beyond the 0.86 cluster in `nvidia-nemotron-model-reasoning-challenge`.

## Verified State

- Workspace `<redacted-path> is not a git repository. Local/remote state is Kaggle kernels, datasets, submissions, reports, and cached artifacts rather than git branches.
- Latest competition submissions are unchanged: best remains `53215612` Kienngx public baseline at `0.86`, with `53315223` tiny low-LR in-proj SFT delta also `0.86`.
- Leaderboard download at `2026-06-06 07:49:01 UTC` shows `3969` teams. Our team `Jiayi Du` / `jahyee` is rank `333`, score `0.86`, submission count `10`.
- Score distribution: `0.89`: 1 team, `0.87`: 17 teams, `0.86`: 1380 teams, `0.85`: 480 teams.
- Remote main Kienngx-recipe kernel and repack kernel do not yet include the local 2026-06-06 packaging fix. Local scripts now force `submission.zip` to exactly `adapter_config.json` and `adapter_model.safetensors`.

## Original Research Thread Recovered

The original handoff framed the task correctly: this is an inverse-engineering problem, not a normal fine-tuning sweep. The goal is to escape the `0.86` crowd by finding a mechanism that adds a few robust correct public/hidden cases without damaging Kienngx's anchor behavior.

The strongest prior facts still hold:

- Kienngx is the safest anchor.
- Full oracle SFT and adapter surgery are net-negative or neutral.
- MoE tying/key layout was diagnosed, but fixing it did not make ordinary SFT work.
- Training loss and trace coverage are not enough; low-confidence tokens and formatting tails matter more.

## Public Intelligence

### Strongest Method Facts

- Huikang's progress-prize writeup says the high-value recipe is deterministic CoT, Tinker/SFT, bit manipulation, and maximizing minimum token logprob.
- His iteration loop explicitly inspects per-trace minimum logprob and worst-loss tokens, then rebalance-trains traces that remain low-confidence.
- He also calls out token-level representation and bit trace compression as future improvements.
- Donald/Kh0a-style discussion provides a dense reward blueprint: per-bit, per-letter, per-rate, per-scan, per-lock, per-encode rewards, plus contamination/thrash penalties.
- Shallow GRPO with only final answer/format reward has already been public and is not a breakout path.

### Public Packages / Adapters

- `llkh0a/nemotron-unsloth-sft-training-3-30-2` is the strongest public high-score lead because author/team `Kh0a` is currently `0.87`. The public notebook itself is old and may not equal the current submission, but its data design is valuable:
  - 3910 train rows.
  - Heavy family sampling: bit `1400`, text `1300`, numeric equation `600`, easy families `200` each, symbol transform only `10`.
  - Custom traces for bit and numeric equation.
  - SFT, not GRPO, `MAX_SEQ_LEN=3500`, 2 epochs, `lr=1e-4`, weighted final boxed-region loss.
- `kuangyicheng/nemotron-087-training` is a real recipe but author's leaderboard evidence is `0.86`, not verified `0.87+`.
- `penguin069/nemotron-dapo-05-06`, `foysalemonshanto/nemotron-cot-tong-submission`, and `profmansoor/nemotron-v34-path` are true large adapters but mapped leaderboard evidence is still `0.86`.
- `wethepeople918/sigilagi-nemotron-rank32-adapter` is config-only and not a usable adapter package.
- `NullSira` has the only verified `0.89`, but no public artifact was found.

### Public Data Assets

- Current local oracle trace coverage is `9056/9500`; remaining gaps are concentrated in bit manipulation and symbolic equation.
- `leevvin/nemotron-bitmanip-traces-pub` is immediately useful: Tong bit traces cover all 162 local bit oracle gaps, and answers match train.
- `bankoglu/hard-families-cot` gives a shorter bit-hard subset; useful as a compact alternative to long bit traces.
- `sybyrr/nemotron-symbolic-beta-trace` is the best symbolic-specific public trace pool, but must be robustly parsed and answer-filtered.
- `gdataranger/huikang-nemotron-nemo-sft-r32` is important because it exposes 14-class Huikang-style coverage including public-train-external/test-only categories. It should be audited before large training.
- `srivathsnatarajan/nemotron-enriched-train-cot` covers all ids but looks like generic short CoT; use for logprob comparison, not direct large-ratio training.

## Main Inference

The next step is not another full SFT phase. The research center should move to:

1. Build token/min-logprob audit on Kienngx and candidate traces.
2. Use the audit to find exact low-confidence tokens, not just low-scoring families.
3. Repair by family and trace style, beginning with bit and symbolic.
4. If using RL, use dense solver/verifier reward, not final answer reward.
5. Keep Kienngx behavior as the anchor; every experiment must prove it does not damage easy families and boxed formatting.

## Priority Experiments

### 1. Token / Min-Logprob Audit

Goal: reproduce the Huikang diagnostic loop locally/Kaggle-side.

Inputs:
- Kienngx root adapter.
- Local oracle traces.
- `leevvin` bit traces.
- `sybyrr` symbolic beta traces after answer filtering.
- Small sample from Huikang/Nemo SFT corpus.

Output:
- Row-level min logprob, mean logprob, worst token, token context, family/method/source aggregates.
- A report ranking trace styles by worst-token risk.

Why first: it decides whether bit repair, symbolic repair, short boxed-answer SFT, or dense RL is the correct next move.

### 2. Bit Gap Repair

Goal: target the 162 bit cases missing from local oracle using `leevvin` Tong traces and compact `bankoglu` traces.

Approach:
- Do audit first.
- Prefer trace compression before training if worst tokens are formatting/long-expression tokens.
- Train only a small targeted delta or low-ratio mix, with easy-family anchors to prevent forgetting.

Submission gate:
- Zip root exactly two files.
- Adapter audit vs Kienngx.
- No submission unless audit suggests lower tail risk than Kienngx/tiny-inproj.

### 3. Symbolic / Cipher-Digit Repair

Goal: reduce the symbolic tail without training on gold-conditioned hallucinations.

Approach:
- Parse and answer-filter `sybyrr` symbolic beta.
- Separate no-gold valid traces from gold-conditioned oracle traces.
- Design scan/lock/encode DSL or reward fields before training.

Risk:
- The symbolic family is where train-answer-conditioned traces most easily teach fake reasoning. A no-gold verifier must remain in the loop.

### 4. Dense Verifier GRPO / DAPO

Goal: only after audit and trace DSL exist, run a minimal dense-reward experiment.

Approach:
- Fork a public GRPO trainer only for plumbing.
- Reward must come from `scripts/reasoning_assets.py` style verifier stages.
- Start from Kienngx or tiny-inproj anchor.
- Use small family-focused prompts, `num_generations=4`, short `80-120` steps, low LR, frozen/highly restricted update surface.

This is the bold top-5/top-3 route, but only if the reward is structural. Plain final-answer GRPO is not worth primary effort.

## Operational Next Step

The safest immediate action is:

1. Push/run the clean repack only as an environment gate if a submission-ready Phase 1a artifact is still desired.
2. In parallel, implement `scripts/trace_logprob_audit.py` or a Kaggle audit kernel.
3. Audit Kienngx on local oracle, leevvin bit, sybyrr symbolic, and Huikang/Nemo samples.
4. Decide the first repair corpus from the actual worst-token report.

Do not submit another 0.86-adjacent package until this audit gives a concrete non-duplicate rationale.

## Multi-Agent Update - 2026-06-06 16:30 CST

Local and remote status were rechecked:

- The workspace is still not a git repository; remote drift means Kaggle kernels/datasets/submissions.
- Remote submissions remain at `10`; best/latest remain `53215612` and `53315223`, both `0.86`.
- `kaggle kernels status` is unreliable today and returned server errors for key kernels; `kernels logs`, `kernels pull -m`, and output listing are the usable fallbacks.
- Local-only packaging fixes still have not been pushed. Remote `jahyee/nemotron-kienngx-recipe-repack` and `jahyee/nemotron-kienngx-recipe-sft-on-oracle-9056` still package extra tokenizer/template files; local scripts now enforce exactly `adapter_config.json` and `adapter_model.safetensors`.
- One old in-proj kernel source mismatch exists: local `KAGGLE_DATASET` uses `lr5e-6`, remote uses `lr5em6`. Do not push that kernel until deciding which dataset slug is canonical.

New public artifacts changed the next-step reasoning:

- `gdataranger/huikang-nemotron-nemo-sft-r32` is now first-class audit data, not just a training corpus. It exposes an 820-row pre-tokenized validation split and 14 categories, including test-only-style subclasses such as `matching`, `splitting`, `concatenation`, `spelling`, `lstrip`, and cryptarithm variants. The audit code now supports `nemo_jsonl` pre-tokenized rows directly.
- `kishanvavdara/nemotron-reasoning-traj` was downloaded to `tmp_kaggle_recon/datasets/kishanvavdara_nemotron_reasoning_traj/nemotron_traj.csv`. It has all 9500 train ids and prompts, but only 3571 generated answers exactly match train. It is a strong contrastive/DPO/error-trajectory corpus, not a positive SFT corpus. Bit has only 128 true generations and symbolic-equation has only 2 true generations, which makes it especially useful for detecting bad long guessing traces.
- `kuangyicheng/nemotron-087-training` is a real continuation recipe over Huikang adapter v27 plus simple synthetic traces, but it is not verified as a 0.87+ public score. Treat it as evidence for short continuation and config/key hygiene, not as a package to copy.
- `dhanushbe06/huikang-pure` reinforces category filtering: it decodes Huikang tokens and filters equation-transform subclasses, suggesting noisy/hard equation traces should be isolated before training.
- `mohamedamr992/cluster-experts-by-routing` is a useful MoE routing diagnostic. Its implication is not dynamic routing at inference, but family-aware audit/merge decisions based on actual expert usage.
- `krrishk0406/nemotron-serious-run-expanded` was downloaded and is the strongest new public engineering package. It exposes Huikang-style reasoners/augmenters, 14-category corpus manifests, category token caps, an 8% holdout split, and a hard-tail oversampling mechanism keyed by prior min-logprob. Its Kaggle runner is too heavy for the immediate path (`2600` main QLoRA steps plus `300` hard-tail steps), but its selection logic is directly useful after our audit.
- `wethepeople918/nemotron-rank32-corpus` was downloaded as a larger 17,963-row 14-category corpus. It should be treated as another Huikang-style corpus candidate and deduped/audited before use.

Implementation status:

- `scripts/trace_manifest.py` now builds the first strict GPU audit manifest: 1356 rows across oracle, llkh0a, Donald, Bankoglu, Sybyrr beta, Srivath, and Tong. Default filters require train join, prompt/answer consistency, final boxed match, and exclude Tong/Sybyrr boxed mismatches and oracle-true rows from the default training-oriented sample.
- `notebooks/jahyee__nemotron-trace-logprob-audit/` now contains a script Kaggle GPU audit kernel. It discovers model/adapter/data, supports pre-tokenized Huikang `nemo_valid.jsonl`, defaults to 8192 max length, and writes row-level logprob plus worst-token aggregates without logits.

Updated inference:

1. First remote GPU run should be an audit run, not a training run. Use a tiny smoke batch first, then the 1356-row manifest/Huikang-valid slices if loading is stable.
2. The first repair decision should be based on p1/p5/min-logprob and worst-token context, not mean loss.
3. Bit repair remains the most likely 0.87 route if Donald/Bankoglu compressed traces improve tail confidence without multiple-boxed instability.
4. The 0.89/top-3 route is now a three-part stack: Huikang 14-class tail audit, symbolic/cipher-digit verifier DSL, and dense reward or generate-verify-distill. Ordinary SFT on all public traces is no longer a serious primary plan.
5. If the first audit produces a stable low-confidence set, borrow `krrishk0406`'s hard-tail selection recipe: category caps, held-out category-balanced audit, and a small oversample of worst min-logprob rows. Do not copy its full QLoRA schedule until a smaller Kienngx-anchored repair proves positive.

## Multi-Agent Update - 2026-06-06 17:30 CST

Subagent conclusions converged on three complementary routes rather than one narrow path:

1. Token/min-logprob active selection remains the immediate critical path. External active-learning/OHEM/focal-loss literature supports selecting examples by uncertainty or loss tail, and public Huikang/Krrish traces independently point to the same min-logprob hard-tail loop. The first remote GPU job should therefore be audit, not training.
2. Verifier/DPO is the second path. `kishanvavdara/nemotron-reasoning-traj` is useful as rejected trajectories, not positive SFT data. Local smoke scripts now prove bit pairs are easy to build from verified manifest positives versus Kishan failures.
3. Adapter arithmetic is the third path. Public "ensemble" notebooks mostly repackage one adapter, but correct delta-space LoRA soup/SVD and routing-informed expert graft are plausible engineering routes, especially if a verified `0.87` adapter such as Kh0a/llkh0a can be inspected.

New local engineering artifacts:

- Added `scripts/verifier_rewards.py`: pure dense reward extraction over `(prompt, completion, answer, task_type)`, including boxed answer parsing, exact/partial rewards, binary/text/numeric format checks, and Kishan-friendly failure flags.
- Added `scripts/build_verifier_dataset.py`: emits reward JSONL rows from strict manifest positives and Kishan trajectories.
- Added `scripts/build_dpo_pairs.py`: pairs exact verified positives with same-id Kishan failures for DPO/IPO/ORPO-style experiments.

Smoke results:

- `reports/verifier_reward_smoke.jsonl`: 200 bit positive rows, all exact, mean reward `1.0`; 98 have `multiple_boxed`, confirming this flag is diagnostic rather than an automatic drop.
- `reports/verifier_reward_kishan_bit_smoke.jsonl`: 200 Kishan bit rows, only 16 exact. Median reward is `0.06`, mean reward `0.15435`; dominant flags are `wrong_final`, `wrong_binary_length`, and `malformed_binary`.
- `reports/dpo_pairs_smoke.jsonl`: 100 bit pairs. Median margin `0.94`, mean margin `0.9141`. Chosen sources: Donald 41, Bankoglu 19, Tong 16, llkh0a 13, oracle 6, Srivath 5. This is strong enough to justify a tiny preference-training probe later, but only after logprob audit gives a holdout gate.

Public radar update:

- Leaderboard shape is unchanged around `2026-06-06T08:37:13Z`: `0.89` has 1 team, `0.87` has 17 teams, `0.86` has 1380 teams.
- `llkh0a` remains the only clearly verified public `0.87` recipe lead; its old public notebook may not equal the current private submission, but the sampling design is still valuable.
- `quincyqiang/public-models-nemotron` is a public 3.55 GB Kaggle model tied to a current `0.87` team member. CLI metadata shows version 1 ready, but no config/description yet. Next low-cost step is file metadata inspection before any large download.
- `dasmwjdasd/nemotron-algorithmic-cot-full-sft` is tied to a `0.87` team member but the pulled notebook is only an environment/input probe, not a recipe.

Generator/verifier update:

- Reliable synthetic generation is feasible for `bit-hard`, `numeric_equation`, safe `cryptarithm`, `text_cipher`, and Huikang-style test-only transformations such as concat/splitting/lstrip/spelling.
- Do not use gold-conditioned `cryptarithm_gold_oracle` as a "new problem" generator. It remains SFT-only training data for known train prompts.
- The bold synthetic path should be verifier-gated: write latent rule, seed, prompt hash, verifier answer, uniqueness status, and reject any generated prompt/demo/query exact hash colliding with train or public traces.

Adapter-merge update:

- SVD/task-arithmetic soup is only meaningful in delta space: `Δ = Σ w_i * scale_i * B_i A_i`, then SVD back to rank 32. Directly averaging LoRA A/B is not basis-invariant.
- First low-cost experiment should be Kienngx + Hammad/Huikang small-weight soup while preserving Kienngx MoE experts and `lm_head`; validate by audit/holdout before submission. This is secondary to the logprob audit because Hammad's local public score was only `0.84`.
- More ambitious route: use MoE routing fingerprints by family/layer/expert to graft deltas only into high-frequency experts for a family. This needs GPU diagnostics and careful non-target regression checks.

Operational change:

- The audit kernel default has been narrowed to a safe remote smoke: `AUDIT_JOB_FILTER=nemo_valid`, `AUDIT_LIMIT_PER_JOB=1`, `AUDIT_MAX_LENGTH=8192`, batch size 1. This is not a competition submission.
- Version 1 exposed a Kaggle script packaging issue: only the metadata `code_file` was available under `/kaggle/src`, so `trace_logprob_audit.py` was missing. Fixed by adding `scripts/bundle_trace_logprob_audit_kernel.py`, generating `notebooks/jahyee__nemotron-trace-logprob-audit/nemotron-trace-logprob-audit-bundled.py`, and changing `kernel-metadata.json` to use that bundled single-file code. Version 2 was pushed with the fix.

## Multi-Agent Update - 2026-06-06 17:23 CST

Current verified state:

- Workspace is still not a git repository; local/remote drift is Kaggle artifacts and local files, not branches.
- Competition submissions are unchanged at `10`. Latest submission `53315223` and anchor `53215612` are both still public `0.86`.
- Leaderboard top snapshot now has `NullSira` alone at `0.89` and `18` teams at `0.87`; the new `0.87` addition versus the earlier report is `coreforged`.
- `quincyqiang/public-models-nemotron` is a real public `0.87`-adjacent adapter model (`3.55GB`, only adapter files plus tokenizer). It is useful later for delta/SVD comparison, but not worth downloading before the audit gives a clear target.

Audit status:

- `jahyee/nemotron-trace-logprob-audit` v2 succeeded. The smoke run loaded the base model plus Kienngx adapter and audited one Huikang `nemo_valid` pre-tokenized row.
- The smoke row had mean logprob `-0.2234` but min logprob `-19.2155`; worst token was ` 【` in a symbolic/operator context. This validates the key research hypothesis: mean loss can look fine while rare tokens expose the hard tail.
- The v2 outputs were pulled into `reports/remote_trace_logprob_audit_v2_smoke/`.
- The audit helper was patched so canonical manifest rows preserve original `source` and `quality_flags` instead of becoming only `oracle_jsonl:<file>`.
- Kernel v3 was pushed with a small expanded default: `AUDIT_JOB_FILTER=local_manifest_first_gpu_audit`, `AUDIT_LIMIT_PER_JOB=64`, `AUDIT_MAX_LENGTH=8192`. This is still a diagnostic GPU run, not a competition submission.
- Kernel v3 failed quickly because Kaggle script mode again did not include sibling data files; `trace_manifest_first_gpu_audit.jsonl` was missing, so `job_filter=local_manifest_first_gpu_audit` discovered zero jobs. This was not a model/loading failure.
- A stratified 64-row slim manifest was generated at `notebooks/jahyee__nemotron-trace-logprob-audit/trace_manifest_first_gpu_audit_slim64.jsonl` and embedded into the bundled script. Full 1356-row manifest should become a dataset if needed; embedding the whole 4MB JSONL made Kaggle reject the kernel save request.
- Kernel v4 was pushed successfully with the slim embedded manifest. `kernels pull -m` confirms the remote code is the 241KB slim64 script and defaults to `local_manifest_first_gpu_audit`/`64`. As of the latest check, output files still show the v2 smoke artifacts and `kernels status` still returns Kaggle `500`; monitor logs/output rather than treating this as a code failure.
- Kernel v4 progressed past discovery/model load and read all 64 slim rows, proving the slim manifest fix worked. It then failed in tokenization: all rows showed `prefix_mismatch=64`, token lengths collapsed to `2`, and `audit_model` hit `TypeError: 'str' object cannot be interpreted as an integer`. The helper now renders chat templates as text first and then encodes to explicit integer IDs, with a normalization fallback for BatchEncoding/tensor/list outputs. Kernel v5 was prepared from this fix.

New engineering artifact:

- Added `scripts/select_hard_tail_from_audit.py`. It consumes audit JSONL, optionally joins back to the canonical trace manifest, splits a category-balanced holdout, and emits a balanced low-min-logprob hard-tail JSONL/CSV plus summary.
- Added `scripts/adapter_svd_soup.py`. It implements a safe delta-space LoRA soup baseline: shape-gated adapter loading, QR-SVD recompression of `sum_i weight_i * scale_i * B_i @ A_i`, dry-run compatibility reports, and optional root-clean `submission.zip`. Smoke dry-run on Kienngx + Hammad layer-0 correctly marks `in_proj` incompatible and `out_proj` SVD-compatible.
- Added `scripts/reward_calibration.py`. It compares the local verifier reward against simple sequence/cosine and Banwait-style composite rewards on DPO pairs or positive-vs-Kishan rows. Smoke on 20 bit DPO pairs shows all three rewards perfectly separate chosen/rejected, which means the first 20 pairs are too easy; the useful next calibration is 500-1000 mixed hard pairs.
- Smoke output from the v2 one-row audit:
  - `reports/hard_tail_smoke.jsonl`
  - `reports/hard_tail_smoke.csv`
  - `reports/hard_tail_smoke_summary.json`
  - `reports/adapter_svd_soup_smoke.json`
  - `reports/reward_calibration_smoke.json`

Notebook/GitHub conclusions now converge more strongly:

- `llkh0a` remains the strongest verified public `0.87` recipe lead, but the useful transferable pieces are sampling ratio, short max length, custom bit/numeric traces, and boxed-region loss; not blind reuse of its full trace mix.
- `sybyrr/run4` gives a concrete masked-token SFT loop over `tokens + mask` and a MoE tied-gradient idea; useful for a future controlled training probe, not before audit.
- `hammad` and Huikang conversion code are the cleanest SVD sources. `hammad` uses correct delta-space QR-SVD, but its `in_proj` shape differs from Kienngx, so the first soup must shape-gate and keep `in_proj` from the anchor.
- `lopure` has a useful eval-driven soup control flow but unsafe merge math in places; do not copy its A/B averaging or column truncation.
- Public discussions warn that blind GRPO and large synthetic SFT still sit at `0.86`. DPO/GRPO should follow audit-discovered hard failures and use structural verifier rewards, not final answer only.
- RL/Verifier notebook sweep did not find a proven public breakout GRPO recipe. The strongest actionable pieces are DAPO config discipline (`loss_type=dapo`, group-scaled rewards, `num_generations=2-4`, `beta≈0.04-0.08`, low LR), reward shaping, and a cache-performance warning. The first executable preference path should be bit-only DPO/IPO from verified positives versus Kishan failures; DAPO should be only a 32-64 prompt smoke after audit.

Next inference:

1. Pull v3 audit outputs as soon as available. If stable, run a larger 256-384 row manifest audit and a separate Huikang `nemo_valid` slice.
2. Feed the audit JSONL into `scripts/select_hard_tail_from_audit.py` and inspect p1/p5, worst-token contexts, and source/family composition.
3. Pick the first repair path from evidence:
   - bit tail: Donald/Bankoglu/Tong compressed traces plus anchor-preserving micro-SFT or DPO.
   - symbolic/operator tail: verifier DSL/generate-verify-distill before training.
   - formatting tail: boxed-region loss or shorter traces rather than new reasoning data.
4. Keep adapter SVD soup as a secondary branch. First candidate should be Kienngx + small Hammad residual, shape-compatible non-`in_proj` keys only, validated by audit/holdout before any submission.
5. Do not spend submission quota until a candidate improves hard-tail audit without hurting anchor/easy-family behavior.

## Continuation Update - 2026-06-06 19:20 CST

Current local/remote state:

- Competition submissions are unchanged. Best remains Kienngx `53215612` and tiny in-proj `53315223`, both public `0.86`; no submission quota was spent in this continuation.
- Latest leaderboard snapshot `2026-06-06T11:12:57 UTC`: 3983 teams, our team rank `334`, score `0.86`, submission count `10`. Distribution: `0.89`: 1, `0.87`: 17, `0.86`: 1386.
- v6 content-tail outputs are preserved at `reports/remote_trace_logprob_audit_v6_content64/`. Standard content decision is mixed/inspect: `bit_repair=0.491`, `symbolic_operator=0.461`, `formatting_repair=0.393`.
- A `no-multiple-boxed` v6 diagnostic also stays mixed/inspect, with bit only modestly ahead. This rules out immediate training from the 64-row slice.
- A 384-row stratified manifest has been generated and embedded for v7 audit: oracle 96, llkh0a 80, sybyrr_beta 64, Donald 56, Bankoglu 32, Srivath 32, Tong 24.
- Kaggle rejected the first 384-row push because the generated single-file script was too large. `scripts/bundle_trace_logprob_audit_kernel.py` now zlib/base64-compresses the embedded manifest, reducing code size to about 272 KB. Kernel v7 pushed successfully; remote source hash matches local. Await v7 outputs.

Subagent convergence:

1. `performance` / wheels / package datasets are infrastructure, not a scoring path. `mayukh18/nemotron-packages`, `llkh0a/rtx-wheels`, and similar Blackwell wheels help run code but do not change what to train.
2. `llkh0a/nemotron-unsloth-sft-training-3-30-2` remains the strongest public `0.87` lead. Treat it as an adapter/output to audit, not as a package to submit blindly.
3. Krrish-style category caps and hard-tail oversampling are directly useful after v7, but its full heavy QLoRA schedule is not the next step.
4. Kishan trajectories remain negative/preference data. Use them for bit/symbolic DPO or verifier calibration only after v7 identifies a dominant family.
5. The bold route is not ordinary SFT. It is logprob hard-tail selection -> verifier/DPO or generate-verify-distill -> small Kienngx-anchored update with holdout audit.

Priority now:

1. Pull/process v7 as soon as Kaggle updates outputs.
2. If v7 tail is bit-dominant: build 500-1000 bit DPO/IPO pairs from Donald/Bankoglu/Tong/verified positives versus Kishan failures.
3. If v7 tail is symbolic/operator-dominant: build a no-gold symbolic verifier/distill corpus; do not blind-SFT gold-conditioned symbolic traces.
4. If v7 tail is formatting/boxed-dominant: normalize trace format and re-audit before any training.
5. If v7 remains mixed: run a separate Huikang `nemo_valid`/14-class audit and inspect `llkh0a` full adapter/output; keep adapter arithmetic as a secondary dry-run only.
