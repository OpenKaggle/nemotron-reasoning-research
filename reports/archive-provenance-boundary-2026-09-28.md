# Remaining archive provenance boundary

This note records the disposition of the remaining research archive after the
source and artifact releases. It is intentionally a provenance summary, not a
dataset mirror: it contains no competition rows, prompts, labels, answers,
predictions, completions, token traces, absolute local paths, credentials, or
third-party file contents.

## Publicly retrievable first-party material

| Material | Public location and receipt | Status |
| --- | --- | --- |
| Low-learning-rate, 40-step in-projection LoRA output | [OpenKaggle Nemotron Model and Submission Artifacts](https://www.kaggle.com/datasets/jahyee/openkaggle-nemotron-model-submission-artifacts): `oracle_sft_r32_inproj_lr5e_6_s40_v1.submission/adapter_model.safetensors` and `adapter_config.json` | Kaggle status `ready`; the file listing reports 3,554,384,888-byte adapter and 1,114-byte config. The local 3,275,791,339-byte source ZIP matches the released `SHA256SUMS` entry: `cc5dbfca7c03c13175d464716aebdb8798aabd19643b30eff21ef8595de8083e`. |
| Five small sweep variants | Same artifact dataset: five `*.override.safetensors` files, released `SHA256SUMS`, and `apply_override_patch.py` | Public compact reconstruction path; the base anchor itself is not mirrored. |
| First-party utility source | [`scripts/`](../scripts) | The four utility copies retained in the handoff archive (`adapter_audit.py`, `adapter_svd_soup.py`, `build_adapter_ablation.py`, and `download_kernel_output_file.py`) are byte-identical to the public copies already in this repository. No duplicate upload is needed. |
| Scalar model-structure audits | [`reports/`](.) | The reviewed audit JSON already in this repository is aggregate/model-structure only, with no record-level competition content. |

The artifact dataset is the delivery surface for eligible binaries. This source
repository remains source-only; see [ARTIFACTS.md](../ARTIFACTS.md) for the
restore boundary and [DATA_SOURCES.md](../DATA_SOURCES.md) for input rules.

## Deliberately not republished

| Archive group | Why it is excluded | Public-safe alternative |
| --- | --- | --- |
| Full handoff bundle | It mixes duplicate first-party source with row-level evaluation files and operational notes. | Use the already released scripts and reviewed scalar audits above. |
| General report cache | It contains a mixture of aggregate receipts, local paths, row-level evaluator exports, and trace-oriented files. | The reviewed scalar audits and `evaluation_aggregate.json` are the public record. |
| Public-radar cache | It is a local cache of other authors' notebooks, datasets, and metadata. Public visibility of a source does not grant this project the right to mirror it. | Retrieve a cited notebook or dataset from its original publisher. |
| External cache | It includes copied third-party source and dataset material. | Follow the upstream publisher and license. |
| Kaggle reconstruction cache | It contains third-party and competition-adjacent downloaded corpora, including generated records and training-shaped tables. | Obtain permitted inputs from their original source after accepting its terms. |

## Verification and future additions

Before a future artifact enters a public release, it must have an identified
author and license, a declared relationship to any official competition input,
a SHA-256 and size, a permitted retrieval location, and a clean restore check.
Derived outputs may be released only when they do not redistribute protected
organizer or third-party bytes. A raw prompt, label, answer, completion, token
trace, or copied upstream checkpoint fails this gate even when it appears in a
locally generated report.

This separation preserves reproducibility where it is permitted: source and
compact first-party deltas are public, while readers fetch official inputs and
third-party dependencies from their own authoritative locations.
