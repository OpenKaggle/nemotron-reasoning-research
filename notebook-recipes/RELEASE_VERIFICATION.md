# Notebook source release verification

This receipt documents the source-only review completed on 2026-09-28 for the
first-party Nemotron notebook recipes.  It is intentionally a source release:
it neither contains organizer data nor provides an execution environment,
weights, generated submissions, trace rows, credentials, or cached packages.
Acquire every input independently under the original provider's terms.

## Checked first-party Python recipes

The nine files below were compared byte-for-byte with their local first-party
counterparts before this receipt was added.  SHA-256 is shown for the checked
source file.

| Public path | SHA-256 |
| --- | --- |
| `recipes/kaggle-kernels/jahyee__nemotron-huikang-087-submit/huikang-087-submit.py` | `1e702245d849840b71721e3e0dc88f78b8a7bacdfa979faa94d0d6af76832bf5` |
| `recipes/kaggle-kernels/jahyee__nemotron-kienngx-recipe-oracle/nemotron-kienngx-recipe-oracle.py` | `03463ea8c262bf1a5b6aaf2b1c3567d4ff40fbf61873539b4cca9f710009f717` |
| `recipes/kaggle-kernels/jahyee__nemotron-kienngx-recipe-repack/nemotron-kienngx-recipe-repack.py` | `3266b8b8fffd2392e22f15ec70878180acb2097ad76527b34090b299165e404a` |
| `recipes/kaggle-kernels/jahyee__nemotron-oracle-sft-r32-diagnostic/nemotron-oracle-sft-r32-diagnostic.py` | `ed4afe695519c3cf7b067872a1dba60f6ac53f7473e72a784790678d92fe4d4a` |
| `recipes/kaggle-kernels/jahyee__nemotron-oracle-sft-r32-inproj-lr5e-6-s40/nemotron-oracle-sft-r32-inproj-lr5e-6-s40.py` | `5e77c2876ddaf40d2ff78a7f36674bc92eb07a41ec93132ed546c91e97ea0e18` |
| `recipes/kaggle-kernels/jahyee__nemotron-oracle-sft-r32-repack/nemotron-oracle-sft-r32-repack.py` | `9156129558fd096f3adbbca1ca4949eec0efedc1ae3016e81a9cc674c1a6ccd2` |
| `recipes/kaggle-kernels/jahyee__nemotron-oracle-sft-r32/nemotron-oracle-sft-r32.py` | `05608548f2fbb10a0b7c3fdcf675aa277545eb8fe4207f1b37a066c7826ba904` |
| `recipes/kaggle-kernels/jahyee__nemotron-trace-logprob-audit/nemotron-trace-logprob-audit.py` | `a76157bbe37a26ddb04afa64372eee9ba8de519b5a78e4cdfb696ab36e5110ed` |
| `recipes/kaggle-kernels/jahyee__nemotron-trace-logprob-audit/trace_logprob_audit.py` | `31959f4b61c717e1e198f5e8564d843e6bd726b32a00275a3154947072e9ad3d` |

The occurrences of `KAGGLE_API_TOKEN` in selected recipes are names of an
environment-variable interface only; the review found no token value.

## Checked notebooks

Both notebooks are code-only snapshots: all saved code-cell `outputs` arrays
are empty, and their code-cell source hash is listed below.  The hash is formed
by serializing the code-cell sources in notebook order, separated by a NUL byte.

| Public notebook | Local source SHA-256 | Released source SHA-256 | Code-cell source SHA-256 | Release action |
| --- | --- | --- | --- | --- |
| `notebook-recipes/jahyee__nemotron-kienngx-notebook/nemotron-kienngx-notebook.ipynb` | `4bda2df4a9c098c672916cd5e0aec8f447edc4f540aa30e3ebe5d714654db87a` | unchanged | `ba1c499e11441b484d658e1b64b33d807d141cf62dfdd9b04c0e5762bc129754` | byte-identical source snapshot |
| `notebook-recipes/jahyee__nemotron-oracle-sft-r32/nemotron-oracle-sft-r32.ipynb` | `639dd61111e8d5397afdf093ae3ddec0a0ad964726b244b89a38cbda6b149d6d` | `276b71e9c9be2acf2d358ff1e25619098a3328a73a923dda7d5d2ba8c8aa9292` | `af34edb218c06051229baafb60d074016f8872e189db535a1527859d5ec1ff90` | removed only top-level `papermill` and `widgets` runtime metadata |

The second notebook's remaining metadata describes its kernel and declared
Kaggle input references; it does not package those inputs.  Its code cells and
their empty saved outputs are unchanged by the metadata-only cleanup.

## Explicit exclusions and boundaries

- `nemotron-trace-logprob-audit-bundled.py` is excluded because it embeds a
  generated trace manifest; the checked launcher and reusable audit module are
  the public source artifacts instead.
- Row-level trace JSONL files, their summary JSON, local execution notes, and
  Python bytecode caches are excluded.  They are not needed to understand or
  reuse the source method and may expose derived record-level material.
- Notebooks authored by other accounts are excluded; this repository does not
  mirror third-party notebooks, weights, or cached packages.
- Competition and provider datasets remain external references only.  They are
  not redistributed here, even when a recipe can consume them.
- The independently released first-party model/submission artifact is in the
  [Kaggle artifact dataset](https://www.kaggle.com/datasets/jahyee/openkaggle-nemotron-model-submission-artifacts).
  Its retained primary submission archive is SHA-256
  `cc5dbfca7c03c13175d464716aebdb8798aabd19643b30eff21ef8595de8083e`.

For the operational boundary and additional artifact provenance, see
[`reports/archive-provenance-boundary-2026-09-28.md`](../reports/archive-provenance-boundary-2026-09-28.md).
