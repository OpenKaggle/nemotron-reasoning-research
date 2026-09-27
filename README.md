# NVIDIA Nemotron Competition Research

This is the source-only public record of research around the
[NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge).
It retains reviewable scripts, dependency information, and compact
model-structure audits. It does not redistribute competition records or carry
large model artifacts in Git history.

## What is here

- first-party preparation, evaluation, and artifact-inspection scripts;
- dependency information for reconstructing the software environment;
- scalar model and adapter audit outputs with no competition rows, prompts,
  answers, predictions, or trace text;
- an aggregate-only historical evaluation summary.
- nine first-party Kaggle recipe scripts and two zero-output notebook sources,
  released with dependency metadata under
  [`recipes/`](recipes/README.md).

## What is deliberately absent

- organizer-provided competition data, prompt text, IDs, labels, answers, and
  predictions;
- record-level evaluator exports, token/context traces, and replay payloads;
- copied notebooks and upstream model files; and
- credentials, local paths, caches, and virtual environments.

Reviewed first-party model/submission outputs are published separately from
this source tree in the [OpenKaggle Nemotron Model and Submission
Artifacts](https://www.kaggle.com/datasets/jahyee/openkaggle-nemotron-model-submission-artifacts)
dataset. Its manifest and SHA-256 file identify the bytes; it does not make
this repository a mirror of organizer data or copied upstream weights.

The original local workspace remains a mixed-provenance working archive. It is
not mirrored here, and this repository is not a claim that every local file is
eligible for redistribution.

## Reproduce responsibly

Read [DATA_SOURCES.md](DATA_SOURCES.md) before running a script. Obtain
competition inputs only from the official source after accepting the applicable
rules. [ARTIFACTS.md](ARTIFACTS.md) distinguishes reconstructible sources from
separately released, reviewed artifacts. The public release gate
and its history-cleanup receipt are in [RELEASE_MANIFEST.md](RELEASE_MANIFEST.md).
The recipe-specific boundary and a source-to-Kaggle provenance map are in
[`recipes/README.md`](recipes/README.md).

## Licensing and evidence

No license is granted for NVIDIA, Kaggle, organizer data, third-party models,
or third-party adapters. The linked artifact dataset states the licence for
each original OpenKaggle-authored output and preserves upstream boundaries.
Individual retained source files may carry their own notices. Aggregate numbers
are historical research receipts, not a claim of a current leaderboard result.
