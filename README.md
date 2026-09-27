# NVIDIA Nemotron Competition Research

This is the source-only public record of research around the
[NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge).
It retains reviewable scripts, dependency information, and compact
model-structure audits. It does not redistribute competition records or model
artifacts.

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
- checkpoints, adapters, submission archives, copied notebooks, and upstream
  model files;
- credentials, local paths, caches, and virtual environments.

The original local workspace remains a mixed-provenance working archive. It is
not mirrored here, and this repository is not a claim that every local file is
eligible for redistribution.

## Reproduce responsibly

Read [DATA_SOURCES.md](DATA_SOURCES.md) before running a script. Obtain
competition inputs only from the official source after accepting the applicable
rules. [ARTIFACTS.md](ARTIFACTS.md) distinguishes reconstructible sources from
artifacts that need a separately authorized archive. The public release gate
and its history-cleanup receipt are in [RELEASE_MANIFEST.md](RELEASE_MANIFEST.md).
The recipe-specific boundary and a source-to-Kaggle provenance map are in
[`recipes/README.md`](recipes/README.md).

## Licensing and evidence

No license is granted for NVIDIA, Kaggle, organizer data, third-party models,
adapters, notebooks, or submission packages. Individual retained source files
may carry their own notices. Aggregate numbers are historical research receipts,
not a claim of a current leaderboard result or an invitation to reconstruct the
restricted evaluation set.
