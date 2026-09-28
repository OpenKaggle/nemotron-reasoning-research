# Remaining Nemotron archive: public publication boundary

This release records the four local cache families that were not part of the
main source repository: `reports/`, `public_radar/`, `external/`, and
`tmp_kaggle_recon/`.

The complete local inventory is preserved in
[`remaining-archive-manifest-2026-09-28.csv`](remaining-archive-manifest-2026-09-28.csv).
Every row has the local relative path, byte size, SHA-256, source family, and
release disposition. The manifest is an index, not a copy of the local cache.

## What is published here

`sanitized-evidence-2026-09-28/` contains compact derivatives of selected
first-party reports and experiment records. It keeps schema, row counts,
numeric metrics, string lengths, aggregate fields, and content hashes where
useful. It removes prompts, answers, completions, official IDs, token arrays,
trace text, and absolute local paths. The files are intended to support review
of experiment shape and conclusions without reconstructing competition rows.

## What remains link-only

- `public_radar/` is a cache of other authors' Kaggle notebooks and datasets;
  this release keeps the source index and hashes, not vendor copies.
- `external/` contains third-party code/corpora; reuse requires the upstream
  license and attribution, so it is indexed rather than relicensed as
  OpenKaggle work.
- `tmp_kaggle_recon/` contains reconstructed or downloaded competition-adjacent
  corpora, including prompt/answer/trace-shaped records. The local competition
  rules require measures preventing transmission, duplication, publication, or
  redistribution to people who have not agreed to the rules. Those bytes are
  therefore not mirrored here.
- Copied public-audit notebooks, ZIPs, model payloads, and row-level CSV/JSONL
  files remain source-index-only even when their upstream page is public.

The public route for a withheld file is its original Kaggle/GitHub URL, source
slug/version, retrieval date, local SHA-256, and the sanitized experiment
derivative when one exists. This preserves provenance and research value
without treating a local download or a locally generated wrapper as a new
redistribution license.
