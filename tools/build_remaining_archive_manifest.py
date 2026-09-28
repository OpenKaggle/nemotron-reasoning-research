#!/usr/bin/env python3
"""Build a hash-bound, content-free inventory of the remaining local cache."""
from __future__ import annotations

import csv
import hashlib
import os
import argparse
from pathlib import Path

OUT = Path("reports/remaining-archive-manifest-2026-09-28.csv")
GROUPS = ("reports", "public_radar", "external", "tmp_kaggle_recon")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def disposition(group: str, rel: str) -> tuple[str, str]:
    lower = rel.lower()
    if group == "reports":
        if any(x in lower for x in ("public_audit", "leaderboard", ".zip", ".ipynb")):
            return "source-index-only", "copied public/competition or third-party artifact"
        if lower.endswith((".jsonl", ".csv")) or any(x in lower for x in ("prompt", "trace", "holdout")):
            return "sanitized-derived-only", "row-level prompt/answer/completion or trace fields"
        return "reviewed-derived-candidate", "first-party report candidate; scrub local paths before release"
    if group == "public_radar":
        return "source-index-only", "downloaded third-party Kaggle notebook/dataset snapshot"
    if group == "external":
        return "license-review-only", "third-party source/corpus; retain attribution and license evidence"
    return "source-index-only", "reconstructed or downloaded competition-adjacent corpus"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="local cache root; never committed")
    args = parser.parse_args()
    root = args.root
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for group in GROUPS:
        base = root / group
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            rel = path.relative_to(root).as_posix()
            release_class, reason = disposition(group, rel)
            rows.append({
                "path": rel,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "source_group": group,
                "release_class": release_class,
                "reason": reason,
            })
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("path", "bytes", "sha256", "source_group", "release_class", "reason"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} entries to {OUT}")


if __name__ == "__main__":
    main()
