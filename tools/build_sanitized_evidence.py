#!/usr/bin/env python3
"""Make compact, content-free evidence derivatives from selected local files."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import argparse
from pathlib import Path
from typing import Any

OUT = Path("reports/sanitized-evidence-2026-09-28")
BAD = re.compile(r"(prompt|completion|answer|question|response|trace|token|label|target|problem|input|output|id|submission|path)", re.I)
PATHISH = re.compile(r"/(?:Users|private|tmp|Volumes)/[^\s\"']+")


def scalar(v: Any) -> bool:
    return v is None or isinstance(v, (bool, int, float))


def clean_value(key: str, value: Any) -> Any:
    if BAD.search(key):
        if isinstance(value, (list, dict, str)):
            return None
    if scalar(value):
        return value
    if isinstance(value, str):
        return {"length": len(value)}
    if isinstance(value, list):
        return {"count": len(value)}
    if isinstance(value, dict):
        return {"keys": sorted(str(k) for k in value)[:32]}
    return None


def clean_json(value: Any, key: str = "") -> Any:
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            cv = clean_json(v, str(k))
            if cv is not None:
                out[str(k)] = cv
        return out
    if isinstance(value, list):
        if BAD.search(key):
            return {"count": len(value)}
        return [clean_json(v, key) for v in value[:128]]
    if isinstance(value, str):
        if BAD.search(key) or "context" in key.lower() or len(value) > 120:
            return {"length": len(value), "sha256": hashlib.sha256(value.encode()).hexdigest()}
        return PATHISH.sub("<redacted-path>", value)
    return value


def record_digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sanitize_jsonl(src: Path, dst: Path) -> None:
    rows = 0
    with src.open(errors="replace") as inp, dst.open("w") as out:
        for line in inp:
            if not line.strip():
                continue
            rows += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                clean = {"row_index": rows}
                for k, v in obj.items():
                    cv = clean_value(str(k), v)
                    if cv is not None and not BAD.search(str(k)):
                        clean[str(k)] = cv
                out.write(json.dumps(clean, sort_keys=True) + "\n")
            else:
                out.write(json.dumps({"row_index": rows, "record_type": type(obj).__name__}) + "\n")


def sanitize_csv(src: Path, dst: Path) -> None:
    with src.open(errors="replace", newline="") as inp, dst.open("w", newline="") as out:
        reader = csv.DictReader(inp)
        keep = [k for k in (reader.fieldnames or []) if not BAD.search(k)]
        writer = csv.DictWriter(out, fieldnames=["row_index", *keep])
        writer.writeheader()
        for i, row in enumerate(reader, 1):
            clean = {"row_index": i}
            for k in keep:
                v = row.get(k, "")
                if v and not re.fullmatch(r"-?\d+(?:\.\d+)?", v):
                    clean[k] = f"<text:{len(v)}>"
                else:
                    clean[k] = v
            writer.writerow(clean)


def sanitize_text(src: Path, dst: Path) -> None:
    text = src.read_text(errors="replace")
    text = PATHISH.sub("<redacted-path>", text)
    text = re.sub(r"(?i)(prompt|completion|answer|question|response|trace)\s*[:=].*", r"\1: <redacted-content>", text)
    dst.write_text(text)


def copy_selected(root: Path) -> None:
    selected: list[Path] = []
    reports = root / "reports"
    selected.extend(p for p in reports.glob("*") if p.is_file() and p.suffix.lower() in {".json", ".jsonl", ".csv", ".md", ".txt", ".log"})
    selected.extend(p for p in (root / "external").glob("*") if p.is_file() and p.suffix.lower() in {".jsonl", ".py"})
    # Direct dataset-level CSV/JSONL outputs only; skip nested corpus and trace trees.
    for d in (root / "tmp_kaggle_recon" / "datasets").glob("*"):
        selected.extend(p for p in d.glob("*") if p.is_file() and p.suffix.lower() in {".csv", ".jsonl"})
    for src in sorted(selected):
        rel = src.relative_to(root)
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix.lower() == ".jsonl":
            sanitize_jsonl(src, dst)
        elif src.suffix.lower() == ".csv":
            sanitize_csv(src, dst)
        elif src.suffix.lower() == ".json":
            try:
                obj = json.loads(src.read_text(errors="replace"))
                dst.write_text(json.dumps(clean_json(obj), ensure_ascii=False, indent=2) + "\n")
            except Exception:
                sanitize_text(src, dst)
        elif src.suffix.lower() in {".md", ".txt", ".log"}:
            sanitize_text(src, dst)
        else:
            dst.write_text("source code retained by reference only; see manifest\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="local cache root; never committed")
    copy_selected(parser.parse_args().root)
