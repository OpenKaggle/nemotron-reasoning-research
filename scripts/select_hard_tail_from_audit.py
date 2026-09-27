from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
JOIN_FIELDS = ("id", "source", "method")
DEFAULT_DROP_FLAGS = {"truncated", "prefix_mismatch"}


@dataclass
class AuditRow:
    row: dict[str, object]
    score: float
    rank_score: tuple[float, float]


def stable_score(*parts: object, seed: int) -> int:
    text = "::".join(str(part) for part in (*parts, seed))
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def iter_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def norm_text(value: object) -> str:
    return "" if value is None else str(value)


def row_key(row: dict[str, object]) -> tuple[str, str, str]:
    return tuple(norm_text(row.get(field)) for field in JOIN_FIELDS)


def relaxed_keys(row: dict[str, object]) -> list[tuple[str, str]]:
    return [
        (norm_text(row.get("id")), norm_text(row.get("method"))),
        (norm_text(row.get("id")), norm_text(row.get("task_type"))),
        (norm_text(row.get("id")), ""),
    ]


def split_flags(value: object) -> set[str]:
    return {flag for flag in norm_text(value).split(";") if flag}


def load_manifest(path: Path | None) -> tuple[dict[tuple[str, str, str], dict[str, object]], dict[tuple[str, str], list[dict[str, object]]]]:
    strict: dict[tuple[str, str, str], dict[str, object]] = {}
    relaxed: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    if path is None:
        return strict, relaxed
    for row in iter_jsonl(path):
        strict[row_key(row)] = row
        for key in relaxed_keys(row):
            relaxed[key].append(row)
    return strict, relaxed


def merge_manifest_row(
    audit_row: dict[str, object],
    strict_manifest: dict[tuple[str, str, str], dict[str, object]],
    relaxed_manifest: dict[tuple[str, str], list[dict[str, object]]],
) -> dict[str, object]:
    merged = dict(audit_row)
    manifest_row = strict_manifest.get(row_key(audit_row))
    match_type = "strict" if manifest_row is not None else ""
    if manifest_row is None:
        for key in relaxed_keys(audit_row):
            matches = relaxed_manifest.get(key, [])
            if len(matches) == 1:
                manifest_row = matches[0]
                match_type = f"relaxed:{key[1] or 'id'}"
                break
    if manifest_row:
        for key, value in manifest_row.items():
            merged.setdefault(key, value)
            if key in {"prompt", "completion", "answer", "quality_flags", "source", "task_type", "method"}:
                merged[key] = value
        merged["manifest_match"] = match_type
    elif strict_manifest or relaxed_manifest:
        merged["manifest_match"] = "missing"
    return merged


def scored_audit_rows(
    audit_path: Path,
    *,
    score_field: str,
    drop_flags: set[str],
    require_prompt: bool,
    require_completion: bool,
    strict_manifest: dict[tuple[str, str, str], dict[str, object]],
    relaxed_manifest: dict[tuple[str, str], list[dict[str, object]]],
) -> list[AuditRow]:
    rows: list[AuditRow] = []
    for audit in iter_jsonl(audit_path):
        if not is_number(audit.get(score_field)):
            continue
        merged = merge_manifest_row(audit, strict_manifest, relaxed_manifest)
        flags = split_flags(merged.get("quality_flags"))
        if drop_flags & flags:
            continue
        if any(bool(merged.get(flag)) for flag in drop_flags):
            continue
        if require_prompt and not norm_text(merged.get("prompt")).strip():
            continue
        if require_completion and not norm_text(merged.get("completion")).strip():
            continue
        score = float(merged[score_field])
        mean_lp = float(merged.get("mean_logprob")) if is_number(merged.get("mean_logprob")) else 0.0
        rows.append(AuditRow(row=merged, score=score, rank_score=(score, mean_lp)))
    return rows


def split_holdout(rows: list[AuditRow], *, fraction: float, seed: int) -> tuple[list[AuditRow], list[AuditRow]]:
    if fraction <= 0:
        return list(rows), []
    by_task: dict[str, list[AuditRow]] = defaultdict(list)
    for item in rows:
        by_task[norm_text(item.row.get("task_type")) or "unknown"].append(item)
    train: list[AuditRow] = []
    holdout: list[AuditRow] = []
    for task, task_rows in sorted(by_task.items()):
        ordered = sorted(
            task_rows,
            key=lambda item: stable_score(task, item.row.get("id"), item.row.get("source"), item.row.get("method"), seed=seed),
        )
        n_holdout = max(1, round(len(ordered) * fraction))
        holdout.extend(ordered[:n_holdout])
        train.extend(ordered[n_holdout:])
    return train, holdout


def parse_caps(raw_caps: list[str] | None) -> dict[str, int]:
    caps: dict[str, int] = {}
    for raw in raw_caps or []:
        if "=" not in raw:
            raise ValueError("--task-cap values must look like task_type=COUNT")
        task, value = raw.split("=", 1)
        caps[task.strip()] = int(value)
    return caps


def select_hard_tail(
    rows: list[AuditRow],
    *,
    per_task: int,
    max_rows: int,
    task_caps: dict[str, int],
    seed: int,
) -> list[AuditRow]:
    by_task: dict[str, list[AuditRow]] = defaultdict(list)
    for item in rows:
        by_task[norm_text(item.row.get("task_type")) or "unknown"].append(item)
    for task_rows in by_task.values():
        task_rows.sort(
            key=lambda item: (
                item.rank_score[0],
                item.rank_score[1],
                stable_score(item.row.get("id"), item.row.get("source"), item.row.get("method"), seed=seed),
            )
        )

    selected: list[AuditRow] = []
    seen: set[tuple[str, str, str]] = set()
    per_task_counts: Counter[str] = Counter()

    round_index = 0
    while len(selected) < max_rows:
        changed = False
        for task in sorted(by_task):
            cap = task_caps.get(task, per_task)
            if cap <= 0 or per_task_counts[task] >= cap:
                continue
            task_rows = by_task[task]
            while task_rows:
                candidate = task_rows.pop(0)
                key = row_key(candidate.row)
                if key in seen:
                    continue
                seen.add(key)
                selected.append(candidate)
                per_task_counts[task] += 1
                changed = True
                break
            if len(selected) >= max_rows:
                break
        round_index += 1
        if not changed or round_index > max_rows + len(by_task):
            break
    selected.sort(key=lambda item: (item.rank_score[0], item.rank_score[1]))
    return selected[:max_rows]


def output_row(item: AuditRow) -> dict[str, object]:
    row = dict(item.row)
    row["hard_tail_score"] = item.score
    return row


def score_related_fields(score_field: str) -> tuple[str, str, str]:
    if score_field == "content_min_logprob":
        return "content_mean_logprob", "content_worst_token_text", "content_worst_context"
    return "mean_logprob", "worst_token_text", "worst_context"


def write_jsonl(rows: Iterable[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(rows: list[dict[str, object]], path: Path, *, score_field: str) -> None:
    if not rows:
        return
    mean_field, token_field, context_field = score_related_fields(score_field)
    fields = [
        "id",
        "source",
        "task_type",
        "method",
        "answer",
        "hard_tail_score",
        score_field,
        mean_field,
        token_field,
        context_field,
        "quality_flags",
        "manifest_match",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def summarize(rows: list[AuditRow], selected: list[AuditRow], holdout: list[AuditRow], *, score_field: str) -> dict[str, object]:
    scored_values = [item.score for item in rows]
    selected_rows = [item.row for item in selected]
    mean_field, token_field, context_field = score_related_fields(score_field)
    return {
        "audit_rows": len(rows),
        "selected_rows": len(selected),
        "holdout_rows": len(holdout),
        "score": {
            "min": min(scored_values) if scored_values else None,
            "p01": percentile(scored_values, 0.01),
            "p05": percentile(scored_values, 0.05),
            "p50": percentile(scored_values, 0.50),
            "mean": sum(scored_values) / len(scored_values) if scored_values else None,
        },
        "selected_by_task_type": dict(Counter(norm_text(row.get("task_type")) for row in selected_rows).most_common()),
        "selected_by_source": dict(Counter(norm_text(row.get("source")) for row in selected_rows).most_common()),
        "selected_quality_flags": dict(
            Counter(flag for row in selected_rows for flag in split_flags(row.get("quality_flags"))).most_common(50)
        ),
        "selected_worst_tokens": [
            {
                "worst_token_text": token,
                "count": count,
            }
            for token, count in Counter(norm_text(row.get(token_field)) for row in selected_rows).most_common(30)
        ],
        "worst_selected_rows": [
            {
                "id": row.get("id"),
                "source": row.get("source"),
                "task_type": row.get("task_type"),
                "method": row.get("method"),
                "score": row.get(score_field),
                "mean_logprob": row.get(mean_field),
                "worst_token_text": row.get(token_field),
                "worst_context": row.get(context_field),
                "manifest_match": row.get("manifest_match"),
            }
            for row in selected_rows[:30]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select balanced hard-tail trace rows from a token logprob audit."
    )
    parser.add_argument("--audit", type=Path, required=True, help="Audit JSONL with min_logprob/mean_logprob fields.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional canonical manifest for prompt/completion backfill.")
    parser.add_argument("--output", type=Path, required=True, help="Selected hard-tail JSONL.")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--holdout-output", type=Path, default=None)
    parser.add_argument("--score-field", default="min_logprob")
    parser.add_argument("--max-rows", type=int, default=256)
    parser.add_argument("--per-task", type=int, default=64)
    parser.add_argument("--task-cap", action="append", default=None, help="Override per-task cap, e.g. bit_manipulation=96")
    parser.add_argument("--holdout-fraction", type=float, default=0.08)
    parser.add_argument(
        "--drop-flag",
        action="append",
        default=None,
        help="Drop rows containing this quality flag or truthy audit boolean. Defaults to truncated and prefix_mismatch when omitted.",
    )
    parser.add_argument("--require-prompt", action="store_true")
    parser.add_argument("--require-completion", action="store_true")
    parser.add_argument("--seed", type=int, default=20260606)
    args = parser.parse_args()

    strict_manifest, relaxed_manifest = load_manifest(args.manifest)
    rows = scored_audit_rows(
        args.audit,
        score_field=args.score_field,
        drop_flags=set(args.drop_flag if args.drop_flag is not None else sorted(DEFAULT_DROP_FLAGS)),
        require_prompt=args.require_prompt,
        require_completion=args.require_completion,
        strict_manifest=strict_manifest,
        relaxed_manifest=relaxed_manifest,
    )
    train_rows, holdout = split_holdout(rows, fraction=args.holdout_fraction, seed=args.seed)
    selected = select_hard_tail(
        train_rows,
        per_task=args.per_task,
        max_rows=args.max_rows,
        task_caps=parse_caps(args.task_cap),
        seed=args.seed,
    )
    selected_out = [output_row(item) for item in selected]
    write_jsonl(selected_out, args.output)
    if args.output_csv:
        write_csv(selected_out, args.output_csv, score_field=args.score_field)
    if args.holdout_output:
        write_jsonl((output_row(item) for item in holdout), args.holdout_output)
    summary = summarize(rows, selected, holdout, score_field=args.score_field)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
