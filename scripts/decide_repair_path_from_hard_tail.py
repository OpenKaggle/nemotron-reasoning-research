from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


BIT_RE = re.compile(
    r"\b(bit|binary|8-bit|rot|rotate|shl|shr|shift|xor|and|or|not|gate|mask|truth)\b|[01]{6,}",
    re.IGNORECASE,
)
SYMBOLIC_RE = re.compile(
    r"operator|cryptarithm|equation|mapping|symbol|digit|encode|decode|transform|【|】|->|[A-Z]\s*[+\-*/=]\s*[A-Z]",
    re.IGNORECASE,
)
FORMAT_RE = re.compile(
    r"\\boxed|boxed|final answer|brace|answer marker|</think>|<think>|\{|\}|newline",
    re.IGNORECASE,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_counter(raw: object) -> Counter[str]:
    counter: Counter[str] = Counter()
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                counter[str(key)] += int(value)
            except (TypeError, ValueError):
                continue
    return counter


def iter_worst_rows(summary: dict) -> Iterable[dict]:
    rows = summary.get("worst_selected_rows")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                yield row


def text_hits(pattern: re.Pattern[str], values: Iterable[object]) -> int:
    hits = 0
    for value in values:
        if pattern.search("" if value is None else str(value)):
            hits += 1
    return hits


def score_paths(summary: dict) -> dict[str, dict[str, object]]:
    selected_rows = int(summary.get("selected_rows") or 0)
    task_counts = normalize_counter(summary.get("selected_by_task_type"))
    source_counts = normalize_counter(summary.get("selected_by_source"))
    flag_counts = normalize_counter(summary.get("selected_quality_flags"))
    token_counts = Counter(
        str(item.get("worst_token_text") or "")
        for item in summary.get("selected_worst_tokens", [])
        if isinstance(item, dict)
        for _ in range(int(item.get("count") or 0))
    )
    worst_rows = list(iter_worst_rows(summary))
    contexts = [row.get("worst_context") for row in worst_rows]
    tokens = list(token_counts.elements()) + [row.get("worst_token_text") for row in worst_rows]

    bit_task = sum(count for task, count in task_counts.items() if "bit" in task.lower())
    bit_source = sum(
        count
        for source, count in source_counts.items()
        if any(name in source.lower() for name in ("donald", "bankoglu", "tong", "llkh0a", "leevvin"))
    )
    symbolic_task = sum(
        count
        for task, count in task_counts.items()
        if any(name in task.lower() for name in ("symbolic", "equation", "huikang_14class"))
    )
    formatting_flags = sum(
        count
        for flag, count in flag_counts.items()
        if flag in {"multiple_boxed", "boxed_appended", "final_boxed_mismatch", "answer_missing"}
    )

    bit_context = text_hits(BIT_RE, [*contexts, *tokens])
    symbolic_context = text_hits(SYMBOLIC_RE, [*contexts, *tokens])
    formatting_context = text_hits(FORMAT_RE, [*contexts, *tokens])

    denom = max(selected_rows, 1)
    scores: dict[str, dict[str, object]] = {
        "symbolic_operator": {
            "score": symbolic_task / denom + 0.35 * symbolic_context / max(len(worst_rows), 1),
            "evidence": {
                "task_rows": symbolic_task,
                "context_hits": symbolic_context,
                "top_tasks": task_counts.most_common(5),
            },
        },
        "bit_repair": {
            "score": bit_task / denom + 0.25 * bit_source / denom + 0.35 * bit_context / max(len(worst_rows), 1),
            "evidence": {
                "task_rows": bit_task,
                "source_rows": bit_source,
                "context_hits": bit_context,
                "top_sources": source_counts.most_common(5),
            },
        },
        "formatting_repair": {
            "score": 0.5 * formatting_flags / denom + 0.6 * formatting_context / max(len(worst_rows), 1),
            "evidence": {
                "quality_flag_rows": formatting_flags,
                "context_hits": formatting_context,
                "top_flags": flag_counts.most_common(8),
                "top_tokens": token_counts.most_common(8),
            },
        },
        "adapter_arithmetic": {
            "score": 0.05,
            "evidence": {
                "reason": "Fallback only when no task-family or formatting tail dominates.",
                "selected_rows": selected_rows,
            },
        },
    }
    return scores


def choose_path(scores: dict[str, dict[str, object]]) -> str:
    ordered = sorted(scores.items(), key=lambda item: float(item[1]["score"]), reverse=True)
    best_name, best = ordered[0]
    second = float(ordered[1][1]["score"]) if len(ordered) > 1 else 0.0
    best_score = float(best["score"])
    if best_score < 0.25 or best_score - second < 0.08:
        return "inspect_manually"
    return best_name


def recommendation(path: str) -> str:
    if path == "symbolic_operator":
        return "Build a symbolic/operator verifier DSL or generate-verify-distill set before training; avoid blind SFT on gold-conditioned traces."
    if path == "bit_repair":
        return "Build bit-only DPO/IPO or tiny micro-SFT from Donald/Bankoglu/Tong positives against Kishan failures, with Kienngx anchor holdout."
    if path == "formatting_repair":
        return "Use shorter traces or boxed-region/final-answer loss; do not add new reasoning data until formatting tail is reduced."
    if path == "adapter_arithmetic":
        return "Run Kienngx-anchored, shape-gated delta-space SVD soup as a secondary probe; keep MoE experts and incompatible keys anchor-copied."
    return "Do not train yet; inspect worst contexts and expand the audit slice until one repair path dominates."


def write_markdown(result: dict, path: Path) -> None:
    chosen = result["chosen_path"]
    lines = [
        "# Hard-Tail Repair Decision",
        "",
        f"Chosen path: `{chosen}`",
        "",
        recommendation(chosen),
        "",
        "## Scores",
        "",
    ]
    for name, payload in sorted(result["scores"].items(), key=lambda item: float(item[1]["score"]), reverse=True):
        lines.append(f"- `{name}`: {float(payload['score']):.3f}")
    lines.extend(["", "## Next Command", ""])
    lines.append(result["next_step"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Choose one repair path from a hard-tail audit summary.")
    parser.add_argument("--summary", type=Path, required=True, help="Summary JSON from select_hard_tail_from_audit.py.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    summary = read_json(args.summary)
    scores = score_paths(summary)
    chosen = choose_path(scores)
    result = {
        "summary": str(args.summary),
        "chosen_path": chosen,
        "recommendation": recommendation(chosen),
        "next_step": recommendation(chosen),
        "scores": scores,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(result, args.output_md)


if __name__ == "__main__":
    main()
