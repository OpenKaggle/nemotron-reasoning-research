from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from verifier_rewards import normalize_answer, normalize_task_type, reward_completion  # noqa: E402


@dataclass
class ScoredCompletion:
    id: str
    task_type: str
    source: str
    scores: dict[str, float]
    flags: list[str]


def clean(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def first_present(row: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return ""


def char_ngrams(text: str, n: int = 3) -> Counter[str]:
    value = f"  {normalize_answer(text).lower()}  "
    if len(value) <= n:
        return Counter([value]) if value.strip() else Counter()
    return Counter(value[idx : idx + n] for idx in range(len(value) - n + 1))


def cosine_similarity(left: str, right: str) -> float:
    left_counts = char_ngrams(left)
    right_counts = char_ngrams(right)
    if not left_counts and not right_counts:
        return 1.0
    if not left_counts or not right_counts:
        return 0.0
    dot = sum(count * right_counts.get(key, 0) for key, count in left_counts.items())
    left_norm = math.sqrt(sum(count * count for count in left_counts.values()))
    right_norm = math.sqrt(sum(count * count for count in right_counts.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def sequence_similarity(left: str, right: str) -> float:
    left_norm = normalize_answer(left).lower()
    right_norm = normalize_answer(right).lower()
    if not left_norm and not right_norm:
        return 1.0
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_completion(
    row: dict[str, Any],
    *,
    completion_key: str = "completion",
    source_fallback: str = "",
    task_type_override: str = "",
) -> ScoredCompletion:
    task_type = normalize_task_type(task_type_override or first_present(row, ("task_type", "problem type", "task")))
    prompt = first_present(row, ("prompt", "question", "input"))
    answer = first_present(row, ("answer", "correct answer", "target", "label"))
    completion = first_present(row, (completion_key, "completion", "generated", "response", "text"))
    generated_answer = first_present(row, ("generated_answer", "generated answer", "extracted_answer"))
    verifier = reward_completion(prompt, completion, answer, task_type, generated_answer)

    exact = verifier.terms.get("answer_exact", 0.0)
    partial = verifier.terms.get("answer_partial", 0.0)
    seq = sequence_similarity(answer, verifier.extracted_answer)
    cos = cosine_similarity(answer, verifier.extracted_answer)
    format_valid = verifier.terms.get("format_valid", 0.0)
    parseable = verifier.terms.get("trace_parseable", 0.0)
    boxed = verifier.terms.get("boxed_valid", 0.0)

    sequence_correctness = clamp01(0.62 * exact + 0.20 * partial + 0.10 * seq + 0.08 * cos)
    banwait_composite_simple = clamp01(
        0.44 * exact
        + 0.22 * partial
        + 0.14 * format_valid
        + 0.10 * parseable
        + 0.06 * boxed
        + 0.04 * cos
    )

    return ScoredCompletion(
        id=clean(first_present(row, ("id", "problem_id", "uid"))),
        task_type=task_type,
        source=clean(first_present(row, ("source", f"{completion_key}_source"))) or source_fallback,
        scores={
            "verifier": round(verifier.reward, 6),
            "sequence_correctness": round(sequence_correctness, 6),
            "banwait_composite_simple": round(banwait_composite_simple, 6),
        },
        flags=verifier.flags,
    )


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
            if isinstance(row, dict):
                yield row


def iter_csv(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def filtered(rows: Iterable[dict[str, Any]], wanted_task: str) -> Iterable[dict[str, Any]]:
    for row in rows:
        task_type = normalize_task_type(first_present(row, ("task_type", "problem type", "task")))
        if wanted_task and task_type != wanted_task:
            continue
        yield row


def load_positive_by_id(path: Path, wanted_task: str, limit: int | None) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in filtered(iter_jsonl(path), wanted_task):
        row_id = clean(first_present(row, ("id", "problem_id", "uid")))
        if not row_id or row_id in by_id:
            continue
        by_id[row_id] = row
        if limit is not None and len(by_id) >= limit:
            break
    return by_id


def build_dpo_pairs(path: Path, wanted_task: str, limit: int | None) -> list[tuple[ScoredCompletion, ScoredCompletion]]:
    pairs: list[tuple[ScoredCompletion, ScoredCompletion]] = []
    for row in filtered(iter_jsonl(path), wanted_task):
        chosen = score_completion(row, completion_key="chosen", source_fallback=clean(row.get("chosen_source", "chosen")))
        rejected = score_completion(row, completion_key="rejected", source_fallback=clean(row.get("rejected_source", "rejected")))
        pairs.append((chosen, rejected))
        if limit is not None and len(pairs) >= limit:
            break
    return pairs


def build_positive_kishan_pairs(
    positive_path: Path,
    kishan_path: Path,
    wanted_task: str,
    limit: int | None,
) -> list[tuple[ScoredCompletion, ScoredCompletion]]:
    positives = load_positive_by_id(positive_path, wanted_task, None)
    pairs: list[tuple[ScoredCompletion, ScoredCompletion]] = []
    for row in filtered(iter_csv(kishan_path), wanted_task):
        row_id = clean(first_present(row, ("id", "problem_id", "uid")))
        positive = positives.get(row_id)
        if not positive:
            continue
        chosen = score_completion(positive, source_fallback=clean(positive.get("source", "positive")))
        rejected = score_completion(row, completion_key="generated", source_fallback="kishanvavdara")
        pairs.append((chosen, rejected))
        if limit is not None and len(pairs) >= limit:
            break
    return pairs


def summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None}
    return {
        "count": len(values),
        "mean": round(mean(values), 6),
        "median": round(median(values), 6),
    }


def summarize_group(rows: list[ScoredCompletion]) -> dict[str, Any]:
    by_reward: dict[str, list[float]] = defaultdict(list)
    flags: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    exact_counts: Counter[str] = Counter()
    for row in rows:
        task_counts[row.task_type or "unknown"] += 1
        source_counts[row.source or "unknown"] += 1
        flags.update(row.flags)
        for name, score in row.scores.items():
            by_reward[name].append(score)
        if "answer_exact" in row.flags:
            exact_counts["exact"] += 1

    count = len(rows)
    return {
        "count": count,
        "rewards": {name: summarize_values(values) for name, values in sorted(by_reward.items())},
        "exact_rate": round(exact_counts["exact"] / count, 6) if count else None,
        "flags_top_counts": dict(flags.most_common(12)),
        "task_counts": dict(sorted(task_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
    }


def summarize_pairs(pairs: list[tuple[ScoredCompletion, ScoredCompletion]]) -> dict[str, Any]:
    chosen_rows = [chosen for chosen, _ in pairs]
    rejected_rows = [rejected for _, rejected in pairs]
    rewards = sorted(chosen_rows[0].scores) if chosen_rows else ["banwait_composite_simple", "sequence_correctness", "verifier"]
    margins = {
        name: [round(chosen.scores[name] - rejected.scores[name], 6) for chosen, rejected in pairs]
        for name in rewards
    }
    wins = {
        name: round(sum(1 for value in values if value > 0.0) / len(values), 6) if values else None
        for name, values in margins.items()
    }

    by_task: dict[str, dict[str, Any]] = {}
    by_source: dict[str, dict[str, Any]] = {}
    for key_name, key_func, target in (
        ("task_type", lambda pair: pair[0].task_type or pair[1].task_type or "unknown", by_task),
        ("source_pair", lambda pair: f"{pair[0].source or 'chosen'}>{pair[1].source or 'rejected'}", by_source),
    ):
        buckets: dict[str, list[tuple[ScoredCompletion, ScoredCompletion]]] = defaultdict(list)
        for pair in pairs:
            buckets[key_func(pair)].append(pair)
        for key, bucket in sorted(buckets.items()):
            bucket_margins = {
                name: [round(chosen.scores[name] - rejected.scores[name], 6) for chosen, rejected in bucket]
                for name in rewards
            }
            target[key] = {
                "count": len(bucket),
                "chosen_minus_rejected": {
                    name: summarize_values(values) for name, values in bucket_margins.items()
                },
            }
        _ = key_name

    return {
        "pair_count": len(pairs),
        "chosen": summarize_group(chosen_rows),
        "rejected": summarize_group(rejected_rows),
        "chosen_minus_rejected": {name: summarize_values(values) for name, values in margins.items()},
        "chosen_gt_rejected_rate": wins,
        "by_task": by_task,
        "by_source": by_source,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare verifier reward against lightweight calibration rewards on preference-style data."
    )
    parser.add_argument("--positive", type=Path, help="Positive/completion JSONL.")
    parser.add_argument("--kishan", type=Path, help="Kishan CSV with generated completions.")
    parser.add_argument("--dpo-pairs", type=Path, help="Chosen/rejected JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON report.")
    parser.add_argument("--task-type", default="", help="Optional task filter.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum pairs to score; 0 means all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wanted_task = normalize_task_type(args.task_type) if args.task_type else ""
    limit = args.limit or None
    if args.dpo_pairs:
        mode = "dpo_pairs"
        pairs = build_dpo_pairs(args.dpo_pairs, wanted_task, limit)
    elif args.positive and args.kishan:
        mode = "positive_vs_kishan"
        pairs = build_positive_kishan_pairs(args.positive, args.kishan, wanted_task, limit)
    else:
        raise SystemExit("Provide either --dpo-pairs or both --positive and --kishan.")

    report = {
        "mode": mode,
        "inputs": {
            "positive": str(args.positive) if args.positive else None,
            "kishan": str(args.kishan) if args.kishan else None,
            "dpo_pairs": str(args.dpo_pairs) if args.dpo_pairs else None,
            "task_type": wanted_task or None,
            "limit": limit,
        },
        "reward_names": ["verifier", "sequence_correctness", "banwait_composite_simple"],
        "summary": summarize_pairs(pairs),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "mode": mode, "pairs": len(pairs)}, sort_keys=True))


if __name__ == "__main__":
    main()
