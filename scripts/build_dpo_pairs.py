from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from verifier_rewards import normalize_task_type, reward_completion


SOURCE_PRIORITY = {
    "donald": 0,
    "bankoglu": 1,
    "tong": 2,
    "oracle": 3,
    "sybyrr_beta": 4,
    "llkh0a": 5,
    "srivath": 6,
}


def load_positive_index(path: Path, wanted_task: str) -> dict[str, dict[str, object]]:
    candidates: dict[str, list[dict[str, object]]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            task = normalize_task_type(row.get("task_type"))
            if wanted_task and task != wanted_task:
                continue
            reward = reward_completion(
                row.get("prompt", ""),
                row.get("completion", ""),
                row.get("answer", ""),
                task,
            )
            if "answer_exact" not in reward.flags:
                continue
            source = str(row.get("source", ""))
            candidates[str(row.get("id", ""))].append(
                {
                    "id": str(row.get("id", "")),
                    "task_type": task,
                    "source": source,
                    "prompt": row.get("prompt", ""),
                    "answer": row.get("answer", ""),
                    "completion": row.get("completion", ""),
                    "reward": reward.reward,
                    "flags": reward.flags,
                }
            )

    selected: dict[str, dict[str, object]] = {}
    for problem_id, rows in candidates.items():
        rows.sort(
            key=lambda row: (
                SOURCE_PRIORITY.get(str(row["source"]), 99),
                len(str(row["completion"])),
            )
        )
        selected[problem_id] = rows[0]
    return selected


def iter_kishan(path: Path, wanted_task: str) -> Iterable[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            task = normalize_task_type(row.get("problem type"))
            if wanted_task and task != wanted_task:
                continue
            yield {
                "id": str(row.get("id", "")),
                "task_type": task,
                "prompt": row.get("prompt", ""),
                "answer": row.get("correct answer", ""),
                "completion": row.get("generated", ""),
                "generated_answer": row.get("generated answer", ""),
                "provided_correctness": row.get("correctness", ""),
            }


def build_pairs(
    positive_index: dict[str, dict[str, object]],
    kishan_rows: Iterable[dict[str, object]],
    min_margin: float,
    limit: int | None,
) -> tuple[list[dict[str, object]], Counter[str]]:
    pairs: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    for rejected in kishan_rows:
        problem_id = str(rejected["id"])
        chosen = positive_index.get(problem_id)
        if not chosen:
            counts["no_positive_overlap"] += 1
            continue
        rejected_reward = reward_completion(
            rejected.get("prompt", ""),
            rejected.get("completion", ""),
            rejected.get("answer", ""),
            rejected.get("task_type", ""),
            rejected.get("generated_answer", ""),
        )
        if "answer_exact" in rejected_reward.flags:
            counts["rejected_exact_skip"] += 1
            continue
        margin = float(chosen["reward"]) - rejected_reward.reward
        if margin < min_margin:
            counts["low_margin_skip"] += 1
            continue
        pair = {
            "id": problem_id,
            "task_type": chosen["task_type"],
            "prompt": chosen["prompt"] or rejected["prompt"],
            "answer": chosen["answer"],
            "chosen": chosen["completion"],
            "rejected": rejected["completion"],
            "chosen_source": chosen["source"],
            "rejected_source": "kishanvavdara",
            "chosen_reward": chosen["reward"],
            "rejected_reward": rejected_reward.reward,
            "margin": round(margin, 6),
            "chosen_flags": chosen["flags"],
            "rejected_flags": rejected_reward.flags,
            "provided_correctness": rejected.get("provided_correctness", ""),
        }
        pairs.append(pair)
        counts[f"task:{pair['task_type']}"] += 1
        counts[f"chosen_source:{pair['chosen_source']}"] += 1
        if limit is not None and len(pairs) >= limit:
            break
    counts["pairs"] = len(pairs)
    return pairs, counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build DPO/IPO chosen-rejected pairs from verified positives and Kishan failures."
    )
    parser.add_argument("--positive", type=Path, required=True)
    parser.add_argument("--kishan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-margin", type=float, default=0.2)
    args = parser.parse_args()

    wanted = normalize_task_type(args.task_type) if args.task_type else ""
    positive_index = load_positive_index(args.positive, wanted)
    pairs, counts = build_pairs(
        positive_index,
        iter_kishan(args.kishan, wanted),
        args.min_margin,
        args.limit or None,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    counts["positive_overlap_ids"] = len(positive_index)
    print(json.dumps(counts, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
