from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from verifier_rewards import normalize_task_type, reward_completion


def iter_positive(path: Path) -> Iterable[dict[str, object]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                yield {
                    "id": str(row.get("id", "")),
                    "task_type": normalize_task_type(row.get("task_type")),
                    "source": f"positive:{row.get('source', 'manifest')}",
                    "prompt": row.get("prompt", ""),
                    "answer": row.get("answer", ""),
                    "completion": row.get("completion", ""),
                    "method": row.get("method", ""),
                    "original_quality_flags": row.get("quality_flags", ""),
                }


def iter_kishan(path: Path) -> Iterable[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yield {
                "id": str(row.get("id", "")),
                "task_type": normalize_task_type(row.get("problem type")),
                "source": "kishanvavdara",
                "prompt": row.get("prompt", ""),
                "answer": row.get("correct answer", ""),
                "completion": row.get("generated", ""),
                "generated_answer": row.get("generated answer", ""),
                "method": "baseline_generation",
                "provided_correctness": row.get("correctness", ""),
            }


def write_rows(rows: Iterable[dict[str, object]], output: Path, limit: int | None) -> Counter[str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    written = 0
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            reward = reward_completion(
                row.get("prompt", ""),
                row.get("completion", ""),
                row.get("answer", ""),
                row.get("task_type", ""),
                row.get("generated_answer", ""),
            )
            out = {
                "id": row.get("id", ""),
                "task_type": row.get("task_type", ""),
                "source": row.get("source", ""),
                "prompt": row.get("prompt", ""),
                "answer": row.get("answer", ""),
                "completion": row.get("completion", ""),
                "extracted_answer": reward.extracted_answer,
                "reward": reward.reward,
                "terms": reward.terms,
                "flags": reward.flags,
                "method": row.get("method", ""),
            }
            for key in ("provided_correctness", "original_quality_flags"):
                if key in row:
                    out[key] = row[key]
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            counts[f"task:{out['task_type']}"] += 1
            counts[f"source:{out['source']}"] += 1
            if "answer_exact" in reward.flags:
                counts["answer_exact"] += 1
            written += 1
            if limit is not None and written >= limit:
                break
    counts["written"] = written
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build dense verifier reward rows from positive traces and Kishan trajectories."
    )
    parser.add_argument("--positive", type=Path, required=True)
    parser.add_argument("--kishan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-type", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--positives-only", action="store_true")
    parser.add_argument("--kishan-only", action="store_true")
    args = parser.parse_args()

    wanted = normalize_task_type(args.task_type) if args.task_type else ""

    def selected() -> Iterable[dict[str, object]]:
        if not args.kishan_only:
            for row in iter_positive(args.positive):
                if not wanted or row["task_type"] == wanted:
                    yield row
        if not args.positives_only:
            for row in iter_kishan(args.kishan):
                if not wanted or row["task_type"] == wanted:
                    yield row

    counts = write_rows(selected(), args.output, args.limit or None)
    print(json.dumps(counts, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
