from __future__ import annotations

import itertools
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from analyze_data import classify_prompt
from reasoning_assets import (
    EQUATION_CANDIDATE_PRIORITY,
    numeric_operands,
    numeric_op_value,
    reverse_numeric_text,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GENERATED_DIR = DATA_DIR / "generated"
REPORT_DIR = ROOT / "reports"

# Per-(mode, op, fmt) candidate-set cap. Backtracking is already early-stopped on
# duplicate solutions, but pathological inputs can blow up the per-item candidate
# list. 5000 = a factor-of-2 above the theoretical max (P(10,4)=5040) for safety.
ITEM_CANDIDATE_CAP = 5000

CRYPT_MODES = ("BA_DC", "AB_CD", "AB_DC", "BA_CD", "CD_AB", "DC_BA")
CRYPT_FORMATS = (
    "raw",
    "rev",
    "zpad2",
    "zpad2_rev",
    "zpad3",
    "zpad3_rev",
    "zpad4",
    "zpad4_rev",
)
CRYPT_OPS = tuple(dict.fromkeys(op_name for _, op_name in EQUATION_CANDIDATE_PRIORITY))


def encoded_equations(prompt: str) -> tuple[list[tuple[str, str]], str | None]:
    examples = [
        (lhs.strip(), rhs.strip())
        for lhs, rhs in re.findall(r"^(.+?) = (.+?)$", prompt, flags=re.M)
        if len(lhs.strip()) == 5
    ]
    target_match = re.search(r"Now, determine the result for: (.+)$", prompt, flags=re.M)
    if not target_match:
        return examples, None
    target = target_match.group(1).strip()
    return examples, target if len(target) == 5 else None


def format_numeric(value: int | str | None) -> dict[str, str]:
    if value is None:
        return {}
    base = str(value)
    outputs = {"raw": base, "rev": reverse_numeric_text(base)}
    if isinstance(value, int):
        sign = "-" if value < 0 else ""
        digits = str(abs(value))
        for width in (2, 3, 4):
            if len(digits) < width:
                padded = sign + digits.zfill(width)
                outputs[f"zpad{width}"] = padded
                outputs[f"zpad{width}_rev"] = reverse_numeric_text(padded)
    return outputs


def merge_maps(
    mapping: dict[str, str],
    inverse: dict[str, str],
    candidate: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]] | None:
    next_mapping = dict(mapping)
    next_inverse = dict(inverse)
    for symbol, digit in candidate.items():
        if symbol in next_mapping and next_mapping[symbol] != digit:
            return None
        if digit in next_inverse and next_inverse[digit] != symbol:
            return None
        next_mapping[symbol] = digit
        next_inverse[digit] = symbol
    return next_mapping, next_inverse


def item_candidate_maps(lhs: str, rhs: str, mode: str, op_name: str, fmt_name: str) -> list[dict[str, str]] | None:
    """Enumerate mappings consistent with one (lhs, rhs) item under a hypothesis.

    Returns None when the candidate set exceeds ``ITEM_CANDIDATE_CAP`` so the
    caller can drop the whole (mode, op, fmt) hypothesis instead of letting the
    outer backtracking churn on a degenerate item.
    """
    input_symbols: list[str] = []
    for symbol in lhs[:2] + lhs[3:]:
        if symbol not in input_symbols:
            input_symbols.append(symbol)

    candidates: list[dict[str, str]] = []
    for values in itertools.permutations("0123456789", len(input_symbols)):
        mapping = dict(zip(input_symbols, values))
        ab_text = mapping[lhs[0]] + mapping[lhs[1]]
        cd_text = mapping[lhs[3]] + mapping[lhs[4]]
        x_value, y_value, x_text, y_text = numeric_operands(ab_text, cd_text, mode)
        value = numeric_op_value(x_value, y_value, x_text, y_text, op_name)
        decoded_rhs = format_numeric(value).get(fmt_name)
        if decoded_rhs is None or len(decoded_rhs) != len(rhs):
            continue

        local = dict(mapping)
        inverse = {digit: symbol for symbol, digit in local.items()}
        ok = True
        for symbol, digit in zip(rhs, decoded_rhs):
            if symbol in local and local[symbol] != digit:
                ok = False
                break
            if digit in inverse and inverse[digit] != symbol:
                ok = False
                break
            local[symbol] = digit
            inverse[digit] = symbol
        if ok:
            candidates.append(local)
            if len(candidates) > ITEM_CANDIDATE_CAP:
                return None
    return candidates


def solve_gold_cryptarithm(prompt: str, gold_answer: str):
    examples, target = encoded_equations(prompt)
    if not examples or target is None:
        return None
    target_op = target[2]
    relevant = [item for item in examples if item[0][2] == target_op]
    if not relevant:
        return None

    items = relevant + [(target, gold_answer)]
    for mode in CRYPT_MODES:
        for op_name in CRYPT_OPS:
            for fmt_name in CRYPT_FORMATS:
                candidate_lists: list[list[dict[str, str]]] = []
                for lhs, rhs in items:
                    maps = item_candidate_maps(lhs, rhs, mode, op_name, fmt_name)
                    if maps is None or not maps:
                        break
                    candidate_lists.append(maps)
                else:
                    order = sorted(range(len(candidate_lists)), key=lambda idx: len(candidate_lists[idx]))
                    solutions: list[dict[str, str]] = []

                    def backtrack(position: int, mapping: dict[str, str], inverse: dict[str, str]) -> None:
                        if len(solutions) > 1:
                            return
                        if position == len(order):
                            solutions.append(mapping)
                            return
                        for candidate in candidate_lists[order[position]]:
                            merged = merge_maps(mapping, inverse, candidate)
                            if merged is None:
                                continue
                            backtrack(position + 1, merged[0], merged[1])

                    backtrack(0, {}, {})
                    if solutions:
                        return {
                            "mode": mode,
                            "op_name": op_name,
                            "fmt_name": fmt_name,
                            "mapping": solutions[0],
                            "examples_used": len(relevant),
                            "candidate_counts": [len(items) for items in candidate_lists],
                        }
    return None


def oracle_trace(prompt: str, gold_answer: str, solution: dict) -> str:
    examples, target = encoded_equations(prompt)
    assert target is not None
    mapping = solution["mapping"]
    decoded_target = "".join(mapping.get(symbol, "?") for symbol in target[:2] + target[3:])
    return (
        "CRYPTARITHM_ORACLE\n"
        "This trace is gold-conditioned for supervised training: the known training answer "
        "is used as a constraint while recovering the latent digit mapping.\n"
        f"Target operator: {target[2]}; demonstrated target-operator examples: {solution['examples_used']}.\n"
        f"Recovered rule: operands={solution['mode']}; operation={solution['op_name']}; format={solution['fmt_name']}.\n"
        f"Recovered symbol mapping: {json.dumps(mapping, sort_keys=True, ensure_ascii=False)}.\n"
        f"Decoded target digits: {target[:2]} {target[2]} {target[3:]} -> {decoded_target[:2]} {target[2]} {decoded_target[2:]}.\n"
        f"Encoded final answer: {gold_answer}\n"
        f"Final answer: \\boxed{{{gold_answer}}}"
    )


def load_base_traces() -> dict[str, dict]:
    path = GENERATED_DIR / "reasoning_traces.jsonl"
    if not path.exists():
        return {}
    traces = {}
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            item = json.loads(line)
            item["oracle"] = False
            traces[item["id"]] = item
    return traces


def load_base_records() -> dict[str, dict]:
    path = REPORT_DIR / "solver_eval.json"
    if not path.exists():
        return {}
    return {item["id"]: item for item in json.loads(path.read_text(encoding="utf-8"))["records"]}


def load_existing_oracle_traces(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    traces: dict[str, dict] = {}
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            item = json.loads(line)
            traces[item["id"]] = item
    return traces


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    train["task_type"] = train["prompt"].map(classify_prompt)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    base_traces = load_base_traces()
    base_records = load_base_records()

    trace_path = GENERATED_DIR / "oracle_reasoning_traces.jsonl"
    partial_path = GENERATED_DIR / "oracle_reasoning_traces.partial.jsonl"
    existing_traces = load_existing_oracle_traces(trace_path)
    existing_traces.update(load_existing_oracle_traces(partial_path))

    report: dict[str, object] = {
        "base_correct": 0,
        "oracle_correct": 0,
        "total": len(train),
        "by_family": defaultdict(Counter),
    }
    written_traces: dict[str, dict] = {}
    if existing_traces:
        print(
            f"resume: reusing {len(existing_traces)} oracle traces "
            f"(from {trace_path.name} + {partial_path.name})"
        )

    new_oracle_count = 0
    progress_every = 500
    start_time = time.time()
    for idx, row in enumerate(train.itertuples(index=False), start=1):
        family_counter = report["by_family"][row.task_type]
        family_counter["total"] += 1
        if row.id in existing_traces:
            item = existing_traces[row.id]
            written_traces[row.id] = item
            report["oracle_correct"] += 1
            family_counter["oracle_correct"] += 1
            if item.get("oracle"):
                family_counter["cryptarithm_oracle"] += 1
            else:
                report["base_correct"] += 1
                family_counter["base_correct"] += 1
            if idx % progress_every == 0:
                print(
                    f"  [{idx}/{len(train)}] reuse={len(existing_traces)} "
                    f"new_oracle={new_oracle_count} elapsed={time.time() - start_time:.1f}s",
                    flush=True,
                )
            continue

        base_item = base_traces.get(row.id)
        if base_item is not None:
            report["base_correct"] += 1
            report["oracle_correct"] += 1
            family_counter["base_correct"] += 1
            family_counter["oracle_correct"] += 1
            written_traces[row.id] = base_item
            if idx % progress_every == 0:
                print(
                    f"  [{idx}/{len(train)}] reuse={len(existing_traces)} "
                    f"new_oracle={new_oracle_count} elapsed={time.time() - start_time:.1f}s",
                    flush=True,
                )
            continue

        base_record = base_records.get(row.id, {})
        if row.task_type == "symbolic_equation" and not base_record.get("correct", False):
            t0 = time.time()
            solution = solve_gold_cryptarithm(row.prompt, str(row.answer))
            dt = time.time() - t0
            if dt > 30.0:
                print(
                    f"  slow oracle solve id={row.id} dt={dt:.1f}s solved={solution is not None}",
                    flush=True,
                )
            if solution is not None:
                trace = oracle_trace(row.prompt, str(row.answer), solution)
                report["oracle_correct"] += 1
                family_counter["oracle_correct"] += 1
                family_counter["cryptarithm_oracle"] += 1
                new_oracle_count += 1
                written_traces[row.id] = {
                    "id": row.id,
                    "task_type": row.task_type,
                    "prompt": row.prompt,
                    "completion": trace,
                    "answer": str(row.answer),
                    "prediction": str(row.answer),
                    "method": "cryptarithm_gold_oracle",
                    "oracle": True,
                    "oracle_rule": {
                        "mode": solution["mode"],
                        "op_name": solution["op_name"],
                        "fmt_name": solution["fmt_name"],
                    },
                }

        if row.id in written_traces and row.id not in existing_traces:
            with partial_path.open("a", encoding="utf-8") as partial_file:
                partial_file.write(json.dumps(written_traces[row.id], ensure_ascii=False) + "\n")

        if idx % progress_every == 0:
            print(
                f"  [{idx}/{len(train)}] reuse={len(existing_traces)} "
                f"new_oracle={new_oracle_count} elapsed={time.time() - start_time:.1f}s",
                flush=True,
            )

    with trace_path.open("w", encoding="utf-8") as output:
        for row in train.itertuples(index=False):
            item = written_traces.get(row.id)
            if item is not None:
                output.write(json.dumps(item, ensure_ascii=False) + "\n")

    if partial_path.exists():
        partial_path.unlink()

    by_family = {}
    for family, counts in report["by_family"].items():
        total = counts["total"]
        by_family[family] = {
            "total": total,
            "base_correct": counts["base_correct"],
            "oracle_correct": counts["oracle_correct"],
            "base_accuracy": counts["base_correct"] / total if total else 0.0,
            "oracle_accuracy": counts["oracle_correct"] / total if total else 0.0,
            "methods": {key: value for key, value in counts.items() if key not in {"total", "base_correct", "oracle_correct"}},
        }

    summary = {
        "trace_path": str(trace_path),
        "base_correct": report["base_correct"],
        "oracle_correct": report["oracle_correct"],
        "total": report["total"],
        "base_accuracy": report["base_correct"] / report["total"],
        "oracle_accuracy": report["oracle_correct"] / report["total"],
        "by_family": by_family,
    }
    (REPORT_DIR / "oracle_trace_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps({key: summary[key] for key in ("base_correct", "oracle_correct", "total", "base_accuracy", "oracle_accuracy")}, indent=2))
    for family, item in sorted(by_family.items()):
        print(
            f"{family}: base {item['base_correct']}/{item['total']} = {item['base_accuracy']:.2%}; "
            f"oracle {item['oracle_correct']}/{item['total']} = {item['oracle_accuracy']:.2%}"
        )


if __name__ == "__main__":
    main()
