from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"


def classify_prompt(prompt: str) -> str:
    text = prompt.lower()
    if "bit manipulation" in text or "8-bit binary" in text:
        return "bit_manipulation"
    if "secret encryption" in text or "used on text" in text:
        return "text_cipher"
    if "different numeral system" in text:
        return "numeral_system"
    if "gravitational constant" in text:
        return "gravity_physics"
    if "unit conversion" in text:
        return "unit_conversion"
    if "transformation rules is applied to equations" in text:
        return "symbolic_equation"
    if "symbol" in text and "transformation" in text:
        return "symbol_transformation"
    if "equation" in text or "algebra" in text or "solve for" in text:
        return "algebra"
    if "sequence" in text:
        return "sequence"
    if "roman" in text:
        return "roman_numerals"
    if "base" in text and "number" in text:
        return "base_conversion"
    return "other"


def answer_type(answer: object) -> str:
    value = str(answer).strip()
    if re.fullmatch(r"[01]+", value):
        return "binary"
    if re.fullmatch(r"-?\d+(?:\.\d+)?", value):
        return "number"
    if re.fullmatch(r"[IVXLCDM]+", value, flags=re.I):
        return "roman"
    if re.fullmatch(r"[a-z ]+", value, flags=re.I):
        return "text"
    return "other"


def prompt_prefix(prompt: str, max_len: int = 180) -> str:
    compact = re.sub(r"\s+", " ", prompt).strip()
    return compact[:max_len]


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    train["task_type"] = train["prompt"].map(classify_prompt)
    train["answer_type"] = train["answer"].map(answer_type)
    test["task_type"] = test["prompt"].map(classify_prompt)

    overlap_by_id = set(test["id"]) & set(train["id"])
    overlap_by_prompt = set(test["prompt"]) & set(train["prompt"])
    train_lookup = train.set_index("id")

    test_matches = []
    for row in test.itertuples(index=False):
        if row.id in train_lookup.index:
            train_row = train_lookup.loc[row.id]
            test_matches.append(
                {
                    "id": row.id,
                    "same_prompt": bool(row.prompt == train_row["prompt"]),
                    "answer": train_row["answer"],
                    "task_type": train_row["task_type"],
                }
            )

    summary = {
        "rows": {"train": int(len(train)), "test_sample": int(len(test))},
        "columns": {
            "train": train.columns.tolist(),
            "test": test.columns.tolist(),
        },
        "overlap": {
            "test_ids_in_train": len(overlap_by_id),
            "test_prompts_in_train": len(overlap_by_prompt),
            "test_rows": int(len(test)),
            "all_sample_test_ids_in_train": len(overlap_by_id) == len(test),
        },
        "train_task_types": Counter(train["task_type"]).most_common(),
        "test_task_types": Counter(test["task_type"]).most_common(),
        "answer_types": Counter(train["answer_type"]).most_common(),
        "answer_length_top": Counter(train["answer"].astype(str).map(len)).most_common(20),
        "sample_test_matches": test_matches[:10],
    }

    families: dict[str, list[str]] = defaultdict(list)
    for task_type, group in train.groupby("task_type"):
        for prompt in group["prompt"].head(5):
            families[task_type].append(prompt_prefix(prompt))

    (REPORT_DIR / "data_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORT_DIR / "prompt_family_samples.json").write_text(
        json.dumps(families, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
