from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from trace_logprob_audit import TraceRecord, classify_prompt, normalize_text


ROOT = Path(__file__).resolve().parents[1]
LLKH0A_PROMPT_SUFFIX_RE = re.compile(
    r"\nPlease put your final answer inside `\\boxed\{\}`\. "
    r"For example: `\\boxed\{your answer\}`\s*$"
)
BOXED_VALUE_RE = re.compile(r"\\boxed\{([^{}]*)\}")
CANONICAL_FIELDS = [
    "id",
    "source",
    "task_type",
    "prompt",
    "answer",
    "completion",
    "method",
    "quality_flags",
]
DEFAULT_TARGETS = {
    "oracle": 360,
    "llkh0a": 280,
    "donald": 180,
    "bankoglu": 96,
    "sybyrr_beta": 240,
    "srivath": 120,
    "tong": 80,
}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    kind: str
    path: Path
    target: int


@dataclass
class Candidate:
    record: TraceRecord
    raw_completion: str
    raw_answer: str = ""
    raw_prompt: str = ""
    source_index: int = 0
    enabled_for_training: bool = True


@dataclass
class SourceStats:
    path: str
    target: int
    exists: bool = True
    raw_rows: int = 0
    train_joined: int = 0
    prompt_match: int = 0
    source_answer_match: int = 0
    boxed_checked: int = 0
    boxed_match: int = 0
    boxed_mismatch: int = 0
    boxed_appended: int = 0
    oracle_true: int = 0
    enabled_pre_sample: int = 0
    sampled: int = 0
    filter_flags: dict[str, int] = field(default_factory=dict)
    sampled_quality_flags: dict[str, int] = field(default_factory=dict)


def default_specs(root: Path) -> list[SourceSpec]:
    sybyrr_short = root / "tmp_kaggle_recon/datasets/sybyrr_symbolic"
    sybyrr_full = (
        root
        / "tmp_kaggle_recon/datasets/sybyrr_symbolic_full/beta_traces/beta_traces"
    )
    sybyrr_path = sybyrr_full if sybyrr_full.exists() else sybyrr_short
    return [
        SourceSpec(
            "oracle",
            "oracle_jsonl",
            root / "data/generated/oracle_reasoning_traces.jsonl",
            DEFAULT_TARGETS["oracle"],
        ),
        SourceSpec(
            "llkh0a",
            "llkh0a_csv",
            root
            / "reports/public_audit_2026-06-06/llkh0a__nemotron-unsloth-sft-training-3-30-2/output_small/formatted_train_dataset.csv",
            DEFAULT_TARGETS["llkh0a"],
        ),
        SourceSpec(
            "donald",
            "donald_csv",
            root
            / "tmp_kaggle_recon/datasets/leevvin_bitmanip/2026-05-04_donald_solvers_bit_manipulation.csv",
            DEFAULT_TARGETS["donald"],
        ),
        SourceSpec(
            "bankoglu",
            "bankoglu_csv",
            root / "tmp_kaggle_recon/datasets/bankoglu_hard/synthetic_hard_families.csv",
            DEFAULT_TARGETS["bankoglu"],
        ),
        SourceSpec("sybyrr_beta", "sybyrr_txt_dir", sybyrr_path, DEFAULT_TARGETS["sybyrr_beta"]),
        SourceSpec(
            "srivath",
            "srivath_csv",
            root / "tmp_kaggle_recon/datasets/srivath_enriched/enriched_train.csv",
            DEFAULT_TARGETS["srivath"],
        ),
        SourceSpec(
            "tong",
            "tong_csv",
            root
            / "tmp_kaggle_recon/datasets/leevvin_bitmanip/2026-05-06_tong_reasoning_bit_manipulation.csv",
            DEFAULT_TARGETS["tong"],
        ),
    ]


def parse_source_spec(raw: str, root: Path) -> SourceSpec:
    try:
        name, kind, path_text, target_text = raw.split(":", 3)
    except ValueError as exc:
        raise ValueError(
            "--source must be name:kind:path:target, for example "
            "donald:donald_csv:tmp/file.csv:180"
        ) from exc
    path = Path(path_text)
    if not path.is_absolute():
        path = root / path
    return SourceSpec(name, kind, path, int(target_text))


def read_train(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    by_id: dict[str, dict[str, str]] = {}
    by_prompt: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_id[row["id"]] = row
            by_prompt[normalize_prompt(row["prompt"])] = row
    return by_id, by_prompt


def normalize_prompt(value: object) -> str:
    return normalize_text(value).strip().replace("\r\n", "\n")


def normalize_answer(value: object) -> str:
    return normalize_text(value).strip()


def boxed_values(text: str) -> list[str]:
    return [match.group(1).strip() for match in BOXED_VALUE_RE.finditer(text)]


def final_boxed(text: str) -> str:
    boxes = [value for value in boxed_values(text) if value]
    return boxes[-1] if boxes else ""


def append_boxed(completion: str, answer: str) -> str:
    return completion.rstrip() + f"\n\n\\boxed{{{answer}}}"


def add_flag(flags: list[str], flag: str) -> None:
    if flag not in flags:
        flags.append(flag)


def join_train(
    train_by_id: dict[str, dict[str, str]],
    train_by_prompt: dict[str, dict[str, str]],
    row_id: str,
    prompt: str,
) -> dict[str, str] | None:
    if row_id and row_id in train_by_id:
        return train_by_id[row_id]
    normalized = normalize_prompt(prompt)
    return train_by_prompt.get(normalized)


def build_candidate(
    *,
    source: str,
    method: str,
    row_id: str,
    prompt: str,
    answer: str,
    completion: str,
    task_type: str,
    source_index: int,
    train_row: dict[str, str] | None,
    append_canonical_box: bool = False,
    disable_oracle_true: bool = False,
    force_disabled_flags: Iterable[str] = (),
) -> Candidate:
    flags: list[str] = []
    enabled = True
    raw_completion = completion
    raw_answer = normalize_answer(answer)
    raw_prompt = normalize_prompt(prompt)

    if train_row is None:
        add_flag(flags, "no_train_join")
        enabled = False
        canonical_id = row_id or f"row_{source_index:06d}"
        canonical_prompt = raw_prompt
        canonical_answer = raw_answer
    else:
        canonical_id = train_row["id"]
        canonical_prompt = normalize_prompt(train_row["prompt"])
        canonical_answer = normalize_answer(train_row["answer"])
        if raw_prompt and raw_prompt == canonical_prompt:
            add_flag(flags, "prompt_match")
        elif raw_prompt:
            add_flag(flags, "prompt_mismatch")
            enabled = False
        else:
            add_flag(flags, "prompt_from_train")
        if raw_answer:
            if raw_answer == canonical_answer:
                add_flag(flags, "source_answer_match")
            else:
                add_flag(flags, "source_answer_mismatch")
                enabled = False
        else:
            add_flag(flags, "answer_from_train")

    boxes = boxed_values(raw_completion)
    boxed_answer = final_boxed(raw_completion)
    if boxes:
        if len(boxes) > 1:
            add_flag(flags, "multiple_boxed")
        if boxed_answer == canonical_answer:
            add_flag(flags, "final_boxed_match")
        else:
            add_flag(flags, "final_boxed_mismatch")
            enabled = False
    elif append_canonical_box and canonical_answer:
        completion = append_boxed(raw_completion, canonical_answer)
        add_flag(flags, "boxed_appended")
    else:
        add_flag(flags, "no_boxed")
        enabled = False

    if disable_oracle_true:
        add_flag(flags, "oracle_true")
        enabled = False
    for flag in force_disabled_flags:
        add_flag(flags, flag)
        enabled = False

    record = TraceRecord(
        id=canonical_id,
        source=source,
        task_type=task_type or classify_prompt(canonical_prompt),
        prompt=canonical_prompt,
        answer=canonical_answer,
        completion=completion,
        method=method,
        quality_flags=";".join(flags),
    )
    return Candidate(
        record=record,
        raw_completion=raw_completion,
        raw_answer=raw_answer,
        raw_prompt=raw_prompt,
        source_index=source_index,
        enabled_for_training=enabled,
    )


def load_oracle(
    spec: SourceSpec,
    train_by_id: dict[str, dict[str, str]],
    train_by_prompt: dict[str, dict[str, str]],
) -> Iterable[Candidate]:
    with spec.path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            oracle_true = normalize_text(row.get("oracle")).lower() == "true"
            yield build_candidate(
                source=spec.name,
                method=normalize_text(row.get("method")) or "oracle_jsonl",
                row_id=normalize_text(row.get("id")),
                prompt=normalize_text(row.get("prompt")),
                answer=normalize_text(row.get("answer")),
                completion=normalize_text(row.get("completion")),
                task_type=normalize_text(row.get("task_type")),
                source_index=idx,
                train_row=join_train(
                    train_by_id,
                    train_by_prompt,
                    normalize_text(row.get("id")),
                    normalize_text(row.get("prompt")),
                ),
                disable_oracle_true=oracle_true,
            )


def load_llkh0a(
    spec: SourceSpec,
    train_by_id: dict[str, dict[str, str]],
    train_by_prompt: dict[str, dict[str, str]],
) -> Iterable[Candidate]:
    del train_by_id
    with spec.path.open(newline="", encoding="utf-8") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            raw_prompt = normalize_text(row.get("user_content"))
            prompt = LLKH0A_PROMPT_SUFFIX_RE.sub("", raw_prompt)
            completion = normalize_text(row.get("assistant_content"))
            answer = final_boxed(completion)
            yield build_candidate(
                source=spec.name,
                method="llkh0a_formatted",
                row_id=f"row_{idx:06d}",
                prompt=prompt,
                answer=answer,
                completion=completion,
                task_type=classify_prompt(prompt),
                source_index=idx,
                train_row=train_by_prompt.get(normalize_prompt(prompt)),
            )


def load_generic_csv(
    spec: SourceSpec,
    train_by_id: dict[str, dict[str, str]],
    train_by_prompt: dict[str, dict[str, str]],
    *,
    completion_field: str,
    task_field: str,
    method: str,
    append_canonical_box: bool = False,
) -> Iterable[Candidate]:
    with spec.path.open(newline="", encoding="utf-8") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            row_id = normalize_text(row.get("id"))
            prompt = normalize_text(row.get("prompt"))
            answer = normalize_text(row.get("answer"))
            completion = normalize_text(row.get(completion_field))
            task_type = normalize_text(row.get(task_field)) or classify_prompt(prompt)
            train_row = join_train(train_by_id, train_by_prompt, row_id, prompt)
            yield build_candidate(
                source=spec.name,
                method=method,
                row_id=row_id,
                prompt=prompt,
                answer=answer,
                completion=completion,
                task_type=canonical_task_type(task_type),
                source_index=idx,
                train_row=train_row,
                append_canonical_box=append_canonical_box,
            )


def load_sybyrr_txt(
    spec: SourceSpec,
    train_by_id: dict[str, dict[str, str]],
    train_by_prompt: dict[str, dict[str, str]],
) -> Iterable[Candidate]:
    del train_by_prompt
    force_disabled_flags: list[str] = []
    if "sybyrr_reasoning_full" in spec.path.parts:
        force_disabled_flags.append("sybyrr_reasoning_full_disabled")
    for idx, path in enumerate(sorted(spec.path.glob("*.txt"))):
        train_row = train_by_id.get(path.stem)
        prompt = train_row["prompt"] if train_row else ""
        answer = train_row["answer"] if train_row else ""
        yield build_candidate(
            source=spec.name,
            method="sybyrr_symbolic_beta",
            row_id=path.stem,
            prompt=prompt,
            answer=answer,
            completion=path.read_text(encoding="utf-8"),
            task_type="symbolic_equation",
            source_index=idx,
            train_row=train_row,
            force_disabled_flags=force_disabled_flags,
        )


def canonical_task_type(task_type: str) -> str:
    mapping = {
        "gravity": "gravity_physics",
        "cipher": "text_cipher",
        "numeral": "numeral_system",
        "equation_or_cryptarithm": "symbolic_equation",
    }
    return mapping.get(task_type, task_type)


def loaders() -> dict[str, Callable[..., Iterable[Candidate]]]:
    return {
        "oracle_jsonl": load_oracle,
        "llkh0a_csv": load_llkh0a,
        "donald_csv": lambda spec, by_id, by_prompt: load_generic_csv(
            spec,
            by_id,
            by_prompt,
            completion_field="cot_trace",
            task_field="task",
            method="donald_solvers_bit_manipulation",
            append_canonical_box=True,
        ),
        "bankoglu_csv": lambda spec, by_id, by_prompt: load_generic_csv(
            spec,
            by_id,
            by_prompt,
            completion_field="generated_cot",
            task_field="type",
            method="bankoglu_hard_families",
        ),
        "sybyrr_txt_dir": load_sybyrr_txt,
        "srivath_csv": lambda spec, by_id, by_prompt: load_generic_csv(
            spec,
            by_id,
            by_prompt,
            completion_field="generated_cot",
            task_field="category",
            method="srivath_enriched",
            append_canonical_box=True,
        ),
        "tong_csv": lambda spec, by_id, by_prompt: load_generic_csv(
            spec,
            by_id,
            by_prompt,
            completion_field="cot_trace",
            task_field="task",
            method="tong_reasoning_bit_manipulation",
        ),
    }


def load_candidates(
    specs: list[SourceSpec],
    train_by_id: dict[str, dict[str, str]],
    train_by_prompt: dict[str, dict[str, str]],
) -> tuple[dict[str, list[Candidate]], dict[str, SourceStats]]:
    candidates_by_source: dict[str, list[Candidate]] = {}
    stats_by_source: dict[str, SourceStats] = {}
    loader_by_kind = loaders()
    for spec in specs:
        stats = SourceStats(path=str(spec.path), target=spec.target, exists=spec.path.exists())
        stats_by_source[spec.name] = stats
        if not spec.path.exists():
            continue
        if spec.kind not in loader_by_kind:
            raise ValueError(f"Unsupported source kind {spec.kind!r}")
        source_candidates = list(loader_by_kind[spec.kind](spec, train_by_id, train_by_prompt))
        candidates_by_source[spec.name] = source_candidates
        update_stats(stats, source_candidates)
    return candidates_by_source, stats_by_source


def update_stats(stats: SourceStats, candidates: list[Candidate]) -> None:
    stats.raw_rows = len(candidates)
    flag_counts = Counter(
        flag
        for candidate in candidates
        for flag in split_flags(candidate.record.quality_flags)
    )
    stats.train_joined = stats.raw_rows - flag_counts.get("no_train_join", 0)
    stats.prompt_match = flag_counts.get("prompt_match", 0) + flag_counts.get("prompt_from_train", 0)
    stats.source_answer_match = flag_counts.get("source_answer_match", 0) + flag_counts.get(
        "answer_from_train", 0
    )
    stats.boxed_checked = stats.raw_rows - flag_counts.get("boxed_appended", 0) - flag_counts.get(
        "no_boxed", 0
    )
    stats.boxed_match = flag_counts.get("final_boxed_match", 0)
    stats.boxed_mismatch = flag_counts.get("final_boxed_mismatch", 0)
    stats.boxed_appended = flag_counts.get("boxed_appended", 0)
    stats.oracle_true = flag_counts.get("oracle_true", 0)
    stats.enabled_pre_sample = sum(1 for candidate in candidates if candidate.enabled_for_training)
    disabled_flags = Counter(
        flag
        for candidate in candidates
        if not candidate.enabled_for_training
        for flag in split_flags(candidate.record.quality_flags)
        if is_filter_flag(flag)
    )
    stats.filter_flags = dict(disabled_flags.most_common())


def is_filter_flag(flag: str) -> bool:
    return flag in {
        "no_train_join",
        "prompt_mismatch",
        "source_answer_mismatch",
        "final_boxed_mismatch",
        "no_boxed",
        "oracle_true",
        "sybyrr_reasoning_full_disabled",
    }


def split_flags(value: str) -> list[str]:
    return [flag for flag in value.split(";") if flag]


def sample_candidates(
    candidates_by_source: dict[str, list[Candidate]],
    stats_by_source: dict[str, SourceStats],
    seed: int,
    include_disabled: bool,
) -> list[TraceRecord]:
    sampled: list[TraceRecord] = []
    for source, candidates in candidates_by_source.items():
        stats = stats_by_source[source]
        pool = candidates if include_disabled else [c for c in candidates if c.enabled_for_training]
        target = stats.target
        if target > 0 and len(pool) > target:
            rng = random.Random(f"{seed}:{source}")
            selected_indexes = set(rng.sample(range(len(pool)), target))
            source_sample = [candidate for i, candidate in enumerate(pool) if i in selected_indexes]
        else:
            source_sample = pool
        source_sample.sort(key=lambda candidate: candidate.source_index)
        stats.sampled = len(source_sample)
        stats.sampled_quality_flags = dict(
            Counter(
                flag
                for candidate in source_sample
                for flag in split_flags(candidate.record.quality_flags)
            ).most_common()
        )
        sampled.extend(candidate.record for candidate in source_sample)
    return sampled


def write_manifest(records: list[TraceRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            row = {field_name: getattr(record, field_name) for field_name in CANONICAL_FIELDS}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(
    records: list[TraceRecord],
    stats_by_source: dict[str, SourceStats],
    path: Path,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "output": str(args.output),
        "seed": args.seed,
        "include_disabled": args.include_disabled,
        "rows": len(records),
        "canonical_fields": CANONICAL_FIELDS,
        "by_source": dict(Counter(record.source for record in records).most_common()),
        "by_task_type": dict(Counter(record.task_type for record in records).most_common()),
        "quality_flags": dict(
            Counter(
                flag
                for record in records
                for flag in split_flags(record.quality_flags)
            ).most_common()
        ),
        "sources": {
            source: asdict(stats)
            for source, stats in sorted(stats_by_source.items(), key=lambda item: item[0])
        },
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def print_summary(records: list[TraceRecord], stats_by_source: dict[str, SourceStats]) -> None:
    print(f"records={len(records)}")
    print("by_source", dict(Counter(record.source for record in records).most_common()))
    print("by_task_type", dict(Counter(record.task_type for record in records).most_common()))
    print(
        "quality_flags",
        dict(
            Counter(
                flag
                for record in records
                for flag in split_flags(record.quality_flags)
            ).most_common(30)
        ),
    )
    print("filter_stats")
    for source in sorted(stats_by_source):
        stats = stats_by_source[source]
        print(
            source,
            {
                "raw_rows": stats.raw_rows,
                "enabled_pre_sample": stats.enabled_pre_sample,
                "sampled": stats.sampled,
                "target": stats.target,
                "boxed_mismatch": stats.boxed_mismatch,
                "boxed_appended": stats.boxed_appended,
                "filter_flags": stats.filter_flags,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a cleaned, sampled trace manifest for first GPU audit."
    )
    parser.add_argument("--train-csv", type=Path, default=ROOT / "data/train.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/trace_manifest_first_gpu_audit.jsonl",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "reports/trace_manifest_first_gpu_audit_summary.json",
    )
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Optional source override as name:kind:path:target. Repeatable.",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Include rows that default training filters disable, for audit-only manifests.",
    )
    args = parser.parse_args()

    root = ROOT
    specs = (
        [parse_source_spec(source, root) for source in args.source]
        if args.source
        else default_specs(root)
    )
    train_by_id, train_by_prompt = read_train(args.train_csv)
    candidates_by_source, stats_by_source = load_candidates(specs, train_by_id, train_by_prompt)
    records = sample_candidates(
        candidates_by_source,
        stats_by_source,
        seed=args.seed,
        include_disabled=args.include_disabled,
    )
    write_manifest(records, args.output)
    write_summary(records, stats_by_source, args.summary_json, args)
    print_summary(records, stats_by_source)
    print(f"wrote {args.output}")
    print(f"wrote {args.summary_json}")


if __name__ == "__main__":
    main()
