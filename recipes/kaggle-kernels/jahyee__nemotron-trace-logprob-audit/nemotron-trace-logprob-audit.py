from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import types
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parents[1]
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
EMBEDDED_TRACE_LOGPROB_AUDIT_SOURCE: str | None = None
EMBEDDED_TRACE_MANIFEST_JSONL: str | None = None

MODEL_SOURCE = "metric/nemotron-3-nano-30b-a3b-bf16"
ADAPTER_SOURCE = "kienngx/nemotron-nano-30b-trained"
COMPETITION = "nvidia-nemotron-model-reasoning-challenge"

DATASET_SOURCES = {
    "jahyee/nemotron-oracle-reasoning-traces": [
        WORKSPACE_ROOT / "data" / "generated",
    ],
    "kienngx/nemotron-30b-competition-trainingdata-cot-labels": [],
    "jahyee/nemotron-solver-cot-augmentation": [
        WORKSPACE_ROOT / "data" / "phase2_kaggle_dataset",
    ],
    "leevvin/nemotron-bitmanip-traces-pub": [
        WORKSPACE_ROOT / "tmp_kaggle_recon" / "datasets" / "leevvin_bitmanip",
    ],
    "sybyrr/nemotron-symbolic-beta-trace": [
        WORKSPACE_ROOT
        / "tmp_kaggle_recon"
        / "datasets"
        / "sybyrr_symbolic_full"
        / "beta_traces"
        / "beta_traces",
        WORKSPACE_ROOT / "tmp_kaggle_recon" / "datasets" / "sybyrr_symbolic",
    ],
    "gdataranger/huikang-nemotron-nemo-sft-r32": [
        WORKSPACE_ROOT / "tmp_kaggle_recon" / "datasets" / "gdataranger_huikang",
        WORKSPACE_ROOT
        / "public_radar"
        / "datasets"
        / "gdataranger__huikang-nemotron-nemo-sft-r32",
    ],
}


@dataclass(frozen=True)
class AuditJob:
    label: str
    dataset: str
    kind: str
    path: Path
    task_type: str | None = None
    require_prompt: bool = True
    require_answer: bool = True

    @property
    def source_arg(self) -> str:
        return f"{self.kind}:{self.path}"


def load_audit_module():
    path = HERE / "trace_logprob_audit.py"
    if path.exists():
        spec = importlib.util.spec_from_file_location("trace_logprob_audit", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    if EMBEDDED_TRACE_LOGPROB_AUDIT_SOURCE:
        module = types.ModuleType("trace_logprob_audit")
        module.__file__ = "<embedded trace_logprob_audit.py>"
        sys.modules[module.__name__] = module
        exec(
            compile(EMBEDDED_TRACE_LOGPROB_AUDIT_SOURCE, module.__file__, "exec"),
            module.__dict__,
        )
        return module

    raise FileNotFoundError(path)


def local_manifest_path() -> Path | None:
    candidates = [
        HERE / "trace_manifest_first_gpu_audit.jsonl",
        KAGGLE_WORKING / "trace_manifest_first_gpu_audit.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    if EMBEDDED_TRACE_MANIFEST_JSONL:
        output = (
            KAGGLE_WORKING / "trace_manifest_first_gpu_audit.jsonl"
            if KAGGLE_WORKING.exists()
            else HERE / "trace_manifest_first_gpu_audit.jsonl"
        )
        output.write_text(EMBEDDED_TRACE_MANIFEST_JSONL, encoding="utf-8")
        return output
    return None


def slug_tail(source: str) -> str:
    return source.split("/", 1)[1]


def kaggle_dataset_roots(source: str) -> list[Path]:
    owner, slug = source.split("/", 1)
    return [
        KAGGLE_INPUT / "datasets" / owner / slug,
        KAGGLE_INPUT / slug,
        KAGGLE_INPUT / source.replace("/", "__"),
        KAGGLE_INPUT / source.replace("/", "-"),
    ]


def existing_roots(source: str) -> list[Path]:
    candidates = [*kaggle_dataset_roots(source), *DATASET_SOURCES.get(source, [])]
    seen: set[Path] = set()
    roots: list[Path] = []
    for root in candidates:
        root = root.resolve()
        if root.exists() and root not in seen:
            seen.add(root)
            roots.append(root)
    return roots


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def resolve_train_csv() -> Path:
    env = os.environ.get("AUDIT_TRAIN_CSV")
    candidates = [
        Path(env).expanduser() if env else None,
        KAGGLE_INPUT / "competitions" / COMPETITION / "train.csv",
        KAGGLE_INPUT / COMPETITION / "train.csv",
        WORKSPACE_ROOT / "data" / "train.csv",
    ]
    path = first_existing(p for p in candidates if p is not None)
    if path is None:
        raise FileNotFoundError("Could not find train.csv for txt trace prompt/answer lookup")
    return path


def has_files(path: Path, names: set[str]) -> bool:
    if not path.is_dir():
        return False
    present = {p.name for p in path.iterdir() if p.is_file()}
    return names <= present


def resolve_model_path() -> Path:
    env = os.environ.get("AUDIT_MODEL_PATH")
    exact = [
        Path(env).expanduser() if env else None,
        KAGGLE_INPUT
        / "models"
        / "metric"
        / "nemotron-3-nano-30b-a3b-bf16"
        / "transformers"
        / "default"
        / "1",
        KAGGLE_INPUT
        / "models"
        / "metric"
        / "nemotron-3-nano-30b-a3b-bf16"
        / "Transformers"
        / "default"
        / "1",
    ]
    path = first_existing(p for p in exact if p is not None)
    if path is not None:
        return path

    for root in [KAGGLE_INPUT / "models", KAGGLE_INPUT, WORKSPACE_ROOT]:
        if not root.exists():
            continue
        for config in root.rglob("config.json"):
            text = str(config.parent).lower()
            if "nemotron-3-nano-30b-a3b-bf16" in text:
                return config.parent
    raise FileNotFoundError(f"Could not resolve model path for {MODEL_SOURCE}")


def resolve_adapter_path() -> Path:
    env = os.environ.get("AUDIT_ADAPTER_PATH")
    exact = [
        Path(env).expanduser() if env else None,
        KAGGLE_INPUT
        / "models"
        / "kienngx"
        / "nemotron-nano-30b-trained"
        / "triton"
        / "tinker-adapter"
        / "1",
        KAGGLE_INPUT
        / "models"
        / "kienngx"
        / "nemotron-nano-30b-trained"
        / "Triton"
        / "tinker-adapter"
        / "1",
    ]
    path = first_existing(p for p in exact if p is not None)
    if path is not None:
        return path

    for root in [KAGGLE_INPUT / "models", KAGGLE_INPUT, WORKSPACE_ROOT]:
        if not root.exists():
            continue
        adapter_dirs = [
            p.parent
            for p in root.rglob("adapter_config.json")
            if has_files(p.parent, {"adapter_config.json", "adapter_model.safetensors"})
        ]
        preferred = [p for p in adapter_dirs if "tinker-adapter" in str(p).lower()]
        if preferred:
            return sorted(preferred)[0]
        kienngx = [p for p in adapter_dirs if "kienngx" in str(p).lower()]
        if kienngx:
            return sorted(kienngx)[0]
    raise FileNotFoundError(f"Could not resolve adapter path for {ADAPTER_SOURCE}")


def csv_columns(path: Path) -> set[str]:
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return set(reader.fieldnames or [])
    except UnicodeDecodeError:
        return set()


def jsonl_keys(path: Path) -> set[str]:
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    return set(row) if isinstance(row, dict) else set()
    except (UnicodeDecodeError, json.JSONDecodeError):
        return set()
    return set()


def best_txt_dir(root: Path) -> Path | None:
    if root.is_file():
        return None
    candidates = []
    for path in [root, *root.rglob("*")]:
        if path.is_dir():
            count = sum(1 for f in path.glob("*.txt") if f.name.lower() != "readme.md")
            if count:
                candidates.append((count, len(path.parts), path))
    if not candidates:
        return None
    return sorted(candidates, reverse=True)[0][2]


def discover_jobs(limit_file_per_dataset: int = 4) -> tuple[list[AuditJob], list[dict[str, object]]]:
    jobs: list[AuditJob] = []
    notes: list[dict[str, object]] = []

    local_manifest = local_manifest_path()
    if local_manifest is not None and local_manifest.exists():
        jobs.append(
            AuditJob(
                "local_manifest_first_gpu_audit",
                "local/trace_manifest_first_gpu_audit",
                "oracle_jsonl",
                local_manifest,
            )
        )

    for dataset in DATASET_SOURCES:
        roots = existing_roots(dataset)
        notes.append({"dataset": dataset, "roots": [str(r) for r in roots]})
        if not roots:
            continue

        added = 0
        for root in roots:
            lower_dataset = dataset.lower()
            if "sybyrr" in lower_dataset:
                txt_dir = best_txt_dir(root)
                if txt_dir is not None:
                    jobs.append(AuditJob(f"{slug_tail(dataset)}__txt", dataset, "txt_dir", txt_dir))
                    added += 1
                continue

            files = sorted(p for p in root.rglob("*") if p.is_file())
            priority = sorted(
                files,
                key=lambda p: (
                    "oracle_reasoning_traces" not in p.name,
                    "tong_reasoning_bit_manipulation" not in p.name,
                    "donald_solvers_bit_manipulation" not in p.name,
                    "codelion" not in p.name.lower(),
                    str(p),
                ),
            )
            for path in priority:
                if added >= limit_file_per_dataset:
                    break
                suffix = path.suffix.lower()
                if suffix == ".jsonl":
                    keys = jsonl_keys(path)
                    if {"input_ids", "labels"} <= keys:
                        kind = "nemo_jsonl"
                    elif {"prompt", "completion", "answer"} <= keys:
                        kind = "oracle_jsonl"
                    else:
                        notes.append(
                            {
                                "dataset": dataset,
                                "skipped": str(path),
                                "reason": f"jsonl keys not auditable: {sorted(keys)}",
                            }
                        )
                        continue
                elif suffix == ".csv":
                    cols = csv_columns(path)
                    has_prompt = "prompt" in cols or "user_content" in cols
                    has_completion = bool(
                        {"completion", "generated_cot", "cot_trace", "assistant_content"} & cols
                    )
                    has_answer = "answer" in cols
                    if not (has_prompt and has_completion and has_answer):
                        notes.append(
                            {
                                "dataset": dataset,
                                "skipped": str(path),
                                "reason": f"csv columns not auditable: {sorted(cols)}",
                            }
                        )
                        continue
                    kind = "llkh0a_formatted" if "formatted_train_dataset" in path.name else "generic_csv"
                else:
                    continue

                label = f"{slug_tail(dataset)}__{path.stem}".replace("/", "_")
                jobs.append(AuditJob(label, dataset, kind, path))
                added += 1
    return dedupe_jobs(jobs), notes


def dedupe_jobs(jobs: list[AuditJob]) -> list[AuditJob]:
    seen: set[tuple[str, Path]] = set()
    label_counts: Counter[str] = Counter()
    out: list[AuditJob] = []
    for job in jobs:
        key = (job.kind, job.path.resolve())
        if key in seen:
            continue
        seen.add(key)
        label_counts[job.label] += 1
        if label_counts[job.label] > 1:
            suffix = job.path.name or job.path.parent.name
            job = AuditJob(
                label=f"{job.label}__{suffix}",
                dataset=job.dataset,
                kind=job.kind,
                path=job.path,
                task_type=job.task_type,
                require_prompt=job.require_prompt,
                require_answer=job.require_answer,
            )
        out.append(job)
    return out


def run_pip(args: list[str]) -> None:
    print("$", " ".join(args), flush=True)
    subprocess.check_call(args)


def maybe_install_offline_deps(skip: bool) -> None:
    if skip:
        return
    try:
        import transformers  # noqa: F401
        import peft  # noqa: F401
        import safetensors  # noqa: F401
    except Exception:
        package_dirs = [
            KAGGLE_INPUT / "datasets" / "mayukh18" / "nemotron-packages" / "packages",
            KAGGLE_INPUT / "nemotron-packages" / "packages",
        ]
        find_links = first_existing(package_dirs)
        if find_links is None:
            raise
        run_pip(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--no-index",
                "--find-links",
                str(find_links),
                "transformers",
                "peft",
                "accelerate",
                "safetensors",
            ]
        )

    wheel_roots = [
        KAGGLE_INPUT / "datasets" / "mayukh18" / "nemotron-packages",
        KAGGLE_INPUT / "nemotron-packages",
        KAGGLE_INPUT / "datasets" / "llkh0a" / "rtx-wheels" / "wheels",
        KAGGLE_INPUT / "rtx-wheels" / "wheels",
    ]
    wheels: list[Path] = []
    for root in wheel_roots:
        if root.exists():
            wheels.extend(sorted(root.rglob("causal_conv1d*.whl")))
            wheels.extend(sorted(root.rglob("mamba_ssm*.whl")))
    for wheel in wheels[:2]:
        module = "causal_conv1d" if wheel.name.startswith("causal_conv1d") else "mamba_ssm"
        try:
            __import__(module)
        except Exception:
            run_pip([sys.executable, "-m", "pip", "install", "-q", str(wheel)])


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_file(src: Path, dst) -> None:
    with src.open(encoding="utf-8") as f:
        shutil.copyfileobj(f, dst)


def aggregate_worst_tokens(jsonl_path: Path, csv_path: Path, json_path: Path) -> None:
    groups: dict[tuple[str, str], dict[str, object]] = {}
    by_task: Counter[str] = Counter()
    by_source: Counter[str] = Counter()

    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            token_id = row.get("worst_token_id")
            token_text = row.get("worst_token_text")
            min_lp = row.get("min_logprob")
            if token_id is None or token_text is None or not isinstance(min_lp, (int, float)):
                continue
            key = (str(token_id), str(token_text))
            bucket = groups.setdefault(
                key,
                {
                    "worst_token_id": token_id,
                    "worst_token_text": token_text,
                    "count": 0,
                    "min_logprob_sum": 0.0,
                    "min_logprob_min": float(min_lp),
                    "sources": Counter(),
                    "task_types": Counter(),
                    "examples": [],
                },
            )
            bucket["count"] = int(bucket["count"]) + 1
            bucket["min_logprob_sum"] = float(bucket["min_logprob_sum"]) + float(min_lp)
            bucket["min_logprob_min"] = min(float(bucket["min_logprob_min"]), float(min_lp))
            bucket["sources"][str(row.get("source"))] += 1
            bucket["task_types"][str(row.get("task_type"))] += 1
            by_source[str(row.get("source"))] += 1
            by_task[str(row.get("task_type"))] += 1
            examples = bucket["examples"]
            if isinstance(examples, list) and len(examples) < 5:
                examples.append(row.get("id"))

    rows = []
    for bucket in groups.values():
        count = int(bucket["count"])
        rows.append(
            {
                "worst_token_id": bucket["worst_token_id"],
                "worst_token_text": bucket["worst_token_text"],
                "count": count,
                "min_logprob_mean": float(bucket["min_logprob_sum"]) / count,
                "min_logprob_min": bucket["min_logprob_min"],
                "top_sources": json.dumps(dict(bucket["sources"].most_common(5)), ensure_ascii=False),
                "top_task_types": json.dumps(dict(bucket["task_types"].most_common(5)), ensure_ascii=False),
                "example_ids": json.dumps(bucket["examples"], ensure_ascii=False),
            }
        )
    rows.sort(key=lambda r: (-int(r["count"]), float(r["min_logprob_min"])))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "worst_token_id",
            "worst_token_text",
            "count",
            "min_logprob_mean",
            "min_logprob_min",
            "top_sources",
            "top_task_types",
            "example_ids",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_json(
        json_path,
        {
            "rows": len(rows),
            "total_scored_rows": sum(int(r["count"]) for r in rows),
            "by_source": dict(by_source.most_common()),
            "by_task_type": dict(by_task.most_common()),
            "top_worst_tokens": rows[:50],
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle launcher for small Nemotron trace logprob audits.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit-per-job", type=int, default=int(os.environ.get("AUDIT_LIMIT_PER_JOB", "64")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("AUDIT_BATCH_SIZE", "1")))
    parser.add_argument("--max-length", type=int, default=int(os.environ.get("AUDIT_MAX_LENGTH", "8192")))
    parser.add_argument("--content-skip-tokens", type=int, default=int(os.environ.get("AUDIT_CONTENT_SKIP_TOKENS", "16")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("AUDIT_SEED", "42")))
    parser.add_argument("--dry-run", action="store_true", default=os.environ.get("AUDIT_DRY_RUN") == "1")
    parser.add_argument("--no-tokenizer", action="store_true", default=os.environ.get("AUDIT_NO_TOKENIZER") == "1")
    parser.add_argument("--skip-install", action="store_true", default=os.environ.get("AUDIT_SKIP_INSTALL") == "1")
    parser.add_argument("--max-jobs", type=int, default=int(os.environ.get("AUDIT_MAX_JOBS", "0")))
    parser.add_argument(
        "--job-filter",
        default=os.environ.get("AUDIT_JOB_FILTER", "local_manifest_first_gpu_audit"),
        help="Comma-separated substrings; keep jobs whose label/path/dataset contains any substring.",
    )
    return parser.parse_args()


def filter_jobs(jobs: list[AuditJob], filter_text: str) -> list[AuditJob]:
    needles = [part.strip().lower() for part in filter_text.split(",") if part.strip()]
    if not needles:
        return jobs
    kept: list[AuditJob] = []
    for job in jobs:
        haystack = " ".join([job.label, job.dataset, job.kind, str(job.path)]).lower()
        if any(needle in haystack for needle in needles):
            kept.append(job)
    return kept


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (KAGGLE_WORKING if KAGGLE_WORKING.exists() else HERE / "local_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    audit = load_audit_module()
    train_csv = resolve_train_csv()
    label_jsonl = first_existing(
        [
            Path(os.environ["AUDIT_LABEL_JSONL"]).expanduser()
            for _ in [0]
            if os.environ.get("AUDIT_LABEL_JSONL")
        ]
        + [
            root / "valid_labels.jsonl"
            for root in existing_roots("gdataranger/huikang-nemotron-nemo-sft-r32")
        ]
    )
    jobs, discovery_notes = discover_jobs()
    jobs = filter_jobs(jobs, args.job_filter)
    if args.max_jobs > 0:
        jobs = jobs[: args.max_jobs]

    discovery = {
        "train_csv": str(train_csv),
        "label_jsonl": str(label_jsonl) if label_jsonl else None,
        "limit_per_job": args.limit_per_job,
        "job_filter": args.job_filter,
        "jobs": [job.__dict__ | {"path": str(job.path)} for job in jobs],
        "notes": discovery_notes,
    }
    write_json(output_dir / "discovery_report.json", discovery)

    if not jobs:
        raise RuntimeError("No auditable trace sources were discovered")

    model_path = None if args.no_tokenizer else resolve_model_path()
    adapter_path = None if args.dry_run or args.no_tokenizer else resolve_adapter_path()
    write_json(
        output_dir / "run_config.json",
        {
            "model_path": str(model_path) if model_path else None,
            "adapter_path": str(adapter_path) if adapter_path else None,
            "train_csv": str(train_csv),
            "label_jsonl": str(label_jsonl) if label_jsonl else None,
            "dry_run": args.dry_run,
            "no_tokenizer": args.no_tokenizer,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "content_skip_tokens": args.content_skip_tokens,
            "seed": args.seed,
            "limit_per_job": args.limit_per_job,
            "job_filter": args.job_filter,
        },
    )

    maybe_install_offline_deps(args.skip_install or args.no_tokenizer)

    tokenizer = None
    model = None
    adapter_tmp = None
    if not args.no_tokenizer:
        tokenizer = audit.load_tokenizer(str(model_path))
        if not args.dry_run:
            model, adapter_tmp = audit.load_model(str(model_path), adapter_path)

    combined_jsonl = output_dir / "trace_logprob_audit.jsonl"
    if combined_jsonl.exists():
        combined_jsonl.unlink()

    manifest: list[dict[str, object]] = []
    try:
        with combined_jsonl.open("w", encoding="utf-8") as combined:
            for idx, job in enumerate(jobs, start=1):
                print(f"[{idx}/{len(jobs)}] {job.label}: {job.source_arg}", flush=True)
                records, tokenized_records = audit.load_records(
                    [job.source_arg],
                    train_csv,
                    label_jsonl if job.kind == "nemo_jsonl" else None,
                )
                records = audit.filter_records(
                    records,
                    task_types=[job.task_type] if job.task_type else None,
                    sources=None,
                    methods=None,
                    ids=None,
                    require_prompt=job.require_prompt,
                    require_answer=job.require_answer,
                    drop_flag=None,
                )
                tokenized_records = audit.filter_tokenized_records(
                    tokenized_records,
                    task_types=[job.task_type] if job.task_type else None,
                    sources=None,
                    methods=None,
                    ids=None,
                    require_answer=job.require_answer,
                    drop_flag=None,
                )
                records = audit.sample_records(records, args.limit_per_job, args.seed + idx)
                tokenized_records = audit.sample_tokenized_records(
                    tokenized_records,
                    args.limit_per_job,
                    args.seed + idx,
                )
                if tokenizer is None:
                    examples = [
                        audit.build_pretokenized_example(record, args.max_length)
                        for record in tokenized_records
                    ]
                else:
                    examples = [
                        audit.build_tokenized_example(tokenizer, record, args.max_length, audit.DEFAULT_PROMPT_SUFFIX)
                        for record in records
                    ]
                    examples.extend(
                        audit.build_pretokenized_example(record, args.max_length)
                        for record in tokenized_records
                    )
                all_records = records + [record.record for record in tokenized_records]
                audit.print_summary(all_records, examples)

                job_jsonl = output_dir / f"{job.label}.jsonl"
                job_csv = output_dir / f"{job.label}.csv"
                job_summary = output_dir / f"{job.label}.summary.json"
                if args.dry_run:
                    audit.write_dry_run(records, examples, job_jsonl)
                else:
                    if model is None or tokenizer is None or examples is None:
                        raise RuntimeError("Model/tokenizer not loaded for scoring run")
                    audit.audit_model(
                        model,
                        tokenizer,
                        examples,
                        args.batch_size,
                        job_jsonl,
                        content_skip_tokens=args.content_skip_tokens,
                    )

                audit.write_csv_from_jsonl(job_jsonl, job_csv)
                audit.write_summary_from_jsonl(job_jsonl, job_summary)
                append_file(job_jsonl, combined)
                manifest.append(
                    {
                        "label": job.label,
                        "dataset": job.dataset,
                        "kind": job.kind,
                        "path": str(job.path),
                        "records": len(all_records),
                        "jsonl": str(job_jsonl),
                        "csv": str(job_csv),
                        "summary_json": str(job_summary),
                    }
                )
    finally:
        if adapter_tmp is not None:
            adapter_tmp.cleanup()

    audit.write_csv_from_jsonl(combined_jsonl, output_dir / "trace_logprob_audit.csv")
    audit.write_summary_from_jsonl(combined_jsonl, output_dir / "trace_logprob_audit_summary.json")
    aggregate_worst_tokens(
        combined_jsonl,
        output_dir / "worst_token_aggregate.csv",
        output_dir / "worst_token_aggregate.json",
    )
    write_json(output_dir / "manifest.json", manifest)
    print(f"Wrote audit outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
