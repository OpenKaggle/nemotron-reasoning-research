#!/usr/bin/env python3
"""Index local public Kaggle notebook exports.

Scans notebook/kernel directories for Python, ipynb, and JSON files, then
emits a compact JSON and CSV index with metadata, Kaggle sources, URLs, and
competition-relevant keyword hits. Parsing errors are recorded per entry and do
not stop the scan.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (
    "notebooks",
    "public_radar/kernels",
    "tmp/public_radar/kernels",
)
EXTENSIONS = {".py", ".ipynb", ".json"}

DEFAULT_OUTPUT_JSON = Path("reports/public_notebook_index.json")
DEFAULT_OUTPUT_CSV = Path("reports/public_notebook_index.csv")

KEYWORD_PATTERNS = (
    ("DPO", re.compile(r"\bdpo\b", re.IGNORECASE)),
    ("GRPO", re.compile(r"\bgrpo\b", re.IGNORECASE)),
    ("SVD", re.compile(r"\bsvd\b", re.IGNORECASE)),
    ("merge", re.compile(r"\b(?:merge|merged|merges|merging|merger)\b", re.IGNORECASE)),
    ("router", re.compile(r"\b(?:router|routers|routing)\b", re.IGNORECASE)),
    ("logprob", re.compile(r"\blog[\s_-]?prob(?:s|ability|abilities)?\b", re.IGNORECASE)),
    ("min_logprob", re.compile(r"\bmin[\s_-]?log[\s_-]?prob(?:s)?\b", re.IGNORECASE)),
    ("LoRA", re.compile(r"\b(?:lora|lo-ra)\b", re.IGNORECASE)),
    ("TIES", re.compile(r"\bties\b", re.IGNORECASE)),
    ("DARE", re.compile(r"\bdare\b", re.IGNORECASE)),
    ("verifier", re.compile(r"\b(?:verifier|verifiers|verification)\b", re.IGNORECASE)),
    ("adapter", re.compile(r"\badapter(?:s)?\b", re.IGNORECASE)),
    ("ensemble", re.compile(r"\bensemble(?:s|d)?\b", re.IGNORECASE)),
    ("reward", re.compile(r"\breward(?:s)?\b", re.IGNORECASE)),
    ("SFT", re.compile(r"\bsft\b", re.IGNORECASE)),
    ("CoT", re.compile(r"\bcot\b", re.IGNORECASE)),
)

URL_RE = re.compile(r"https?://[^\s\]\)\}\"'<>\\\\]+", re.IGNORECASE)
KAGGLE_INPUT_RE = re.compile(
    r"(?:/kaggle/input|\.\./input)/([A-Za-z0-9_.-]+)", re.IGNORECASE
)
KAGGLE_DATASET_CMD_RE = re.compile(
    r"\bkaggle\s+datasets\s+download(?:\s+(?:-d|--dataset))?\s+([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",
    re.IGNORECASE,
)
KAGGLE_KERNEL_CMD_RE = re.compile(
    r"\bkaggle\s+kernels\s+(?:pull|output|status)(?:\s+(?:-k|--kernel))?\s+([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",
    re.IGNORECASE,
)
OWNER_SLUG_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")

SOURCE_LIST_KEYS = {
    "dataset_sources": "dataset_sources",
    "model_sources": "model_sources",
    "kernel_sources": "kernel_sources",
    "competition_sources": "competition_sources",
}

CSV_FIELDS = (
    "source_root",
    "notebook_dir",
    "notebook_id",
    "title",
    "code_file",
    "language",
    "kernel_type",
    "enable_gpu",
    "enable_tpu",
    "enable_internet",
    "machine_shape",
    "file_count",
    "ipynb_count",
    "py_count",
    "json_count",
    "cell_count",
    "dataset_sources",
    "model_sources",
    "kernel_sources",
    "competition_sources",
    "kaggle_dataset_refs",
    "kaggle_model_refs",
    "kaggle_kernel_refs",
    "kaggle_input_refs",
    "github_urls",
    "http_urls",
    "matched_keywords",
    "keyword_counts",
    "parse_error_count",
    "parse_errors",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index local public notebook exports into JSON and CSV reports."
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT_JSON),
        help=f"JSON output path (default: {DEFAULT_OUTPUT_JSON})",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_CSV),
        help=f"CSV output path (default: {DEFAULT_OUTPUT_CSV})",
    )
    return parser.parse_args()


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def stable_unique(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                continue
            value = normalized
        key = json.dumps(value, sort_keys=True, ensure_ascii=True) if isinstance(value, (dict, list)) else str(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def append_many(entry: dict[str, Any], key: str, values: Any) -> None:
    if values is None:
        return
    if isinstance(values, (str, int, float, bool)):
        values = [values]
    if not isinstance(values, list):
        return
    for value in values:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
        entry[key].append(value)


def scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def clean_url(url: str) -> str:
    for separator in ("\\n", "\\r", "\\t"):
        if separator in url:
            url = url.split(separator, 1)[0]
    return url.rstrip(".,;:!?)]}'\"")


def parse_kaggle_url(url: str) -> tuple[str | None, str | None]:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host != "kaggle.com":
        return None, None

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 3 and parts[0] in {"datasets", "d"}:
        return "dataset", f"{parts[1]}/{parts[2]}"
    if len(parts) >= 3 and parts[0] in {"code", "kernels"}:
        return "kernel", f"{parts[1]}/{parts[2]}"
    if len(parts) >= 3 and parts[0] == "models":
        return "model", "/".join(parts[1:])
    return None, None


def extract_text_signals(text: str) -> dict[str, Any]:
    urls = [clean_url(match.group(0)) for match in URL_RE.finditer(text)]
    github_urls = [
        url
        for url in urls
        if "github.com" in urlparse(url).netloc.lower()
        or "raw.githubusercontent.com" in urlparse(url).netloc.lower()
    ]

    kaggle_dataset_refs: list[str] = []
    kaggle_model_refs: list[str] = []
    kaggle_kernel_refs: list[str] = []
    for url in urls:
        ref_type, ref = parse_kaggle_url(url)
        if not ref:
            continue
        if ref_type == "dataset":
            kaggle_dataset_refs.append(ref)
        elif ref_type == "model":
            kaggle_model_refs.append(ref)
        elif ref_type == "kernel":
            kaggle_kernel_refs.append(ref)

    kaggle_dataset_refs.extend(match.group(1) for match in KAGGLE_DATASET_CMD_RE.finditer(text))
    kaggle_kernel_refs.extend(match.group(1) for match in KAGGLE_KERNEL_CMD_RE.finditer(text))
    kaggle_input_refs = [match.group(1) for match in KAGGLE_INPUT_RE.finditer(text)]

    keyword_counts = Counter()
    for keyword, pattern in KEYWORD_PATTERNS:
        count = len(pattern.findall(text))
        if count:
            keyword_counts[keyword] += count

    return {
        "http_urls": urls,
        "github_urls": github_urls,
        "kaggle_dataset_refs": kaggle_dataset_refs,
        "kaggle_model_refs": kaggle_model_refs,
        "kaggle_kernel_refs": kaggle_kernel_refs,
        "kaggle_input_refs": kaggle_input_refs,
        "keyword_counts": keyword_counts,
    }


def infer_owner_slug(dir_name: str, notebook_id: str | None = None) -> tuple[str | None, str | None]:
    if notebook_id and "/" in notebook_id:
        owner, slug = notebook_id.split("/", 1)
        return owner or None, slug or None
    if "__" in dir_name:
        owner, slug = dir_name.split("__", 1)
        return owner or None, slug or None
    return None, None


def new_entry(source_root: str, notebook_dir: Path) -> dict[str, Any]:
    owner, slug = infer_owner_slug(notebook_dir.name)
    notebook_id = f"{owner}/{slug}" if owner and slug else None
    return {
        "source_root": source_root,
        "notebook_dir": relpath(notebook_dir),
        "dir_name": notebook_dir.name,
        "owner": owner,
        "slug": slug,
        "notebook_id": notebook_id,
        "id_no": None,
        "title": None,
        "code_file": None,
        "language": None,
        "kernel_type": None,
        "enable_gpu": None,
        "enable_tpu": None,
        "enable_internet": None,
        "machine_shape": None,
        "docker_image": None,
        "files": [],
        "file_counts": {"py": 0, "ipynb": 0, "json": 0},
        "total_bytes": 0,
        "cell_count": 0,
        "cell_counts": {},
        "notebook_metadata": {
            "kernel_metadata": [],
            "ipynb": [],
        },
        "dataset_sources": [],
        "model_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "metadata_keywords": [],
        "kaggle_data_sources_raw": [],
        "kaggle_dataset_refs": [],
        "kaggle_model_refs": [],
        "kaggle_kernel_refs": [],
        "kaggle_input_refs": [],
        "http_urls": [],
        "github_urls": [],
        "matched_keywords": [],
        "keyword_counts": Counter(),
        "parse_errors": [],
    }


def iter_candidate_files() -> list[tuple[str, Path, Path]]:
    candidates: list[tuple[str, Path, Path]] = []
    for root_text in SCAN_ROOTS:
        root = PROJECT_ROOT / root_text
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            relative = path.relative_to(root)
            notebook_dir = root / relative.parts[0] if len(relative.parts) > 1 else path.parent
            candidates.append((root_text, notebook_dir, path))
    return sorted(candidates, key=lambda item: (item[0], relpath(item[1]), relpath(item[2])))


def read_text(path: Path, entry: dict[str, Any]) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        entry["parse_errors"].append({"file": relpath(path), "error": f"read failed: {exc}"})
        return None


def load_json(path: Path, text: str, entry: dict[str, Any]) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        entry["parse_errors"].append(
            {
                "file": relpath(path),
                "error": f"json parse failed at line {exc.lineno}, column {exc.colno}: {exc.msg}",
            }
        )
        return None


def cell_source_to_text(source: Any) -> str:
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    if isinstance(source, str):
        return source
    return ""


def add_raw_kaggle_data_source(entry: dict[str, Any], file_path: Path, source: Any) -> None:
    if not isinstance(source, dict):
        return
    compact = {
        "file": relpath(file_path),
        "sourceType": source.get("sourceType"),
        "sourceId": source.get("sourceId"),
        "databundleVersionId": source.get("databundleVersionId"),
        "modelInstanceId": source.get("modelInstanceId"),
        "modelId": source.get("modelId"),
        "isSourceIdPinned": source.get("isSourceIdPinned"),
    }
    compact = {key: value for key, value in compact.items() if value is not None}
    entry["kaggle_data_sources_raw"].append(compact)


def harvest_json_sources(entry: dict[str, Any], path: Path, obj: Any) -> None:
    stack = [obj]
    visited = 0
    while stack:
        current = stack.pop()
        visited += 1
        if visited > 20000:
            entry["parse_errors"].append(
                {"file": relpath(path), "error": "json source harvest stopped after 20000 nodes"}
            )
            return

        if isinstance(current, dict):
            for key, value in current.items():
                normalized_key = str(key)
                if normalized_key in SOURCE_LIST_KEYS:
                    append_many(entry, SOURCE_LIST_KEYS[normalized_key], value)
                elif normalized_key == "keywords":
                    append_many(entry, "metadata_keywords", value)
                elif normalized_key == "dataSources" and isinstance(value, list):
                    for source in value:
                        add_raw_kaggle_data_source(entry, path, source)

                if isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(current, list):
            stack.extend(value for value in current if isinstance(value, (dict, list)))


def apply_kernel_metadata(entry: dict[str, Any], path: Path, data: Any) -> None:
    if not isinstance(data, dict):
        return
    entry["notebook_metadata"]["kernel_metadata"].append({"file": relpath(path), "raw": data})

    scalar_fields = (
        "id",
        "id_no",
        "title",
        "code_file",
        "language",
        "kernel_type",
        "enable_gpu",
        "enable_tpu",
        "enable_internet",
        "machine_shape",
        "docker_image",
    )
    for field in scalar_fields:
        value = data.get(field)
        if value is None:
            continue
        target_field = "notebook_id" if field == "id" else field
        if entry.get(target_field) in (None, ""):
            entry[target_field] = value

    owner, slug = infer_owner_slug(entry["dir_name"], entry.get("notebook_id"))
    entry["owner"] = entry["owner"] or owner
    entry["slug"] = entry["slug"] or slug

    append_many(entry, "dataset_sources", data.get("dataset_sources"))
    append_many(entry, "model_sources", data.get("model_sources"))
    append_many(entry, "kernel_sources", data.get("kernel_sources"))
    append_many(entry, "competition_sources", data.get("competition_sources"))
    append_many(entry, "metadata_keywords", data.get("keywords"))


def apply_ipynb_metadata(entry: dict[str, Any], path: Path, data: Any) -> None:
    if not isinstance(data, dict):
        return

    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    kaggle = metadata.get("kaggle") if isinstance(metadata.get("kaggle"), dict) else {}
    kernelspec = metadata.get("kernelspec") if isinstance(metadata.get("kernelspec"), dict) else {}
    language_info = metadata.get("language_info") if isinstance(metadata.get("language_info"), dict) else {}
    cells = data.get("cells") if isinstance(data.get("cells"), list) else []

    cell_counts = Counter()
    source_chars = 0
    for cell in cells:
        if not isinstance(cell, dict):
            cell_counts["unknown"] += 1
            continue
        cell_counts[str(cell.get("cell_type") or "unknown")] += 1
        source_chars += len(cell_source_to_text(cell.get("source")))

    entry["cell_count"] += len(cells)
    for key, value in cell_counts.items():
        entry["cell_counts"][key] = entry["cell_counts"].get(key, 0) + value

    record = {
        "file": relpath(path),
        "nbformat": data.get("nbformat"),
        "nbformat_minor": data.get("nbformat_minor"),
        "metadata_keys": sorted(metadata.keys()),
        "kernelspec": kernelspec,
        "language_info": language_info,
        "kaggle": kaggle,
        "cell_count": len(cells),
        "cell_counts": dict(sorted(cell_counts.items())),
        "source_chars": source_chars,
    }
    entry["notebook_metadata"]["ipynb"].append(record)

    if entry.get("language") in (None, ""):
        entry["language"] = kaggle.get("language") or kernelspec.get("language") or language_info.get("name")
    if entry.get("kernel_type") in (None, ""):
        entry["kernel_type"] = kaggle.get("sourceType")
    if entry.get("enable_gpu") is None and "isGpuEnabled" in kaggle:
        entry["enable_gpu"] = kaggle.get("isGpuEnabled")
    if entry.get("enable_internet") is None and "isInternetEnabled" in kaggle:
        entry["enable_internet"] = kaggle.get("isInternetEnabled")
    if entry.get("machine_shape") in (None, ""):
        entry["machine_shape"] = kaggle.get("accelerator")

    for source in kaggle.get("dataSources", []) if isinstance(kaggle.get("dataSources"), list) else []:
        add_raw_kaggle_data_source(entry, path, source)


def add_signals(entry: dict[str, Any], signals: dict[str, Any]) -> None:
    for key in (
        "http_urls",
        "github_urls",
        "kaggle_dataset_refs",
        "kaggle_model_refs",
        "kaggle_kernel_refs",
        "kaggle_input_refs",
    ):
        append_many(entry, key, signals.get(key))
    entry["keyword_counts"].update(signals["keyword_counts"])


def process_file(entry: dict[str, Any], path: Path) -> None:
    suffix = path.suffix.lower().lstrip(".")
    entry["files"].append(relpath(path))
    entry["file_counts"][suffix] = entry["file_counts"].get(suffix, 0) + 1
    try:
        entry["total_bytes"] += path.stat().st_size
    except OSError:
        pass

    text = read_text(path, entry)
    if text is None:
        return

    add_signals(entry, extract_text_signals(text))

    if path.suffix.lower() not in {".ipynb", ".json"}:
        return

    data = load_json(path, text, entry)
    if data is None:
        return

    harvest_json_sources(entry, path, data)
    if path.suffix.lower() == ".ipynb":
        apply_ipynb_metadata(entry, path, data)
    if path.name == "kernel-metadata.json":
        apply_kernel_metadata(entry, path, data)


def finalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    for source_key, ref_key in (
        ("dataset_sources", "kaggle_dataset_refs"),
        ("model_sources", "kaggle_model_refs"),
        ("kernel_sources", "kaggle_kernel_refs"),
    ):
        append_many(entry, ref_key, entry[source_key])

    for key in (
        "files",
        "dataset_sources",
        "model_sources",
        "kernel_sources",
        "competition_sources",
        "metadata_keywords",
        "kaggle_data_sources_raw",
        "kaggle_dataset_refs",
        "kaggle_model_refs",
        "kaggle_kernel_refs",
        "kaggle_input_refs",
        "http_urls",
        "github_urls",
        "parse_errors",
    ):
        entry[key] = stable_unique(entry[key])

    if not entry.get("notebook_id") and entry.get("owner") and entry.get("slug"):
        entry["notebook_id"] = f"{entry['owner']}/{entry['slug']}"
    if not entry.get("title") and entry.get("slug"):
        entry["title"] = str(entry["slug"]).replace("-", " ")

    keyword_counts = {
        keyword: entry["keyword_counts"].get(keyword, 0)
        for keyword, _ in KEYWORD_PATTERNS
        if entry["keyword_counts"].get(keyword, 0)
    }
    entry["keyword_counts"] = keyword_counts
    entry["matched_keywords"] = list(keyword_counts.keys())
    entry["parse_error_count"] = len(entry["parse_errors"])
    entry["file_count"] = len(entry["files"])
    entry["ipynb_count"] = entry["file_counts"].get("ipynb", 0)
    entry["py_count"] = entry["file_counts"].get("py", 0)
    entry["json_count"] = entry["file_counts"].get("json", 0)
    return entry


def build_index() -> dict[str, Any]:
    entries: dict[tuple[str, str], dict[str, Any]] = {}
    missing_roots = [
        root_text for root_text in SCAN_ROOTS if not (PROJECT_ROOT / root_text).exists()
    ]

    for source_root, notebook_dir, path in iter_candidate_files():
        key = (source_root, relpath(notebook_dir))
        if key not in entries:
            entries[key] = new_entry(source_root, notebook_dir)
        process_file(entries[key], path)

    finalized_entries = [finalize_entry(entry) for entry in entries.values()]
    finalized_entries.sort(key=lambda item: (item["source_root"], item["notebook_dir"]))

    by_source_root = Counter(entry["source_root"] for entry in finalized_entries)
    by_extension = Counter()
    keyword_entry_counts = Counter()
    dataset_source_counts = Counter()
    model_source_counts = Counter()
    for entry in finalized_entries:
        by_extension.update(entry["file_counts"])
        keyword_entry_counts.update(entry["matched_keywords"])
        dataset_source_counts.update(entry["dataset_sources"])
        model_source_counts.update(entry["model_sources"])

    summary = {
        "entry_count": len(finalized_entries),
        "file_count": sum(entry["file_count"] for entry in finalized_entries),
        "by_source_root": dict(sorted(by_source_root.items())),
        "by_extension": dict(sorted(by_extension.items())),
        "entries_with_errors": sum(1 for entry in finalized_entries if entry["parse_error_count"]),
        "keyword_entry_counts": {
            keyword: keyword_entry_counts[keyword]
            for keyword, _ in KEYWORD_PATTERNS
            if keyword_entry_counts[keyword]
        },
        "top_dataset_sources": dataset_source_counts.most_common(20),
        "top_model_sources": model_source_counts.most_common(20),
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": PROJECT_ROOT.as_posix(),
        "scanned_roots": list(SCAN_ROOTS),
        "missing_roots": missing_roots,
        "extensions": sorted(EXTENSIONS),
        "keywords": [keyword for keyword, _ in KEYWORD_PATTERNS],
        "summary": summary,
        "entries": finalized_entries,
    }


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return " | ".join(csv_value(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def write_json(index: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(entries: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for entry in entries:
            writer.writerow({field: csv_value(entry.get(field)) for field in CSV_FIELDS})


def main() -> int:
    args = parse_args()
    output_json = resolve_project_path(args.output_json)
    output_csv = resolve_project_path(args.output_csv)

    index = build_index()
    write_json(index, output_json)
    write_csv(index["entries"], output_csv)

    summary = index["summary"]
    print(
        "Indexed "
        f"{summary['entry_count']} notebook dirs / {summary['file_count']} files; "
        f"errors={summary['entries_with_errors']}; "
        f"json={relpath(output_json)}; csv={relpath(output_csv)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
