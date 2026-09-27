from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import tempfile
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_SUFFIX = "\nPut your final answer inside \\boxed{}."
BOXED_RE = re.compile(r"\\boxed\{(?P<answer>.*)\}\s*$", re.DOTALL)


@dataclass
class TraceRecord:
    id: str
    source: str
    task_type: str
    prompt: str
    answer: str
    completion: str
    method: str = ""
    quality_flags: str = ""


@dataclass
class TokenizedRecord:
    record: TraceRecord
    input_ids: list[int]
    loss_mask: list[int]


def classify_prompt(prompt: str) -> str:
    text = prompt.lower()
    if "bit manipulation" in text or "8-bit" in text:
        return "bit_manipulation"
    if "secret encryption rules are used on text" in text or "decrypt the following text" in text:
        return "text_cipher"
    if "roman" in text:
        return "numeral_system"
    if "distance" in text and "for t =" in text:
        return "gravity_physics"
    if "convert the following measurement" in text:
        return "unit_conversion"
    if "secret set of transformation rules is applied to equations" in text:
        return "symbolic_equation"
    return "unknown"


def read_train_index(path: Path | None) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def has_boxed(text: str) -> bool:
    return "\\boxed{" in text


def ensure_boxed(completion: str, answer: str) -> tuple[str, list[str]]:
    flags: list[str] = []
    if has_boxed(completion):
        if completion.count("\\boxed{") > 1:
            flags.append("multiple_boxed")
        return completion, flags
    flags.append("boxed_appended")
    return completion.rstrip() + f"\n\n\\boxed{{{answer}}}", flags


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def load_oracle_jsonl(path: Path, source_name: str) -> Iterable[TraceRecord]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            completion, flags = ensure_boxed(str(row["completion"]), str(row["answer"]))
            existing_flags = [
                flag
                for flag in normalize_text(row.get("quality_flags")).split(";")
                if flag
            ]
            for flag in existing_flags:
                if flag not in flags:
                    flags.append(flag)
            yield TraceRecord(
                id=str(row["id"]),
                source=normalize_text(row.get("source")) or source_name,
                task_type=str(row.get("task_type") or classify_prompt(row["prompt"])),
                prompt=str(row["prompt"]),
                answer=str(row["answer"]),
                completion=completion,
                method=str(row.get("method") or ""),
                quality_flags=";".join(flags),
            )


def load_csv_trace(path: Path, source_name: str, kind: str) -> Iterable[TraceRecord]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if kind == "llkh0a_formatted":
                prompt = normalize_text(row.get("user_content"))
                completion = normalize_text(row.get("assistant_content"))
                answer_match = BOXED_RE.search(completion)
                answer = answer_match.group("answer") if answer_match else ""
                yield TraceRecord(
                    id=f"row_{idx:06d}",
                    source=source_name,
                    task_type=classify_prompt(prompt),
                    prompt=prompt,
                    answer=answer,
                    completion=completion,
                    method="llkh0a_formatted",
                    quality_flags="" if answer else "answer_missing",
                )
                continue

            prompt = normalize_text(row.get("prompt"))
            answer = normalize_text(row.get("answer"))
            completion = (
                normalize_text(row.get("completion"))
                or normalize_text(row.get("generated_cot"))
                or normalize_text(row.get("cot_trace"))
                or normalize_text(row.get("assistant_content"))
            )
            completion, flags = ensure_boxed(completion, answer)
            task_type = (
                normalize_text(row.get("task_type"))
                or normalize_text(row.get("task"))
                or normalize_text(row.get("type"))
                or normalize_text(row.get("category"))
                or classify_prompt(prompt)
            )
            method = normalize_text(row.get("method")) or kind
            yield TraceRecord(
                id=normalize_text(row.get("id")) or f"row_{idx:06d}",
                source=source_name,
                task_type=task_type,
                prompt=prompt,
                answer=answer,
                completion=completion,
                method=method,
                quality_flags=";".join(flags),
            )


def load_txt_dir(path: Path, source_name: str, train_index: dict[str, dict[str, str]]) -> Iterable[TraceRecord]:
    for file in sorted(path.glob("*.txt")):
        if file.name.lower() == "readme.md":
            continue
        problem_id = file.stem
        train_row = train_index.get(problem_id, {})
        prompt = normalize_text(train_row.get("prompt"))
        answer = normalize_text(train_row.get("answer"))
        completion = file.read_text(encoding="utf-8")
        completion, flags = ensure_boxed(completion, answer)
        if not prompt:
            flags.append("prompt_missing")
        if not answer:
            flags.append("answer_missing")
        yield TraceRecord(
            id=problem_id,
            source=source_name,
            task_type=classify_prompt(prompt) if prompt else "unknown",
            prompt=prompt,
            answer=answer,
            completion=completion,
            method="txt_trace",
            quality_flags=";".join(flags),
        )


def read_label_index(path: Path | None) -> dict[int, tuple[str, str]]:
    if not path or not path.exists():
        return {}
    labels: dict[int, tuple[str, str]] = {}
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            labels[idx] = (normalize_text(row.get("id")), normalize_text(row.get("answer")))
    return labels


def load_nemo_jsonl(path: Path, source_name: str, label_index: dict[int, tuple[str, str]]) -> Iterable[TokenizedRecord]:
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            input_ids = row.get("input_ids")
            labels = row.get("labels")
            if not isinstance(input_ids, list) or not isinstance(labels, list):
                continue
            if len(input_ids) != len(labels):
                continue
            problem_id, answer = label_index.get(idx, (f"row_{idx:06d}", ""))
            loss_mask = [0 if int(label) == -100 else 1 for label in labels]
            flags = ["pretokenized_nemo"]
            if not answer:
                flags.append("empty_answer")
            yield TokenizedRecord(
                record=TraceRecord(
                    id=problem_id or f"row_{idx:06d}",
                    source=source_name,
                    task_type="huikang_14class",
                    prompt="",
                    answer=answer,
                    completion="",
                    method="nemo_pre_tokenized",
                    quality_flags=";".join(flags),
                ),
                input_ids=[int(token_id) for token_id in input_ids],
                loss_mask=loss_mask,
            )


def infer_kind(path: Path) -> str:
    name = path.name.lower()
    parent = path.parent.name.lower()
    if path.is_dir():
        return "txt_dir"
    if name.endswith(".jsonl") and name.startswith("nemo_"):
        return "nemo_jsonl"
    if name.endswith(".jsonl"):
        return "oracle_jsonl"
    if "formatted_train_dataset" in name:
        return "llkh0a_formatted"
    if "leevvin" in str(path).lower() or "bitmanip" in str(path).lower() or "tong" in name or "donald" in name:
        return "generic_csv"
    if name.endswith(".csv"):
        return "generic_csv"
    if parent.startswith("sybyrr"):
        return "txt_dir"
    raise ValueError(f"Could not infer source kind for {path}")


def parse_source(spec: str) -> tuple[str, Path, str]:
    if ":" in spec and not spec.startswith("/"):
        kind, raw_path = spec.split(":", 1)
    else:
        raw_path = spec
        kind = "auto"
    path = Path(raw_path).expanduser().resolve()
    if kind == "auto":
        kind = infer_kind(path)
    source_name = f"{kind}:{path.name}"
    return kind, path, source_name


def load_records(sources: list[str], train_csv: Path | None, label_jsonl: Path | None = None) -> tuple[list[TraceRecord], list[TokenizedRecord]]:
    train_index = read_train_index(train_csv)
    label_index = read_label_index(label_jsonl)
    records: list[TraceRecord] = []
    tokenized_records: list[TokenizedRecord] = []
    for spec in sources:
        kind, path, source_name = parse_source(spec)
        if kind == "oracle_jsonl":
            records.extend(load_oracle_jsonl(path, source_name))
        elif kind == "nemo_jsonl":
            tokenized_records.extend(load_nemo_jsonl(path, source_name, label_index))
        elif kind in {"generic_csv", "llkh0a_formatted"}:
            records.extend(load_csv_trace(path, source_name, kind))
        elif kind == "txt_dir":
            records.extend(load_txt_dir(path, source_name, train_index))
        else:
            raise ValueError(f"Unsupported source kind {kind!r} for {path}")
    return records, tokenized_records


def sample_records(records: list[TraceRecord], limit: int | None, seed: int) -> list[TraceRecord]:
    if limit is None or limit <= 0 or len(records) <= limit:
        return records
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(records)), limit))
    return [records[i] for i in idx]


def sample_tokenized_records(records: list[TokenizedRecord], limit: int | None, seed: int) -> list[TokenizedRecord]:
    if limit is None or limit <= 0 or len(records) <= limit:
        return records
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(records)), limit))
    return [records[i] for i in idx]


def read_id_filter(paths: list[Path] | None) -> set[str] | None:
    if not paths:
        return None
    ids: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                value = line.strip()
                if value and not value.startswith("#"):
                    ids.add(value.split(",")[0])
    return ids


def filter_records(
    records: list[TraceRecord],
    task_types: list[str] | None,
    sources: list[str] | None,
    methods: list[str] | None,
    ids: set[str] | None,
    require_prompt: bool,
    require_answer: bool,
    drop_flag: list[str] | None,
) -> list[TraceRecord]:
    task_set = set(task_types or [])
    source_set = set(sources or [])
    method_set = set(methods or [])
    drop_flags = set(drop_flag or [])
    kept: list[TraceRecord] = []
    for record in records:
        flags = set(flag for flag in record.quality_flags.split(";") if flag)
        if task_set and record.task_type not in task_set:
            continue
        if source_set and record.source not in source_set:
            continue
        if method_set and record.method not in method_set:
            continue
        if ids is not None and record.id not in ids:
            continue
        if require_prompt and not record.prompt:
            continue
        if require_answer and not record.answer:
            continue
        if drop_flags and flags & drop_flags:
            continue
        kept.append(record)
    return kept


def filter_tokenized_records(
    records: list[TokenizedRecord],
    task_types: list[str] | None,
    sources: list[str] | None,
    methods: list[str] | None,
    ids: set[str] | None,
    require_answer: bool,
    drop_flag: list[str] | None,
) -> list[TokenizedRecord]:
    kept_records = filter_records(
        [record.record for record in records],
        task_types=task_types,
        sources=sources,
        methods=methods,
        ids=ids,
        require_prompt=False,
        require_answer=require_answer,
        drop_flag=drop_flag,
    )
    keep_ids = {id(record) for record in kept_records}
    return [record for record in records if id(record.record) in keep_ids]


def resolve_adapter_dir(path: Path | None) -> tuple[Path | None, tempfile.TemporaryDirectory[str] | None]:
    if path is None:
        return None, None
    path = path.expanduser().resolve()
    if path.is_dir():
        return path, None
    if path.is_file() and path.suffix == ".zip":
        tmpdir = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(path) as zf:
            zf.extractall(tmpdir.name)
        root = Path(tmpdir.name)
        candidates = [p.parent for p in root.rglob("adapter_config.json")]
        if not candidates:
            tmpdir.cleanup()
            raise FileNotFoundError(f"No adapter_config.json inside {path}")
        return candidates[0], tmpdir
    raise FileNotFoundError(path)


def load_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def normalize_token_ids(value: object) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        value = value.get("input_ids", [])
    if hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise TypeError(f"Expected token id list, got {type(value).__name__}")
    return [int(token_id) for token_id in value]


def encode_text(tokenizer, text: str) -> list[int]:
    try:
        return normalize_token_ids(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return normalize_token_ids(tokenizer(text, add_special_tokens=False))


def render_messages(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if isinstance(rendered, str) and rendered:
            return rendered
    except Exception:
        pass
    text = ""
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    if add_generation_prompt:
        text += "<|im_start|>assistant\n"
    return text


def build_tokenized_example(tokenizer, record: TraceRecord, max_length: int, prompt_suffix: str) -> dict:
    user_msg = record.prompt + prompt_suffix
    assistant_msg = record.completion
    messages_user = [{"role": "user", "content": user_msg}]
    messages_full = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    prompt_text = render_messages(tokenizer, messages_user, add_generation_prompt=True)
    full_text = render_messages(tokenizer, messages_full, add_generation_prompt=False)
    prompt_ids = encode_text(tokenizer, prompt_text)
    full_ids = encode_text(tokenizer, full_text)

    prompt_len = len(prompt_ids)
    prefix_mismatch = full_ids[:prompt_len] != prompt_ids
    if prefix_mismatch:
        prompt_len = longest_common_prefix(full_ids, prompt_ids)
    loss_mask = [0] * prompt_len + [1] * max(0, len(full_ids) - prompt_len)
    truncated = len(full_ids) > max_length
    if truncated:
        full_ids = full_ids[:max_length]
        loss_mask = loss_mask[:max_length]

    completion_token_len = sum(loss_mask)
    return {
        "record": record,
        "input_ids": full_ids,
        "loss_mask": loss_mask,
        "prompt_token_len": min(prompt_len, len(full_ids)),
        "completion_token_len": completion_token_len,
        "truncated": truncated,
        "prefix_mismatch": prefix_mismatch,
    }


def build_pretokenized_example(tokenized: TokenizedRecord, max_length: int) -> dict:
    input_ids = tokenized.input_ids
    loss_mask = tokenized.loss_mask
    truncated = len(input_ids) > max_length
    if truncated:
        input_ids = input_ids[:max_length]
        loss_mask = loss_mask[:max_length]
    prompt_token_len = 0
    for value in loss_mask:
        if value:
            break
        prompt_token_len += 1
    return {
        "record": tokenized.record,
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "prompt_token_len": min(prompt_token_len, len(input_ids)),
        "completion_token_len": sum(loss_mask),
        "truncated": truncated,
        "prefix_mismatch": False,
        "pretokenized": True,
    }


def longest_common_prefix(left: list[int], right: list[int]) -> int:
    n = min(len(left), len(right))
    for i in range(n):
        if left[i] != right[i]:
            return i
    return n


def load_model(model_path: str, adapter: Path | None):
    import torch
    from transformers import AutoModelForCausalLM

    kwargs = {
        "trust_remote_code": True,
        "device_map": {"": 0} if torch.cuda.is_available() else None,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    adapter_tmp = None
    if adapter is not None:
        from peft import PeftModel

        adapter_dir, adapter_tmp = resolve_adapter_dir(adapter)
        assert adapter_dir is not None
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, adapter_tmp


def audit_model(
    model,
    tokenizer,
    examples: list[dict],
    batch_size: int,
    output_path: Path,
    content_skip_tokens: int = 16,
) -> None:
    import torch
    import torch.nn.functional as F

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    device = next(model.parameters()).device
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            max_len = max(len(item["input_ids"]) for item in batch)
            input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
            attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
            loss_mask = torch.zeros((len(batch), max_len), dtype=torch.bool, device=device)
            for i, item in enumerate(batch):
                ids = torch.tensor(item["input_ids"], dtype=torch.long, device=device)
                mask = torch.tensor(item["loss_mask"], dtype=torch.bool, device=device)
                input_ids[i, : len(ids)] = ids
                attention_mask[i, : len(ids)] = 1
                loss_mask[i, : len(mask)] = mask

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                logits = outputs.logits[:, :-1, :].float()
                labels = input_ids[:, 1:]
                target_mask = loss_mask[:, 1:] & attention_mask[:, 1:].bool()
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    reduction="none",
                ).reshape(labels.shape)
                logprobs = -ce

            for i, item in enumerate(batch):
                row = summarize_logprobs(
                    tokenizer,
                    item,
                    labels[i],
                    logprobs[i],
                    target_mask[i],
                    content_skip_tokens=content_skip_tokens,
                )
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

            del input_ids, attention_mask, loss_mask, outputs, logits, labels, target_mask, ce, logprobs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def summarize_logprobs(tokenizer, item: dict, labels, logprobs, target_mask, content_skip_tokens: int = 16) -> dict:
    import torch

    selected = torch.nonzero(target_mask, as_tuple=False).flatten()
    base = base_output_row(item)
    base.update(summarize_token_slice(tokenizer, labels, logprobs, selected, prefix=""))
    skip = max(0, int(content_skip_tokens))
    content_selected = selected[skip:] if skip else selected
    base["content_skip_tokens"] = skip
    base.update(summarize_token_slice(tokenizer, labels, logprobs, content_selected, prefix="content"))
    return base


def summarize_token_slice(tokenizer, labels, logprobs, selected, prefix: str) -> dict:
    import torch

    field = f"{prefix}_" if prefix else ""
    if selected.numel() == 0:
        return {
            f"{field}scored_tokens": 0,
            f"{field}mean_logprob": None,
            f"{field}min_logprob": None,
            f"{field}worst_token_index": None,
            f"{field}worst_token_id": None,
            f"{field}worst_token_text": None,
            f"{field}worst_context": None,
        }
    vals = logprobs[selected]
    worst_rel = int(torch.argmin(vals).item())
    worst_pos = int(selected[worst_rel].item())
    worst_token_id = int(labels[worst_pos].item())
    context_start = max(0, worst_pos - 8)
    context_end = min(labels.shape[0], worst_pos + 9)
    context_ids = [int(x) for x in labels[context_start:context_end].detach().cpu().tolist()]
    return {
        f"{field}scored_tokens": int(selected.numel()),
        f"{field}mean_logprob": float(vals.mean().item()),
        f"{field}min_logprob": float(vals.min().item()),
        f"{field}worst_token_index": worst_pos + 1,
        f"{field}worst_token_id": worst_token_id,
        f"{field}worst_token_text": tokenizer.decode([worst_token_id], skip_special_tokens=False),
        f"{field}worst_context": tokenizer.decode(context_ids, skip_special_tokens=False),
    }


def base_output_row(item: dict) -> dict:
    record: TraceRecord = item["record"]
    return {
        "id": record.id,
        "source": record.source,
        "task_type": record.task_type,
        "method": record.method,
        "answer": record.answer,
        "quality_flags": record.quality_flags,
        "prompt_chars": len(record.prompt),
        "completion_chars": len(record.completion),
        "prompt_token_len": item.get("prompt_token_len"),
        "completion_token_len": item.get("completion_token_len"),
        "total_token_len": len(item.get("input_ids") or []),
        "truncated": bool(item.get("truncated")),
        "prefix_mismatch": bool(item.get("prefix_mismatch")),
        "pretokenized": bool(item.get("pretokenized")),
    }


def write_dry_run(records: list[TraceRecord], examples: list[dict] | None, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        example_record_ids = {id(item.get("record")) for item in examples or []}
        for record in records:
            if id(record) in example_record_ids:
                continue
            row = asdict(record)
            row.update(
                {
                    "prompt_chars": len(record.prompt),
                    "completion_chars": len(record.completion),
                }
            )
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
        if examples is not None:
            for item in examples:
                out.write(json.dumps(base_output_row(item), ensure_ascii=False) + "\n")


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


CSV_FIELDS = [
    "id",
    "source",
    "task_type",
    "method",
    "answer",
    "quality_flags",
    "prompt_chars",
    "completion_chars",
    "prompt_token_len",
    "completion_token_len",
    "total_token_len",
    "truncated",
    "prefix_mismatch",
    "pretokenized",
    "scored_tokens",
    "mean_logprob",
    "min_logprob",
    "worst_token_index",
    "worst_token_id",
    "worst_token_text",
    "worst_context",
    "content_skip_tokens",
    "content_scored_tokens",
    "content_mean_logprob",
    "content_min_logprob",
    "content_worst_token_index",
    "content_worst_token_id",
    "content_worst_token_text",
    "content_worst_context",
]


def write_csv_from_jsonl(jsonl_path: Path, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in iter_jsonl(jsonl_path):
            writer.writerow(row)


def write_summary_from_jsonl(jsonl_path: Path, summary_path: Path) -> None:
    rows = list(iter_jsonl(jsonl_path))
    summary: dict[str, object] = {
        "rows": len(rows),
        "by_source": dict(Counter(str(r.get("source")) for r in rows).most_common()),
        "by_task_type": dict(Counter(str(r.get("task_type")) for r in rows).most_common()),
        "quality_flags": dict(
            Counter(
                flag
                for r in rows
                for flag in str(r.get("quality_flags") or "").split(";")
                if flag
            ).most_common(50)
        ),
        "truncated": sum(1 for r in rows if r.get("truncated")),
        "prefix_mismatch": sum(1 for r in rows if r.get("prefix_mismatch")),
    }
    for score_field in ["min_logprob", "content_min_logprob"]:
        scored = [r for r in rows if isinstance(r.get(score_field), (int, float))]
        if not scored:
            continue
        mean_field = "content_mean_logprob" if score_field.startswith("content_") else "mean_logprob"
        token_field = "content_worst_token_text" if score_field.startswith("content_") else "worst_token_text"
        context_field = "content_worst_context" if score_field.startswith("content_") else "worst_context"
        by_task: dict[str, list[float]] = {}
        for r in scored:
            by_task.setdefault(str(r.get("task_type")), []).append(float(r[score_field]))
        summary[score_field] = {
            "mean": sum(float(r[score_field]) for r in scored) / len(scored),
            "min": min(float(r[score_field]) for r in scored),
            "by_task_mean": {
                task: sum(vals) / len(vals) for task, vals in sorted(by_task.items())
            },
        }
        worst = sorted(scored, key=lambda r: float(r[score_field]))[:20]
        summary[f"{score_field}_worst_rows"] = [
            {
                "id": r.get("id"),
                "source": r.get("source"),
                "task_type": r.get("task_type"),
                "method": r.get("method"),
                score_field: r.get(score_field),
                mean_field: r.get(mean_field),
                "worst_token_text": r.get(token_field),
                "worst_context": r.get(context_field),
            }
            for r in worst
        ]
    if "min_logprob_worst_rows" in summary:
        summary["worst_rows"] = summary["min_logprob_worst_rows"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def print_summary(records: list[TraceRecord], examples: list[dict] | None) -> None:
    print(f"records={len(records)}")
    print("by_source", dict(Counter(r.source for r in records).most_common()))
    print("by_task", dict(Counter(r.task_type for r in records).most_common()))
    print("quality_flags", dict(Counter(flag for r in records for flag in r.quality_flags.split(";") if flag).most_common(20)))
    if examples is not None:
        completion_lengths = [e["completion_token_len"] for e in examples]
        total_lengths = [len(e["input_ids"]) for e in examples]
        if completion_lengths:
            print(
                "tokens",
                {
                    "completion_min": min(completion_lengths),
                    "completion_median": sorted(completion_lengths)[len(completion_lengths) // 2],
                    "completion_max": max(completion_lengths),
                    "total_max": max(total_lengths),
                    "truncated": sum(1 for e in examples if e["truncated"]),
                    "prefix_mismatch": sum(1 for e in examples if e["prefix_mismatch"]),
                },
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit per-token logprobs for Nemotron trace corpora.")
    parser.add_argument("--source", action="append", required=True, help="Trace source as kind:/path or /path. Kinds: oracle_jsonl,generic_csv,llkh0a_formatted,txt_dir,nemo_jsonl.")
    parser.add_argument("--train-csv", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument("--label-jsonl", type=Path, default=None, help="Optional labels aligned with pre-tokenized nemo_jsonl rows, e.g. valid_labels.jsonl.")
    parser.add_argument("--model-path", default=None, help="HF/model path. Required unless --no-tokenizer is set.")
    parser.add_argument("--adapter", type=Path, default=None, help="Optional PEFT adapter dir or submission zip.")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "trace_logprob_audit.jsonl")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-type", action="append", default=None)
    parser.add_argument("--source-name", action="append", default=None, help="Keep only exact normalized source names from dry-run summary.")
    parser.add_argument("--method", action="append", default=None)
    parser.add_argument("--id-file", action="append", type=Path, default=None, help="One id per line, or id as first CSV column.")
    parser.add_argument("--require-prompt", action="store_true")
    parser.add_argument("--require-answer", action="store_true")
    parser.add_argument("--drop-flag", action="append", default=None, help="Drop rows containing a quality flag, e.g. multiple_boxed.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--content-skip-tokens", type=int, default=16)
    parser.add_argument("--prompt-suffix", default=DEFAULT_PROMPT_SUFFIX)
    parser.add_argument("--dry-run", action="store_true", help="Parse/tokenize and write metadata only.")
    parser.add_argument("--no-tokenizer", action="store_true", help="For dry-run only: skip tokenizer/model loading.")
    args = parser.parse_args()

    records, tokenized_records = load_records(args.source, args.train_csv, args.label_jsonl)
    id_filter = read_id_filter(args.id_file)
    records = filter_records(
        records,
        task_types=args.task_type,
        sources=args.source_name,
        methods=args.method,
        ids=id_filter,
        require_prompt=args.require_prompt,
        require_answer=args.require_answer,
        drop_flag=args.drop_flag,
    )
    tokenized_records = filter_tokenized_records(
        tokenized_records,
        task_types=args.task_type,
        sources=args.source_name,
        methods=args.method,
        ids=id_filter,
        require_answer=args.require_answer,
        drop_flag=args.drop_flag,
    )
    records = sample_records(records, args.limit, args.seed)
    tokenized_records = sample_tokenized_records(tokenized_records, args.limit, args.seed)

    if args.no_tokenizer:
        if not args.dry_run:
            raise ValueError("--no-tokenizer is only valid with --dry-run")
        examples = [build_pretokenized_example(record, args.max_length) for record in tokenized_records]
        write_dry_run(records, examples if examples else None, args.output)
        if args.csv:
            write_csv_from_jsonl(args.output, args.csv)
        if args.summary_json:
            write_summary_from_jsonl(args.output, args.summary_json)
        print_summary(records + [record.record for record in tokenized_records], examples if examples else None)
        print(f"wrote {args.output}")
        return

    if not args.model_path:
        raise ValueError("--model-path is required unless --no-tokenizer is set")
    tokenizer = load_tokenizer(args.model_path)
    examples = [
        build_tokenized_example(tokenizer, record, args.max_length, args.prompt_suffix)
        for record in records
    ]
    examples.extend(build_pretokenized_example(record, args.max_length) for record in tokenized_records)
    print_summary(records + [record.record for record in tokenized_records], examples)

    if args.dry_run:
        write_dry_run(records, examples, args.output)
        if args.csv:
            write_csv_from_jsonl(args.output, args.csv)
        if args.summary_json:
            write_summary_from_jsonl(args.output, args.summary_json)
        print(f"wrote {args.output}")
        return

    model, adapter_tmp = load_model(args.model_path, args.adapter)
    try:
        audit_model(
            model,
            tokenizer,
            examples,
            args.batch_size,
            args.output,
            content_skip_tokens=args.content_skip_tokens,
        )
    finally:
        if adapter_tmp is not None:
            adapter_tmp.cleanup()
    if args.csv:
        write_csv_from_jsonl(args.output, args.csv)
    if args.summary_json:
        write_summary_from_jsonl(args.output, args.summary_json)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
