from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any

from reasoning_assets import verify


TASK_TYPE_MAP = {
    "bit": "bit_manipulation",
    "bit_manipulation": "bit_manipulation",
    "cipher": "text_cipher",
    "text_cipher": "text_cipher",
    "numeral": "numeral_system",
    "numeral_system": "numeral_system",
    "gravity": "gravity_physics",
    "gravity_physics": "gravity_physics",
    "unit_conversion": "unit_conversion",
    "equation_symbolic": "symbolic_equation",
    "equation_numeric": "symbolic_equation",
    "symbolic_equation": "symbolic_equation",
}


ROMAN_RE = re.compile(r"^[IVXLCDM]+$", re.I)
BINARY_RE = re.compile(r"^[01]{8}$")
NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


@dataclass
class RewardResult:
    extracted_answer: str
    reward: float
    terms: dict[str, float]
    flags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_task_type(value: object) -> str:
    raw = str(value or "").strip()
    return TASK_TYPE_MAP.get(raw, raw or "unknown")


def clean_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def normalize_answer(value: object) -> str:
    return re.sub(r"\s+", " ", clean_text(value)).strip()


def boxed_answers(text: str) -> list[str]:
    answers: list[str] = []
    needle = r"\boxed{"
    start = 0
    while True:
        pos = text.find(needle, start)
        if pos < 0:
            break
        idx = pos + len(needle)
        depth = 1
        chars: list[str] = []
        while idx < len(text) and depth:
            char = text[idx]
            if char == "{":
                depth += 1
                chars.append(char)
            elif char == "}":
                depth -= 1
                if depth:
                    chars.append(char)
            else:
                chars.append(char)
            idx += 1
        if depth == 0:
            answers.append(normalize_answer("".join(chars)))
        start = max(idx, pos + len(needle))
    return answers


def extract_answer(completion: object, generated_answer: object = "") -> str:
    generated = normalize_answer(generated_answer)
    if generated:
        return generated
    text = clean_text(completion)
    boxed = boxed_answers(text)
    if boxed:
        return boxed[-1]
    for pattern in (
        r"final answer\s*[:=]\s*([^\n]+)$",
        r"answer\s*[:=]\s*([^\n]+)$",
    ):
        match = re.search(pattern, text, flags=re.I | re.M)
        if match:
            return normalize_answer(match.group(1))
    return ""


def numeric_value(value: str) -> float | None:
    if not NUMBER_RE.fullmatch(value):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def sequence_ratio(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left.lower(), right.lower()).ratio()


def bit_partial(answer: str, pred: str) -> tuple[float, list[str]]:
    flags: list[str] = []
    if not pred:
        return 0.0, ["missing_final"]
    if not re.fullmatch(r"[01]+", pred):
        return 0.0, ["malformed_binary"]
    if len(pred) != len(answer):
        flags.append("wrong_binary_length")
    overlap = min(len(answer), len(pred))
    if overlap == 0:
        return 0.0, flags
    matches = sum(1 for a, b in zip(answer[:overlap], pred[:overlap]) if a == b)
    return matches / max(len(answer), len(pred)), flags


def copied_example_penalty(prompt: str, pred: str, answer: str) -> float:
    if not pred or pred == answer:
        return 0.0
    outputs = set(re.findall(r"->\s*([01]{8})", prompt))
    return -0.3 if pred in outputs else 0.0


def text_partial(answer: str, pred: str) -> tuple[float, list[str]]:
    flags: list[str] = []
    if not pred:
        return 0.0, ["missing_final"]
    ratio = sequence_ratio(answer, pred)
    answer_words = answer.lower().split()
    pred_words = pred.lower().split()
    if len(answer_words) != len(pred_words):
        flags.append("wrong_word_count")
    if [len(w) for w in answer_words] != [len(w) for w in pred_words]:
        flags.append("word_length_mismatch")
    word_hits = 0.0
    if answer_words:
        pred_set = set(pred_words)
        word_hits = sum(1 for word in answer_words if word in pred_set) / len(answer_words)
    return 0.65 * ratio + 0.35 * word_hits, flags


def numeric_partial(answer: str, pred: str) -> tuple[float, list[str]]:
    flags: list[str] = []
    target = numeric_value(answer)
    value = numeric_value(pred)
    if target is None or value is None:
        return sequence_ratio(answer, pred), ["non_numeric_partial"]
    denom = max(abs(target), 1.0)
    rel = abs(target - value) / denom
    if rel > 0.01:
        flags.append("numeric_outside_metric_tolerance")
    return max(0.0, 1.0 - min(rel / 0.25, 1.0)), flags


def generic_partial(answer: str, pred: str) -> tuple[float, list[str]]:
    if re.fullmatch(r"[01]+", answer):
        return bit_partial(answer, pred)
    if NUMBER_RE.fullmatch(answer):
        return numeric_partial(answer, pred)
    if ROMAN_RE.fullmatch(answer):
        flags = [] if ROMAN_RE.fullmatch(pred) else ["malformed_roman"]
        return sequence_ratio(answer, pred), flags
    return text_partial(answer, pred)


def reward_completion(
    prompt: object,
    completion: object,
    answer: object,
    task_type: object,
    generated_answer: object = "",
) -> RewardResult:
    prompt_text = clean_text(prompt)
    completion_text = clean_text(completion)
    answer_text = normalize_answer(answer)
    pred = extract_answer(completion_text, generated_answer)
    task = normalize_task_type(task_type)

    boxed = boxed_answers(completion_text)
    exact = 1.0 if pred and verify(answer_text, pred) else 0.0
    partial, partial_flags = generic_partial(answer_text, pred)
    format_valid = 0.0
    copy_penalty = 0.0
    flags = list(partial_flags)

    if task == "bit_manipulation":
        format_valid = 1.0 if BINARY_RE.fullmatch(pred) else 0.0
        copy_penalty = copied_example_penalty(prompt_text, pred, answer_text)
    elif task == "text_cipher":
        format_valid = 1.0 if pred and re.fullmatch(r"[a-z ]+", pred, re.I) else 0.0
    elif task in {"gravity_physics", "unit_conversion", "symbolic_equation"}:
        if NUMBER_RE.fullmatch(answer_text):
            format_valid = 1.0 if NUMBER_RE.fullmatch(pred) else 0.0
        elif re.fullmatch(r"[01]+", answer_text):
            format_valid = 1.0 if re.fullmatch(r"[01]+", pred) else 0.0
        else:
            format_valid = 1.0 if pred else 0.0
    elif task == "numeral_system":
        format_valid = 1.0 if ROMAN_RE.fullmatch(pred) else 0.0
    else:
        format_valid = 1.0 if pred else 0.0

    boxed_valid = 1.0 if boxed else 0.0
    trace_parseable = 1.0 if pred else 0.0
    if boxed and len(boxed) > 1:
        flags.append("multiple_boxed")
    if not pred:
        flags.append("missing_extracted_answer")
    if exact:
        flags.append("answer_exact")
    else:
        flags.append("wrong_final")

    terms = {
        "boxed_valid": boxed_valid,
        "answer_exact": exact,
        "answer_partial": partial,
        "format_valid": format_valid,
        "trace_parseable": trace_parseable,
        "copy_example_penalty": copy_penalty,
    }
    reward = (
        0.58 * exact
        + 0.24 * partial
        + 0.08 * format_valid
        + 0.06 * trace_parseable
        + 0.04 * boxed_valid
        + copy_penalty
    )
    reward = max(0.0, min(1.0, reward))
    return RewardResult(
        extracted_answer=pred,
        reward=round(reward, 6),
        terms={key: round(value, 6) for key, value in terms.items()},
        flags=sorted(set(flags)),
    )
