from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from analyze_data import classify_prompt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
GENERATED_DIR = ROOT / "data" / "generated"


@dataclass
class SolveResult:
    prediction: str | None
    trace: str
    method: str


def verify(stored_answer: object, predicted: object) -> bool:
    stored = str(stored_answer).strip()
    pred = str(predicted).strip()
    if re.fullmatch(r"[01]+", stored):
        return pred.lower() == stored.lower()
    try:
        return math.isclose(float(stored), float(pred), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return pred.lower() == stored.lower()


def fmt2(value: float) -> str:
    return f"{value:.2f}"


def infer_scalar_from_rounded_pairs(pairs: list[tuple[float, float]], power: int) -> float:
    intervals = []
    for x_value, y_value in pairs:
        denom = x_value**power
        intervals.append(((y_value - 0.005) / denom, (y_value + 0.005) / denom))

    low = max(item[0] for item in intervals)
    high = min(item[1] for item in intervals)
    if low <= high:
        return (low + high) / 2

    # Fall back to a through-origin least-squares estimate when rounded
    # demonstrations have no exact overlapping interval.
    xs = [x_value**power for x_value, _ in pairs]
    ys = [y_value for _, y_value in pairs]
    denom = sum(x * x for x in xs)
    return sum(x * y for x, y in zip(xs, ys)) / denom


def int_to_roman(number: int) -> str:
    values = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    parts = []
    rest = number
    for value, symbol in values:
        count, rest = divmod(rest, value)
        if count:
            parts.append(symbol * count)
    return "".join(parts)


def solve_numeral(prompt: str) -> SolveResult:
    match = re.search(r"write the number\s+(\d+)\s+in the Wonderland numeral system", prompt, re.I)
    if not match:
        return SolveResult(None, "Could not find the target number.", "numeral")

    number = int(match.group(1))
    answer = int_to_roman(number)
    trace = (
        f"NUMERAL\n"
        f"Target number: {number}\n"
        f"Roman conversion: {number} -> {answer}\n"
        f"Final answer: \\boxed{{{answer}}}"
    )
    return SolveResult(answer, trace, "roman_forward")


def solve_gravity(prompt: str) -> SolveResult:
    pairs = [
        (float(t_value), float(d_value))
        for t_value, d_value in re.findall(
            r"For t = ([0-9]+(?:\.[0-9]+)?)s, distance = ([0-9]+(?:\.[0-9]+)?) m",
            prompt,
        )
    ]
    target_match = re.search(
        r"determine the falling distance for t = ([0-9]+(?:\.[0-9]+)?)s", prompt, re.I
    )
    if not pairs or not target_match:
        return SolveResult(None, "Could not parse gravity examples.", "gravity")

    rate = infer_scalar_from_rounded_pairs(pairs, power=2)
    target = float(target_match.group(1))
    prediction = fmt2(rate * target * target)
    trace = (
        "GRAVITY\n"
        f"Use d = rate * t^2 where rate = 0.5*g.\n"
        f"Rounded-example interval estimate gives rate = {rate:.6f}.\n"
        f"Target t^2 = {target:.4f}^2 = {target * target:.6f}.\n"
        f"Distance = {rate:.6f} * {target * target:.6f} = {float(prediction):.2f}.\n"
        f"Final answer: \\boxed{{{prediction}}}"
    )
    return SolveResult(prediction, trace, "gravity_interval")


def solve_unit_conversion(prompt: str) -> SolveResult:
    pairs = [
        (float(input_value), float(output_value))
        for input_value, output_value in re.findall(
            r"([0-9]+(?:\.[0-9]+)?) m becomes ([0-9]+(?:\.[0-9]+)?)",
            prompt,
        )
    ]
    target_match = re.search(
        r"convert the following measurement:\s*([0-9]+(?:\.[0-9]+)?) m", prompt, re.I
    )
    if not pairs or not target_match:
        return SolveResult(None, "Could not parse unit conversion examples.", "unit_conversion")

    rate = infer_scalar_from_rounded_pairs(pairs, power=1)
    target = float(target_match.group(1))
    prediction = fmt2(rate * target)
    trace = (
        "UNIT_CONVERSION\n"
        f"Use output = rate * input.\n"
        f"Rounded-example interval estimate gives rate = {rate:.6f}.\n"
        f"Target output = {rate:.6f} * {target:.4f} = {float(prediction):.2f}.\n"
        f"Final answer: \\boxed{{{prediction}}}"
    )
    return SolveResult(prediction, trace, "unit_interval")


def iter_plaintext_phrases(prompts: Iterable[str]) -> Iterable[str]:
    for prompt in prompts:
        for _, plain in re.findall(r"^(.+?) -> ([a-z ]+)$", prompt, flags=re.M):
            yield plain.strip().lower()


def build_cipher_vocab(prompts: Iterable[str]) -> Counter[str]:
    vocab: Counter[str] = Counter()
    for phrase in iter_plaintext_phrases(prompts):
        for word in phrase.split():
            vocab[word] += 1
    return vocab


def cipher_candidates(
    cipher_word: str,
    vocab: Counter[str],
    mapping: dict[str, str],
    inverse: dict[str, str],
) -> list[str]:
    candidates = []
    for word, freq in vocab.items():
        if len(word) != len(cipher_word):
            continue
        local_map = dict(mapping)
        local_inverse = dict(inverse)
        ok = True
        for c_char, p_char in zip(cipher_word, word):
            if c_char in local_map and local_map[c_char] != p_char:
                ok = False
                break
            if p_char in local_inverse and local_inverse[p_char] != c_char:
                ok = False
                break
            local_map[c_char] = p_char
            local_inverse[p_char] = c_char
        if ok:
            candidates.append(word)
    candidates.sort(key=lambda item: (-vocab[item], item))
    return candidates


def solve_text_cipher(prompt: str, vocab: Counter[str]) -> SolveResult:
    examples = re.findall(r"^([a-z ]+) -> ([a-z ]+)$", prompt, flags=re.M)
    target_match = re.search(r"decrypt the following text:\s*([a-z ]+)", prompt, re.I)
    if not examples or not target_match:
        return SolveResult(None, "Could not parse cipher examples.", "text_cipher")

    mapping: dict[str, str] = {}
    inverse: dict[str, str] = {}
    for cipher_text, plain_text in examples:
        for c_char, p_char in zip(cipher_text, plain_text):
            if c_char == " " or p_char == " ":
                continue
            if c_char in mapping and mapping[c_char] != p_char:
                continue
            if p_char in inverse and inverse[p_char] != c_char:
                continue
            mapping[c_char] = p_char
            inverse[p_char] = c_char

    target_words = target_match.group(1).strip().split()
    candidate_lists = [
        cipher_candidates(cipher_word, vocab, mapping, inverse) for cipher_word in target_words
    ]
    if any(not items for items in candidate_lists):
        decoded = []
        for word in target_words:
            decoded.append("".join(mapping.get(char, "?") for char in word))
        prediction = " ".join(decoded)
        return SolveResult(
            prediction if "?" not in prediction else None,
            f"CIPHER\nPartial decode failed to fill all words: {prediction}",
            "cipher_partial",
        )

    order = sorted(range(len(target_words)), key=lambda idx: len(candidate_lists[idx]))
    best_words: list[str] | None = None
    best_score = -1

    def backtrack(
        position: int,
        current_mapping: dict[str, str],
        current_inverse: dict[str, str],
        chosen: list[str | None],
        score: int,
    ) -> None:
        nonlocal best_words, best_score
        if position == len(order):
            words = [str(item) for item in chosen]
            if score > best_score:
                best_score = score
                best_words = words
            return

        idx = order[position]
        cipher_word = target_words[idx]
        for word in candidate_lists[idx][:80]:
            local_mapping = dict(current_mapping)
            local_inverse = dict(current_inverse)
            ok = True
            for c_char, p_char in zip(cipher_word, word):
                if c_char in local_mapping and local_mapping[c_char] != p_char:
                    ok = False
                    break
                if p_char in local_inverse and local_inverse[p_char] != c_char:
                    ok = False
                    break
                local_mapping[c_char] = p_char
                local_inverse[p_char] = c_char
            if not ok:
                continue
            chosen[idx] = word
            backtrack(
                position + 1,
                local_mapping,
                local_inverse,
                chosen,
                score + vocab[word],
            )
            chosen[idx] = None

    backtrack(0, mapping, inverse, [None] * len(target_words), 0)
    if best_words is None:
        return SolveResult(None, "CIPHER\nNo consistent vocabulary fill found.", "cipher_backtrack")

    prediction = " ".join(best_words)
    trace = (
        "CIPHER\n"
        f"Extracted {len(mapping)} cipher-letter mappings from examples.\n"
        f"Target words: {' / '.join(target_words)}\n"
        f"Vocabulary-consistent decode: {prediction}\n"
        f"Final answer: \\boxed{{{prediction}}}"
    )
    return SolveResult(prediction, trace, "cipher_vocab_backtrack")


BitFunc = Callable[[list[int]], int]


@dataclass(frozen=True)
class BitCandidate:
    name: str
    cost: int
    func: BitFunc


def build_bit_candidates() -> list[BitCandidate]:
    candidates: list[BitCandidate] = [
        BitCandidate("C0", 0, lambda bits: 0),
        BitCandidate("C1", 0, lambda bits: 1),
    ]

    for idx in range(8):
        candidates.append(BitCandidate(f"B{idx}", 1, lambda bits, idx=idx: bits[idx]))
        candidates.append(BitCandidate(f"NB{idx}", 1, lambda bits, idx=idx: 1 - bits[idx]))

    binary_ops: list[tuple[str, Callable[[int, int], int]]] = [
        ("AND", lambda a, b: a & b),
        ("OR", lambda a, b: a | b),
        ("XOR", lambda a, b: a ^ b),
        ("XNOR", lambda a, b: 1 - (a ^ b)),
        ("NAND", lambda a, b: 1 - (a & b)),
        ("NOR", lambda a, b: 1 - (a | b)),
        ("A_AND_NOT_B", lambda a, b: a & (1 - b)),
        ("NOT_A_AND_B", lambda a, b: (1 - a) & b),
        ("A_OR_NOT_B", lambda a, b: a | (1 - b)),
        ("NOT_A_OR_B", lambda a, b: (1 - a) | b),
    ]
    for op_name, op_func in binary_ops:
        for left in range(8):
            for right in range(8):
                if left == right:
                    continue
                candidates.append(
                    BitCandidate(
                        f"{op_name}(B{left},B{right})",
                        2,
                        lambda bits, left=left, right=right, op_func=op_func: op_func(
                            bits[left], bits[right]
                        ),
                    )
                )
    return candidates


BIT_CANDIDATES = build_bit_candidates()

BIT_TRANSFORM_SPECS = [
    (f"{kind}{shift}", kind, shift)
    for kind in ("ROT", "SHL", "SHR")
    for shift in range(1, 8)
]

BIT_GLOBAL_OPS = ("AND", "ANDN", "OR", "ORN", "XOR", "XORN")
BIT_MASK = 0xFF


def transform_int(value: int, kind: str, shift: int) -> int:
    value &= BIT_MASK
    if kind == "ROT":
        return ((value << shift) | (value >> (8 - shift))) & BIT_MASK
    if kind == "SHL":
        return value >> shift
    if kind == "SHR":
        return (value << shift) & BIT_MASK
    raise KeyError(kind)


def bit_op_int(left: int, right: int, op_name: str) -> int:
    if op_name == "AND":
        return left & right
    if op_name == "ANDN":
        return left & (~right & BIT_MASK)
    if op_name == "OR":
        return (left | right) & BIT_MASK
    if op_name == "ORN":
        return (left | (~right & BIT_MASK)) & BIT_MASK
    if op_name == "XOR":
        return (left ^ right) & BIT_MASK
    if op_name == "XORN":
        return (left ^ (~right & BIT_MASK)) & BIT_MASK
    raise KeyError(op_name)


def bit_string(value: int) -> str:
    return f"{value & BIT_MASK:08b}"


def solve_bit_global(prompt: str) -> SolveResult | None:
    examples = re.findall(r"([01]{8}) -> ([01]{8})", prompt)
    target_match = re.search(r"output for:\s*([01]{8})", prompt, re.I)
    if not examples or not target_match:
        return None

    input_values = [int(item[0], 2) for item in examples]
    output_values = tuple(int(item[1], 2) for item in examples)
    target_value = int(target_match.group(1), 2)
    transform_vectors = [
        (
            name,
            tuple(transform_int(value, kind, shift) for value in input_values),
            transform_int(target_value, kind, shift),
        )
        for name, kind, shift in BIT_TRANSFORM_SPECS
    ]

    for name, signature, target_prediction in transform_vectors:
        if signature == output_values:
            prediction = bit_string(target_prediction)
            trace = (
                "BIT_MANIPULATION\n"
                "Matched one global shift/rotation transform against every example.\n"
                f"Expression: {name}\n"
                f"Target input: {target_match.group(1)} -> {prediction}\n"
                f"Final answer: \\boxed{{{prediction}}}"
            )
            return SolveResult(prediction, trace, "bit_global_unary")

    binary_cache = []
    for left_name, left_signature, left_target in transform_vectors:
        for right_name, right_signature, right_target in transform_vectors:
            for op_name in BIT_GLOBAL_OPS:
                signature = tuple(
                    bit_op_int(left, right, op_name)
                    for left, right in zip(left_signature, right_signature)
                )
                target_prediction = bit_op_int(left_target, right_target, op_name)
                expression = f"({left_name} {op_name} {right_name})"
                if signature == output_values:
                    prediction = bit_string(target_prediction)
                    trace = (
                        "BIT_MANIPULATION\n"
                        "Matched a two-transform global bit expression against every example.\n"
                        f"Expression: {expression}\n"
                        f"Target input: {target_match.group(1)} -> {prediction}\n"
                        f"Final answer: \\boxed{{{prediction}}}"
                    )
                    return SolveResult(prediction, trace, "bit_global_binary")
                binary_cache.append((expression, signature, target_prediction))

    for left_expression, left_signature, left_target in binary_cache:
        for right_name, right_signature, right_target in transform_vectors:
            for op_name in BIT_GLOBAL_OPS:
                signature = tuple(
                    bit_op_int(left, right, op_name)
                    for left, right in zip(left_signature, right_signature)
                )
                if signature == output_values:
                    target_prediction = bit_op_int(left_target, right_target, op_name)
                    prediction = bit_string(target_prediction)
                    expression = f"({left_expression} {op_name} {right_name})"
                    trace = (
                        "BIT_MANIPULATION\n"
                        "Matched a three-transform global bit expression against every example.\n"
                        f"Expression: {expression}\n"
                        f"Target input: {target_match.group(1)} -> {prediction}\n"
                        f"Final answer: \\boxed{{{prediction}}}"
                    )
                    return SolveResult(prediction, trace, "bit_global_ternary")

    return None


def bits_from_string(value: str) -> list[int]:
    return [1 if char == "1" else 0 for char in value]


def solve_bit_manipulation(prompt: str) -> SolveResult:
    global_result = solve_bit_global(prompt)
    if global_result is not None:
        return global_result

    examples = re.findall(r"([01]{8}) -> ([01]{8})", prompt)
    target_match = re.search(r"output for:\s*([01]{8})", prompt, re.I)
    if not examples or not target_match:
        return SolveResult(None, "Could not parse bit examples.", "bit_manipulation")

    input_bits = [bits_from_string(item[0]) for item in examples]
    output_bits = [bits_from_string(item[1]) for item in examples]
    target_bits = bits_from_string(target_match.group(1))

    answer_bits = []
    selected = []
    ambiguous = 0
    missing = 0
    for out_idx in range(8):
        column = [bits[out_idx] for bits in output_bits]
        matches = []
        for candidate in BIT_CANDIDATES:
            if [candidate.func(bits) for bits in input_bits] == column:
                matches.append(candidate)

        if not matches:
            missing += 1
            bit = 1 if sum(column) >= len(column) / 2 else 0
            answer_bits.append(str(bit))
            selected.append(f"O{out_idx}=FALLBACK{bit}")
            continue

        predictions = defaultdict(list)
        for candidate in matches:
            predictions[candidate.func(target_bits)].append(candidate)
        if len(predictions) > 1:
            ambiguous += 1

        best_bit, best_group = sorted(
            predictions.items(),
            key=lambda item: (
                min(candidate.cost for candidate in item[1]),
                len(item[1]),
                item[0],
            ),
        )[0]
        best_candidate = sorted(best_group, key=lambda item: (item.cost, item.name))[0]
        answer_bits.append(str(best_bit))
        selected.append(f"O{out_idx}={best_candidate.name}->{best_bit}")

    prediction = "".join(answer_bits)
    trace = (
        "BIT_MANIPULATION\n"
        f"Matched each output bit independently with constants, unary bits, and binary gates.\n"
        f"Ambiguous output-bit matches: {ambiguous}; fallback bits: {missing}.\n"
        + "\n".join(selected)
        + f"\nFinal answer: \\boxed{{{prediction}}}"
    )
    return SolveResult(prediction, trace, "bit_per_output_gate")


def parse_numeric_equation(prompt: str) -> tuple[list[tuple[str, str, str, str]], tuple[str, str, str] | None]:
    examples = []
    for lhs, rhs in re.findall(r"^(.+?) = (.+?)$", prompt, flags=re.M):
        match = re.fullmatch(r"(\d{2})(.)(\d{2})", lhs.strip())
        if match:
            examples.append((*match.groups(), rhs.strip()))

    target_match = re.search(r"Now, determine the result for: (.+)$", prompt, flags=re.M)
    if not target_match:
        return examples, None
    target_lhs = re.fullmatch(r"(\d{2})(.)(\d{2})", target_match.group(1).strip())
    if not target_lhs:
        return examples, None
    return examples, target_lhs.groups()


def numeric_operands(ab: str, cd: str, mode: str) -> tuple[int, int, str, str]:
    values = {
        "AB_CD": (ab, cd),
        "BA_DC": (ab[::-1], cd[::-1]),
        "AB_DC": (ab, cd[::-1]),
        "BA_CD": (ab[::-1], cd),
        "CD_AB": (cd, ab),
        "DC_BA": (cd[::-1], ab[::-1]),
    }[mode]
    return int(values[0]), int(values[1]), values[0], values[1]


def numeric_op_value(x_value: int, y_value: int, x_text: str, y_text: str, op_name: str) -> int | str | None:
    if op_name == "add":
        return x_value + y_value
    if op_name == "add1":
        return x_value + y_value + 1
    if op_name == "addm1":
        return x_value + y_value - 1
    if op_name == "sub":
        return x_value - y_value
    if op_name == "sub1":
        return x_value - y_value + 1
    if op_name == "subm1":
        return x_value - y_value - 1
    if op_name == "rsub":
        return y_value - x_value
    if op_name == "rsub1":
        return y_value - x_value + 1
    if op_name == "rsubm1":
        return y_value - x_value - 1
    if op_name == "absdiff":
        return abs(x_value - y_value)
    if op_name == "absdiff1":
        return abs(x_value - y_value) + 1
    if op_name == "absdiffm1":
        return abs(x_value - y_value) - 1
    if op_name == "negabsdiff":
        return -abs(x_value - y_value)
    if op_name == "mul":
        return x_value * y_value
    if op_name == "mul1":
        return x_value * y_value + 1
    if op_name == "mulm1":
        return x_value * y_value - 1
    if op_name == "floordiv" and y_value != 0:
        return x_value // y_value
    if op_name == "rfloordiv" and x_value != 0:
        return y_value // x_value
    if op_name == "mod" and y_value != 0:
        return x_value % y_value
    if op_name == "rmod" and x_value != 0:
        return y_value % x_value
    if op_name == "concat":
        return x_text + y_text
    if op_name == "rconcat":
        return y_text + x_text
    if op_name == "first":
        return x_value
    if op_name == "second":
        return y_value
    if op_name == "dsum":
        return sum(map(int, str(abs(x_value)))) + sum(map(int, str(abs(y_value))))
    return None


def reverse_numeric_text(value: str) -> str:
    sign = ""
    if value.startswith("-"):
        sign = "-"
        value = value[1:]
    return sign + value[::-1]


def numeric_equation_formats(value: int | str | None, op_char: str) -> list[tuple[str, str]]:
    if value is None:
        return []
    base = str(value)
    reversed_base = reverse_numeric_text(base)
    outputs = [
        ("raw", base),
        ("rev", reversed_base),
        ("raw_op_suffix", base + op_char),
        ("raw_op_prefix", op_char + base),
        ("rev_op_suffix", reversed_base + op_char),
        ("rev_op_prefix", op_char + reversed_base),
    ]
    if isinstance(value, int):
        sign = "-" if value < 0 else ""
        digits = str(abs(value))
        for width in (2, 3, 4):
            if len(digits) < width:
                padded = sign + digits.zfill(width)
                outputs.append((f"zpad{width}", padded))
                outputs.append((f"zpad{width}_rev", reverse_numeric_text(padded)))

    seen = set()
    unique = []
    for name, output in outputs:
        if (name, output) in seen:
            continue
        seen.add((name, output))
        unique.append((name, output))
    return unique


EQUATION_CANDIDATE_PRIORITY = [
    (mode, op_name)
    for mode in ("BA_DC", "AB_CD", "AB_DC", "BA_CD")
    for op_name in (
        "add",
        "mul",
        "sub",
        "absdiff",
        "concat",
        "add1",
        "mulm1",
        "mul1",
        "addm1",
        "negabsdiff",
        "subm1",
        "sub1",
        "rsub",
        "rsubm1",
        "rsub1",
        "rconcat",
        "floordiv",
        "rfloordiv",
        "mod",
        "rmod",
        "first",
        "second",
        "dsum",
        "absdiff1",
        "absdiffm1",
    )
]

EQUATION_FORMAT_PRIORITY = [
    "raw",
    "rev",
    "raw_op_suffix",
    "rev_op_suffix",
    "raw_op_prefix",
    "rev_op_prefix",
    "zpad2",
    "zpad2_rev",
    "zpad3",
    "zpad3_rev",
    "zpad4",
    "zpad4_rev",
]


CRYPT_DIGIT_MODES = ("BA_DC", "AB_CD", "AB_DC", "BA_CD", "CD_AB", "DC_BA")
CRYPT_DIGIT_FORMATS = (
    "raw",
    "rev",
    "zpad2",
    "zpad2_rev",
    "zpad3",
    "zpad3_rev",
    "zpad4",
    "zpad4_rev",
)
CRYPT_DIGIT_OPS = tuple(dict.fromkeys(op_name for _, op_name in EQUATION_CANDIDATE_PRIORITY))
MAX_CIPHER_DIGIT_SYMBOLS = 7
# Cap candidates per (mode, op, fmt) to keep the backtracking from exploding on
# degenerate prompts. This no-gold path is only a trace bonus; the gold-oracle
# trace generator handles cryptarithms more thoroughly.
CIPHER_ITEM_CANDIDATE_CAP = 60
# Hard ceiling on backtracking DFS nodes per (mode, op, fmt). Keep this modest:
# a previous 50k cap made full no-gold evaluation spend minutes on ambiguous
# cipher prompts while adding only a handful of correct traces.
CIPHER_BACKTRACK_NODE_CAP = 5_000
CIPHER_DIGIT_DEADLINE_SECONDS = 0.20


def _cipher_digit_formats(value: int | str | None) -> dict[str, str]:
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


def _merge_cipher_maps(
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


def _cipher_item_maps(
    lhs: str,
    rhs: str,
    mode: str,
    op_name: str,
    fmt_name: str,
    deadline: float,
) -> list[dict[str, str]] | None:
    """Enumerate mappings consistent with one cipher (lhs, rhs) under a hypothesis.

    Returns ``None`` when the candidate set blows past ``CIPHER_ITEM_CANDIDATE_CAP``
    so the caller can drop this whole (mode, op, fmt) hypothesis instead of letting
    the outer backtracking churn.
    """
    input_symbols: list[str] = []
    for symbol in lhs[:2] + lhs[3:]:
        if symbol not in input_symbols:
            input_symbols.append(symbol)

    if len(input_symbols) > MAX_CIPHER_DIGIT_SYMBOLS:
        return []

    candidates: list[dict[str, str]] = []
    for values in itertools.permutations("0123456789", len(input_symbols)):
        if time.monotonic() > deadline:
            return None
        mapping = dict(zip(input_symbols, values))
        ab_text = mapping[lhs[0]] + mapping[lhs[1]]
        cd_text = mapping[lhs[3]] + mapping[lhs[4]]
        x_value, y_value, x_text, y_text = numeric_operands(ab_text, cd_text, mode)
        value = numeric_op_value(x_value, y_value, x_text, y_text, op_name)
        decoded_rhs = _cipher_digit_formats(value).get(fmt_name)
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
            if len(candidates) > CIPHER_ITEM_CANDIDATE_CAP:
                return None
    return candidates


def solve_cipher_digit_equation(prompt: str) -> SolveResult | None:
    deadline = time.monotonic() + CIPHER_DIGIT_DEADLINE_SECONDS
    examples, target = parse_cryptarithm(prompt)
    if not examples or target is None or len(target) != 5:
        return None
    # Cipher RHS is the cipher-encoded numeric result and is typically 1..5 chars
    # (one symbol per digit, sometimes with sign). The earlier `len(rhs) == len(lhs)`
    # check rejected every real prompt; keep only the LHS shape constraint.
    if not all(len(lhs) == 5 for lhs, _rhs in examples):
        return None

    target_op = target[2]
    relevant = [item for item in examples if item[0][2] == target_op]
    if len(relevant) < 2:
        return None

    for mode in CRYPT_DIGIT_MODES:
        for op_name in CRYPT_DIGIT_OPS:
            for fmt_name in CRYPT_DIGIT_FORMATS:
                if time.monotonic() > deadline:
                    return None
                candidate_lists: list[list[dict[str, str]]] = []
                hypothesis_dead = False
                for lhs, rhs in relevant:
                    maps = _cipher_item_maps(lhs, rhs, mode, op_name, fmt_name, deadline)
                    if maps is None or not maps:
                        hypothesis_dead = True
                        break
                    candidate_lists.append(maps)
                if hypothesis_dead:
                    continue

                order = sorted(range(len(candidate_lists)), key=lambda idx: len(candidate_lists[idx]))
                solutions: list[dict[str, str]] = []
                node_count = [0]
                aborted = [False]

                def backtrack(position: int, mapping: dict[str, str], inverse: dict[str, str]) -> None:
                    if aborted[0] or len(solutions) > 1:
                        return
                    if time.monotonic() > deadline:
                        aborted[0] = True
                        return
                    node_count[0] += 1
                    if node_count[0] > CIPHER_BACKTRACK_NODE_CAP:
                        aborted[0] = True
                        return
                    if position == len(order):
                        solutions.append(mapping)
                        return
                    for candidate in candidate_lists[order[position]]:
                        merged = _merge_cipher_maps(mapping, inverse, candidate)
                        if merged is None:
                            continue
                        backtrack(position + 1, merged[0], merged[1])

                backtrack(0, {}, {})
                if aborted[0] or len(solutions) != 1:
                    continue

                mapping = solutions[0]
                target_symbols = [target[0], target[1], target[3], target[4]]
                if not all(symbol in mapping for symbol in target_symbols):
                    continue
                ab_text = mapping[target[0]] + mapping[target[1]]
                cd_text = mapping[target[3]] + mapping[target[4]]
                x_value, y_value, x_text, y_text = numeric_operands(ab_text, cd_text, mode)
                value = numeric_op_value(x_value, y_value, x_text, y_text, op_name)
                prediction = _cipher_digit_formats(value).get(fmt_name)
                if prediction is None:
                    continue

                trace = (
                    "CRYPTARITHM_DEDUCE\n"
                    f"Target operator {target_op} appears in {len(relevant)} demonstration(s).\n"
                    f"Recovered rule: operands={mode}; operation={op_name}; format={fmt_name}.\n"
                    f"Recovered symbol mapping: {json.dumps(mapping, sort_keys=True, ensure_ascii=False)}.\n"
                    f"Apply to target {target} -> {prediction}.\n"
                    f"Final answer: \\boxed{{{prediction}}}"
                )
                return SolveResult(prediction, trace, "cryptarithm_digit_deduce")

    return None


def solve_symbolic_equation(prompt: str) -> SolveResult:
    examples, target = parse_numeric_equation(prompt)
    if not examples or target is None:
        cryptarithm_result = solve_cryptarithm_concat(prompt)
        if cryptarithm_result is not None:
            return cryptarithm_result
        cipher_digit_result = solve_cipher_digit_equation(prompt)
        if cipher_digit_result is not None:
            return cipher_digit_result
        return SolveResult(None, "No symbolic equation solver match.", "missing")

    target_ab, target_op, target_cd = target
    relevant_examples = [item for item in examples if item[1] == target_op]
    if not relevant_examples:
        # Public writeups note that absent target operators are a weaker guess-only
        # case. Keep it separated so trace training can distinguish proof from prior.
        mode, op_name, fmt_name = ("BA_DC", "add", "rev")
        x_value, y_value, x_text, y_text = numeric_operands(target_ab, target_cd, mode)
        value = numeric_op_value(x_value, y_value, x_text, y_text, op_name)
        prediction = dict(numeric_equation_formats(value, target_op)).get(fmt_name)
        trace = (
            "NUMERIC_EQUATION_GUESS\n"
            f"Target operator {target_op} does not appear in demonstrations.\n"
            f"Use prior rule {mode}|{op_name}|{fmt_name}.\n"
            f"Final answer: \\boxed{{{prediction}}}"
        )
        return SolveResult(prediction, trace, "equation_numeric_guess")

    matches: list[tuple[str, str, str, str]] = []
    for mode, op_name in EQUATION_CANDIDATE_PRIORITY:
        possible_formats: set[str] | None = None
        for ab_text, op_char, cd_text, rhs in relevant_examples:
            x_value, y_value, x_text, y_text = numeric_operands(ab_text, cd_text, mode)
            value = numeric_op_value(x_value, y_value, x_text, y_text, op_name)
            format_names = {
                name
                for name, output in numeric_equation_formats(value, op_char)
                if output == rhs
            }
            possible_formats = format_names if possible_formats is None else possible_formats & format_names
            if not possible_formats:
                break
        if not possible_formats:
            continue

        for fmt_name in sorted(
            possible_formats,
            key=lambda name: (
                EQUATION_FORMAT_PRIORITY.index(name)
                if name in EQUATION_FORMAT_PRIORITY
                else len(EQUATION_FORMAT_PRIORITY)
            ),
        ):
            x_value, y_value, x_text, y_text = numeric_operands(target_ab, target_cd, mode)
            value = numeric_op_value(x_value, y_value, x_text, y_text, op_name)
            prediction = dict(numeric_equation_formats(value, target_op)).get(fmt_name)
            if prediction is not None:
                matches.append((mode, op_name, fmt_name, prediction))

    if not matches:
        return SolveResult(None, "NUMERIC_EQUATION\nNo consistent candidate rule found.", "equation_numeric_missing")

    mode, op_name, fmt_name, prediction = matches[0]
    trace = (
        "NUMERIC_EQUATION\n"
        f"Target operator {target_op} appears in {len(relevant_examples)} demonstration(s).\n"
        f"Locked rule: operands={mode}; operation={op_name}; format={fmt_name}.\n"
        f"Apply to target {target_ab}{target_op}{target_cd} -> {prediction}.\n"
        f"Final answer: \\boxed{{{prediction}}}"
    )
    return SolveResult(prediction, trace, "equation_numeric_deduce")


def parse_cryptarithm(prompt: str) -> tuple[list[tuple[str, str]], str | None]:
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


def cryptarithm_concat_type(items: list[tuple[str, str]]) -> str | None:
    if all(rhs == lhs[0] + lhs[1] + lhs[3] + lhs[4] for lhs, rhs in items):
        return "concat"
    if all(rhs == lhs[3] + lhs[4] + lhs[0] + lhs[1] for lhs, rhs in items):
        return "reverse_concat"
    return None


def solve_cryptarithm_concat(prompt: str) -> SolveResult | None:
    examples, target = parse_cryptarithm(prompt)
    if not examples or target is None:
        return None

    by_op: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for lhs, rhs in examples:
        by_op[lhs[2]].append((lhs, rhs))

    target_op = target[2]
    if target_op in by_op:
        concat_type = cryptarithm_concat_type(by_op[target_op])
        if concat_type is None:
            return None
    else:
        # Public baseline traces use forward concatenation as a weak prior when
        # the target operator never appears. Keep this method separate because
        # it is guessier than a demonstrated operator rule.
        concat_type = "concat"

    if concat_type == "concat":
        prediction = target[0] + target[1] + target[3] + target[4]
        rule_text = "left followed by right"
    else:
        prediction = target[3] + target[4] + target[0] + target[1]
        rule_text = "right followed by left"

    trace = (
        "CRYPTARITHM_CONCAT\n"
        f"Target operator: {target_op}\n"
        f"Detected rule for this operator: {rule_text}.\n"
        f"Target left={target[:2]}, right={target[3:]} -> {prediction}.\n"
        f"Final answer: \\boxed{{{prediction}}}"
    )
    method = (
        "cryptarithm_concat_deduce"
        if target_op in by_op
        else "cryptarithm_concat_guess"
    )
    return SolveResult(prediction, trace, method)


def solve_prompt(prompt: str, task_type: str, vocab: Counter[str]) -> SolveResult:
    if task_type == "numeral_system":
        return solve_numeral(prompt)
    if task_type == "gravity_physics":
        return solve_gravity(prompt)
    if task_type == "unit_conversion":
        return solve_unit_conversion(prompt)
    if task_type == "text_cipher":
        return solve_text_cipher(prompt, vocab)
    if task_type == "bit_manipulation":
        return solve_bit_manipulation(prompt)
    if task_type == "symbolic_equation":
        return solve_symbolic_equation(prompt)
    return SolveResult(None, f"No solver implemented for {task_type}.", "missing")


def evaluate(limit: int | None = None) -> dict:
    train = pd.read_csv(DATA_DIR / "train.csv")
    if limit is not None:
        train = train.head(limit)
    train["task_type"] = train["prompt"].map(classify_prompt)
    vocab = build_cipher_vocab(train["prompt"])

    records = []
    traces = []
    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in train.itertuples(index=False):
        result = solve_prompt(row.prompt, row.task_type, vocab)
        correct = result.prediction is not None and verify(row.answer, result.prediction)
        records.append(
            {
                "id": row.id,
                "task_type": row.task_type,
                "answer": str(row.answer),
                "prediction": result.prediction,
                "correct": correct,
                "method": result.method,
            }
        )
        family_counts[row.task_type]["total"] += 1
        family_counts[row.task_type]["correct"] += int(correct)
        family_counts[row.task_type][result.method] += 1
        if correct:
            traces.append(
                {
                    "id": row.id,
                    "task_type": row.task_type,
                    "prompt": row.prompt,
                    "completion": result.trace,
                    "answer": str(row.answer),
                    "prediction": result.prediction,
                    "method": result.method,
                }
            )

    total = len(records)
    correct = sum(1 for item in records if item["correct"])
    by_family = {}
    for family, counts in sorted(family_counts.items()):
        family_total = counts.pop("total")
        family_correct = counts.pop("correct")
        by_family[family] = {
            "correct": family_correct,
            "total": family_total,
            "accuracy": family_correct / family_total if family_total else 0.0,
            "methods": dict(counts),
        }

    return {
        "overall": {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0.0,
        },
        "by_family": by_family,
        "records": records,
        "traces": traces,
        "cipher_vocab_size": len(vocab),
    }


def write_traces(eval_result: dict, max_records: int | None = None) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GENERATED_DIR / "reasoning_traces.jsonl"

    with out_path.open("w", encoding="utf-8") as output:
        for idx, trace in enumerate(eval_result["traces"], start=1):
            output.write(json.dumps(trace, ensure_ascii=False) + "\n")
            if max_records is not None and idx >= max_records:
                break

    return out_path


def write_reports(eval_result: dict) -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "solver_eval.json").write_text(
        json.dumps(
            {key: value for key, value in eval_result.items() if key != "traces"},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Deterministic Solver Baseline",
        "",
        "This report evaluates local deterministic solvers used to generate SFT traces.",
        "",
        "## Accuracy",
        "",
        "| Family | Correct | Total | Accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for family, item in eval_result["by_family"].items():
        lines.append(
            f"| {family} | {item['correct']} | {item['total']} | {item['accuracy']:.2%} |"
        )
    overall = eval_result["overall"]
    lines.extend(
        [
            f"| **overall** | **{overall['correct']}** | **{overall['total']}** | **{overall['accuracy']:.2%}** |",
            "",
            "## Notes",
            "",
            f"- Cipher vocabulary size from prompt examples: {eval_result['cipher_vocab_size']}.",
            "- Numeral, gravity, and unit conversion are solved directly from parsed structure.",
            "- Text cipher uses prompt-derived substitution mappings plus global Wonderland vocabulary fill.",
            "- Bit manipulation now prioritizes global ROT/SHL/SHR expressions with up to three transforms, then falls back to the per-output-bit gate matcher.",
            "- Symbolic equation remains the main unsolved family; numeric equations are partially solved, while cipher/symbol-digit equations need a stronger mapping search.",
        ]
    )
    (REPORT_DIR / "solver_baseline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--write-traces", action="store_true")
    parser.add_argument("--max-traces", type=int, default=None)
    args = parser.parse_args()

    eval_result = evaluate(limit=args.limit)
    if args.limit is None:
        write_reports(eval_result)
    else:
        print("Skipping report write for limited evaluation.")
    if args.write_traces:
        trace_path = write_traces(eval_result, max_records=args.max_traces)
        print(f"Wrote traces: {trace_path}")
    print(json.dumps(eval_result["overall"], indent=2))
    for family, item in eval_result["by_family"].items():
        print(f"{family}: {item['correct']}/{item['total']} = {item['accuracy']:.2%}")


if __name__ == "__main__":
    main()
