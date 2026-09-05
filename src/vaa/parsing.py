"""Safe, deterministic parsers for generated task responses."""

from __future__ import annotations

import ast
import json
import re
from typing import Any


VERDICT_PATTERN = re.compile(
    r"(?<![A-Za-z])(?:Verdict|Final verdict):[ \t]*(Strong|Weak)\b"
    r"(?P<trailing>[^\r\n]*)"
)

ANSWER_FIELD_PATTERN = re.compile(
    r'''["']answer["']\s*:\s*["']([^"']+)["']''',
    flags=re.IGNORECASE,
)


def parse_strong_weak_verdict(text: str) -> dict[str, Any]:
    stripped = text.strip()
    matches = list(VERDICT_PATTERN.finditer(stripped))
    labels = {match.group(1).upper() for match in matches}
    if not matches or len(labels) != 1:
        return {
            "verdict_valid": False,
            "verdict_label": "MALFORMED",
            "verdict_strong": None,
            "verdict_terminal_period": None,
            "verdict_trailing_text": None,
            "verdict_has_trailing_text": None,
        }
    match = matches[-1]
    trailing_text = stripped[match.end(1) :].strip()
    label = match.group(1).upper()
    return {
        "verdict_valid": True,
        "verdict_label": label,
        "verdict_strong": 1 if label == "STRONG" else 0,
        "verdict_terminal_period": trailing_text == ".",
        "verdict_trailing_text": trailing_text,
        "verdict_has_trailing_text": bool(trailing_text),
    }


def parse_label_response(text: str, true_label: str, false_label: str) -> str:
    normalized = str(text).strip().lower()
    tokens = re.findall(r"[a-z]+", normalized)
    for label in (true_label.lower(), false_label.lower()):
        if label in tokens or normalized.startswith(label):
            return label
    return "other"


def extract_outermost_mapping(
    text: str,
    *,
    allow_python_literals: bool = True,
) -> dict[str, Any] | None:
    """Extract the first balanced JSON object, optionally allowing literals."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    quote = ""
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            continue
        if char in {'"', "'"}:
            in_string = True
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                try:
                    value = json.loads(candidate)
                except json.JSONDecodeError:
                    if not allow_python_literals:
                        return None
                    try:
                        value = ast.literal_eval(candidate)
                    except (SyntaxError, ValueError):
                        return None
                return value if isinstance(value, dict) else None
    return None


def extract_unambiguous_answer_field(
    text: str,
    *,
    allowed_answers: set[str],
) -> str | None:
    """Return one uniquely identified quoted answer label.

    This fallback matches the analysis used for the submitted JSON-format
    prompts when otherwise usable completions contained malformed reasoning
    text or punctuation. Strict JSON validity remains a separate field.
    """
    allowed = {answer.lower() for answer in allowed_answers}
    matches = {
        match.strip().lower().rstrip(".")
        for match in ANSWER_FIELD_PATTERN.findall(str(text))
    }
    valid = matches & allowed
    return next(iter(valid)) if len(valid) == 1 and len(matches) == 1 else None


def parse_json_answer(
    text: str,
    *,
    allowed_answers: set[str],
    correct_answer: str | None = None,
) -> dict[str, Any]:
    """Report strict JSON compliance separately from a usable answer field."""
    parsed = extract_outermost_mapping(text)
    strict_parsed = extract_outermost_mapping(
        text,
        allow_python_literals=False,
    )
    answer_raw = parsed.get("answer") if parsed is not None else None
    mapping_answer = (
        str(answer_raw).strip().lower().rstrip(".")
        if answer_raw is not None
        else None
    )
    allowed = {answer.lower() for answer in allowed_answers}
    strict_valid = bool(
        strict_parsed is not None
        and set(strict_parsed) == {"think", "answer"}
        and mapping_answer in allowed
    )
    fallback_answer = extract_unambiguous_answer_field(
        text,
        allowed_answers=allowed_answers,
    )
    canonical = mapping_answer if mapping_answer in allowed else fallback_answer
    answer_valid = canonical in allowed
    return {
        "json_object_found": parsed is not None,
        "strict_json_valid": strict_valid,
        "answer_parse_mode": (
            "mapping"
            if mapping_answer in allowed
            else "quoted_answer_field"
            if fallback_answer is not None
            else "unparsed"
        ),
        "answer_valid": answer_valid,
        "answer_raw": answer_raw,
        "answer_canonical": canonical if answer_valid else None,
        "correct": (
            bool(canonical == correct_answer.lower())
            if answer_valid and correct_answer is not None
            else None
        ),
    }
