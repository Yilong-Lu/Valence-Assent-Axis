"""Frozen prompt builders and response parsers for generative tasks."""

from __future__ import annotations

from typing import Any, Mapping

from .parsing import extract_outermost_mapping, parse_json_answer
from .prompts import render_prompt


def build_alphabetical_prompts(
    items: list[dict[str, Any]],
    conditions: Mapping[str, str],
    *,
    model_key: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        required = {"item_id", "difficulty", "word_a", "word_b"}
        if not required.issubset(item):
            raise ValueError(f"Incomplete alphabetical item: {item}")
        orders = (
            ("a_first", item["word_a"], item["word_b"]),
            ("b_first", item["word_b"], item["word_a"]),
        )
        for condition, prompt_key in conditions.items():
            for order, option1, option2 in orders:
                statement_truth = str(option1).casefold() < str(option2).casefold()
                correct_answer = "right" if statement_truth else "wrong"
                rows.append(
                    {
                        "task": "alphabetical_order",
                        "item_id": str(item["item_id"]),
                        "prompt_id": f"{item['item_id']}::{condition}::{order}",
                        "condition": condition,
                        "order": order,
                        "difficulty": str(item["difficulty"]),
                        "option1": str(option1),
                        "option2": str(option2),
                        "statement_truth": bool(statement_truth),
                        "correct_answer": correct_answer,
                        "truth_direction": 1 if statement_truth else -1,
                        "prompt_key": prompt_key,
                        "prompt": render_prompt(
                            prompt_key,
                            model_key=model_key,
                            option1=option1,
                            option2=option2,
                        ),
                    }
                )
    return rows


def build_factual_prompts(
    items: list[dict[str, Any]],
    prompt_key: str,
    *,
    model_key: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        if not {"item_id", "question", "true_answer"}.issubset(item):
            raise ValueError(f"Incomplete factual item: {item}")
        true_answer = item["true_answer"]
        if not isinstance(true_answer, bool):
            raise ValueError(f"true_answer must be boolean: {item['item_id']}")
        correct_answer = "yes" if true_answer else "no"
        rows.append(
            {
                "task": "factual_judgment",
                "item_id": str(item["item_id"]),
                "prompt_id": str(item["item_id"]),
                "category": str(item.get("category", "")),
                "group": str(item.get("group", "")),
                "question": str(item["question"]),
                "statement_truth": true_answer,
                "correct_answer": correct_answer,
                "truth_direction": 1 if true_answer else -1,
                "prompt_key": prompt_key,
                "prompt": render_prompt(
                    prompt_key,
                    model_key=model_key,
                    question=item["question"],
                ),
            }
        )
    return rows


def build_stance_prompts(
    items: list[dict[str, Any]],
    prompt_key: str,
    *,
    model_key: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        if not {"item_id", "sentence"}.issubset(item):
            raise ValueError(f"Incomplete stance item: {item}")
        rows.append(
            {
                "task": "stance_taking",
                "item_id": str(item["item_id"]),
                "prompt_id": str(item["item_id"]),
                "subject": str(item.get("subject", "")),
                "group": str(item.get("group", "")),
                "statement": str(item["sentence"]),
                "statement_chinese": str(item.get("sentence_chinese", "")),
                "prompt_key": prompt_key,
                "prompt": render_prompt(
                    prompt_key,
                    model_key=model_key,
                    statement=item["sentence"],
                ),
            }
        )
    return rows


def parse_object_response(text: str) -> dict[str, Any]:
    mapping = extract_outermost_mapping(text)
    think = mapping.get("think") if mapping is not None else None
    answer = mapping.get("answer") if mapping is not None else None
    return {
        "json_object_found": mapping is not None,
        "strict_json_valid": bool(
            mapping is not None
            and set(mapping) == {"think", "answer"}
            and isinstance(think, str)
            and isinstance(answer, str)
        ),
        "think": think.strip() if isinstance(think, str) else None,
        "answer": answer.strip() if isinstance(answer, str) else None,
    }


def parse_task_response(
    task_key: str,
    text: str,
    correct_answer: str | None,
) -> dict[str, Any]:
    if task_key == "alphabetical_order":
        return parse_json_answer(
            text,
            allowed_answers={"right", "wrong"},
            correct_answer=correct_answer,
        )
    if task_key == "factual_judgment":
        return parse_json_answer(
            text,
            allowed_answers={"yes", "no"},
            correct_answer=correct_answer,
        )
    if task_key == "stance_taking":
        return parse_object_response(text)
    raise ValueError(f"Unknown generative task: {task_key}")
