"""Frozen item selection and prompts for generation robustness checks."""

from __future__ import annotations

from typing import Any

from .generative_config import load_json_records
from .prompts import render_prompt
from .robustness_config import RobustnessTaskSpec


def build_robustness_items(task: RobustnessTaskSpec) -> list[dict[str, Any]]:
    source = load_json_records(task.stimulus_file)
    if len(source) != 30:
        raise ValueError(f"Expected 30 frozen items for {task.key}")
    rows = []
    if task.key == "alphabetical_order":
        for index, item in enumerate(source):
            earlier, later = sorted(
                (str(item["word_a"]), str(item["word_b"])),
                key=str.casefold,
            )
            option1, option2 = (
                (earlier, later) if index % 2 == 0 else (later, earlier)
            )
            statement_truth = index % 2 == 0
            rows.append(
                {
                    "task": task.key,
                    "item_id": str(item["item_id"]),
                    "difficulty": str(item["difficulty"]),
                    "option1": option1,
                    "option2": option2,
                    "statement_truth": statement_truth,
                    "truth_direction": 1 if statement_truth else -1,
                    "correct_answer": "right" if statement_truth else "wrong",
                }
            )
        if sum(row["statement_truth"] for row in rows) != 15:
            raise ValueError("Alphabetical robustness items must be truth-balanced")
        return rows

    if task.key == "factual_judgment":
        for item in source:
            statement_truth = item["true_answer"]
            if not isinstance(statement_truth, bool):
                raise ValueError(f"Invalid truth label for {item['item_id']}")
            rows.append(
                {
                    "task": task.key,
                    "item_id": str(item["item_id"]),
                    "category": str(item.get("category", "")),
                    "group": str(item.get("group", "")),
                    "question": str(item["question"]),
                    "statement_truth": statement_truth,
                    "truth_direction": 1 if statement_truth else -1,
                    "correct_answer": "yes" if statement_truth else "no",
                }
            )
        return rows
    raise ValueError(f"Unknown robustness task: {task.key}")


def build_robustness_prompts(
    task: RobustnessTaskSpec,
    items: list[dict[str, Any]],
    prompt_version: str,
    *,
    model_key: str,
) -> list[dict[str, Any]]:
    try:
        prompt_key = task.prompts[prompt_version]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt version: {prompt_version}") from exc
    rows = []
    for item in items:
        format_values = (
            {"option1": item["option1"], "option2": item["option2"]}
            if task.key == "alphabetical_order"
            else {"question": item["question"]}
        )
        rows.append(
            {
                **item,
                "prompt_version": prompt_version,
                "prompt_key": prompt_key,
                "prompt": render_prompt(
                    prompt_key,
                    model_key=model_key,
                    **format_values,
                ),
            }
        )
    return rows
