"""Pure helpers for the Arithmetic Answering and Verification task."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np

from .prompts import render_prompt


ARITHMETIC_MODES = (
    "direct_numeric",
    "verification_true",
    "verification_false",
)


def build_arithmetic_prompts(
    items: list[dict[str, Any]],
    direct_prompt_key: str,
    verification_prompt_key: str,
    *,
    model_key: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        base = {
            "item_id": str(item["item_id"]),
            "a": int(item["a"]),
            "b": int(item["b"]),
            "correct": int(item["correct"]),
            "incorrect": int(item["incorrect"]),
            "offset": int(item["offset"]),
        }
        common = {"operand1": base["a"], "operand2": base["b"]}
        rows.extend(
            [
                {
                    **base,
                    "prompt_id": f"{base['item_id']}::direct_numeric",
                    "mode": "direct_numeric",
                    "statement_truth": None,
                    "candidate_A": str(base["correct"]),
                    "candidate_B": str(base["incorrect"]),
                    "candidate_A_role": "correct_number",
                    "candidate_B_role": "incorrect_number",
                    "prompt": render_prompt(
                        direct_prompt_key,
                        model_key=model_key,
                        **common,
                    ),
                },
                {
                    **base,
                    "prompt_id": f"{base['item_id']}::verification_true",
                    "mode": "verification_true",
                    "statement_truth": True,
                    "candidate_A": "True",
                    "candidate_B": "False",
                    "candidate_A_role": "true_label",
                    "candidate_B_role": "false_label",
                    "prompt": render_prompt(
                        verification_prompt_key,
                        model_key=model_key,
                        proposed_answer=base["correct"],
                        **common,
                    ),
                },
                {
                    **base,
                    "prompt_id": f"{base['item_id']}::verification_false",
                    "mode": "verification_false",
                    "statement_truth": False,
                    "candidate_A": "True",
                    "candidate_B": "False",
                    "candidate_A_role": "true_label",
                    "candidate_B_role": "false_label",
                    "prompt": render_prompt(
                        verification_prompt_key,
                        model_key=model_key,
                        proposed_answer=base["incorrect"],
                        **common,
                    ),
                },
            ]
        )
    return rows


def calculate_arithmetic_metrics(
    mode: str,
    statement_truth: bool | None,
    logprob_margin_a_minus_b: float,
) -> dict[str, float | int | None]:
    margin = float(logprob_margin_a_minus_b)
    if mode == "direct_numeric":
        if statement_truth is not None:
            raise ValueError("Direct numeric prompts must not have a truth label")
        return {
            "primary_margin": margin,
            "numeric_correct_margin": margin,
            "assent_margin": None,
            "verification_correct_margin": None,
            "candidate_accuracy": int(margin > 0),
        }
    if mode not in {"verification_true", "verification_false"}:
        raise ValueError(f"Unsupported arithmetic mode: {mode}")
    if not isinstance(statement_truth, bool):
        raise ValueError(f"Verification mode {mode} requires a truth label")
    correctness_margin = margin if statement_truth else -margin
    return {
        "primary_margin": margin,
        "numeric_correct_margin": None,
        "assent_margin": margin,
        "verification_correct_margin": correctness_margin,
        "candidate_accuracy": int(correctness_margin > 0),
    }


def linear_slope(x: Iterable[float], y: Iterable[float]) -> float | None:
    x_array = np.asarray(list(x), dtype=float)
    y_array = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[mask]
    y_array = y_array[mask]
    if len(np.unique(x_array)) < 2:
        return None
    centered_x = x_array - x_array.mean()
    numerator = np.dot(centered_x, y_array - y_array.mean())
    denominator = np.dot(centered_x, centered_x)
    return float(numerator / denominator)


def estimate_arithmetic_item_slopes(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["item_id"]), str(row["mode"]))].append(row)

    output = []
    metrics = (
        "primary_margin",
        "numeric_correct_margin",
        "assent_margin",
        "verification_correct_margin",
        "candidate_accuracy",
    )
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda row: float(row["alpha_norm"]))
        first = group[0]
        record = {
            field: first[field]
            for field in (
                "model_key",
                "target_layer",
                "item_id",
                "prompt_id",
                "mode",
                "statement_truth",
                "a",
                "b",
                "correct",
                "incorrect",
                "offset",
            )
        }
        for metric in metrics:
            observed = [
                (float(row["alpha_norm"]), float(row[metric]))
                for row in group
                if row[metric] is not None
            ]
            if not observed:
                record[f"{metric}_slope"] = None
            else:
                record[f"{metric}_slope"] = linear_slope(
                    [value[0] for value in observed],
                    [value[1] for value in observed],
                )
        output.append(record)
    return output


def summarize_arithmetic_by_alpha(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, bool | None, float],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["mode"]),
                row["statement_truth"],
                float(row["alpha_norm"]),
            )
        ].append(row)

    output = []
    metrics = (
        "primary_margin",
        "numeric_correct_margin",
        "assent_margin",
        "verification_correct_margin",
        "candidate_accuracy",
    )
    for key in sorted(grouped, key=lambda value: (value[0], value[2])):
        group = grouped[key]
        first = group[0]
        record = {
            "model_key": first["model_key"],
            "target_layer": first["target_layer"],
            "mode": key[0],
            "statement_truth": key[1],
            "alpha_norm": key[2],
            "alpha_raw": first["alpha_raw"],
            "n_items": len(group),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group if row[metric] is not None]
            record[metric] = float(np.mean(values)) if values else None
        output.append(record)
    return output
