"""Prompt construction and state summaries for Feedback-Induced Sycophancy."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .prompts import render_feedback_prompt


def build_sycophancy_prompts(
    items: list[dict[str, Any]],
    conditions: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        for condition, feedback_key in conditions.items():
            rows.append(
                {
                    "task": "feedback_induced_sycophancy",
                    "item_id": str(item["item_id"]),
                    "prompt_id": f"{item['item_id']}::{condition}",
                    "condition": condition,
                    "argument": str(item["argument"]),
                    "logical_error": str(item["logical_error"]),
                    "rating": float(item["rating"]),
                    "in_intervention_subset": bool(
                        item["in_intervention_subset"]
                    ),
                    "intervention_split": str(item["intervention_split"]),
                    "prompt_key": "feedback_induced_sycophancy_v1",
                    "prompt": render_feedback_prompt(
                        str(item["argument"]),
                        feedback_key,
                    ),
                }
            )
    return rows


def add_baseline_state_zscores(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    output = [dict(row) for row in rows]
    baseline_values = np.asarray(
        [
            float(row["pre_addition_vaa_projection_unit"])
            for row in output
            if row["condition"] == "no_feedback"
            and float(row["alpha_norm"]) == 0.0
        ],
        dtype=float,
    )
    if not len(baseline_values):
        raise ValueError("No-feedback alpha-zero states are required")
    mean = float(baseline_values.mean())
    sd = float(baseline_values.std(ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(f"Invalid no-feedback state SD: {sd}")
    for row in output:
        row["pre_addition_vaa_projection_unit_z_baseline"] = (
            float(row["pre_addition_vaa_projection_unit"]) - mean
        ) / sd
        row["post_addition_vaa_projection_unit_z_baseline"] = (
            float(row["post_addition_vaa_projection_unit"]) - mean
        ) / sd
    return output, {
        "condition": "no_feedback",
        "alpha_norm": 0.0,
        "mean": mean,
        "population_sd": sd,
        "n": len(baseline_values),
    }
