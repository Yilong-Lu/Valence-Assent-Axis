"""Pure stimulus and AB/BA decomposition helpers for word preferences."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from .prompts import render_prompt


def limit_pairs_per_class(
    pairs: list[dict[str, str]],
    maximum: int | None,
) -> list[dict[str, str]]:
    if maximum is None:
        return list(pairs)
    if maximum <= 0:
        raise ValueError("maximum pairs per class must be positive")
    counts: dict[str, int] = defaultdict(int)
    selected = []
    for pair in pairs:
        pair_class = pair["pair_class"]
        if counts[pair_class] < maximum:
            selected.append(pair)
            counts[pair_class] += 1
    return selected


def build_ordered_prompts(
    pairs: list[dict[str, str]],
    prompt_key: str,
    *,
    model_key: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for pair in pairs:
        for order, option1, option2, a_first in (
            ("AB", pair["word_A"], pair["word_B"], True),
            ("BA", pair["word_B"], pair["word_A"], False),
        ):
            rows.append(
                {
                    **pair,
                    "order": order,
                    "A_first": a_first,
                    "option1": option1,
                    "option2": option2,
                    "prompt": render_prompt(
                        prompt_key,
                        model_key=model_key,
                        option1=option1,
                        option2=option2,
                    ),
                }
            )
    return rows


def semantic_orientation_sign(row: dict[str, Any]) -> int:
    """Orient neutral pairs as alphabetically earlier minus later."""
    if row["valence_status"] == "valenced":
        return 1
    word_a = str(row["word_A"]).casefold()
    word_b = str(row["word_B"]).casefold()
    return 1 if word_a <= word_b else -1


def _group_by_pair_and_alpha(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, float], dict[str, dict[str, Any]]]:
    grouped: dict[tuple[str, float], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (str(row["pair_id"]), float(row["alpha_norm"]))
        order = str(row["order"])
        if order in grouped[key]:
            raise ValueError(f"Duplicate preference row for {key} and order {order}")
        grouped[key][order] = row
    return grouped


def decompose_order_effects(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for (pair_id, alpha_norm), by_order in _group_by_pair_and_alpha(rows).items():
        if set(by_order) != {"AB", "BA"}:
            raise ValueError(f"Incomplete AB/BA rows for {pair_id} at {alpha_norm}")
        ab = by_order["AB"]
        ba = by_order["BA"]
        d_ab = float(ab["logprob_margin_A_minus_B"])
        d_ba = float(ba["logprob_margin_A_minus_B"])
        first_ab = float(ab["first_token_margin_A_minus_B"])
        first_ba = float(ba["first_token_margin_A_minus_B"])
        sign = semantic_orientation_sign(ab)
        semantic_source = (d_ab + d_ba) / 2
        first_semantic_source = (first_ab + first_ba) / 2
        output.append(
            {
                "model_key": ab["model_key"],
                "target_layer": ab["target_layer"],
                "pair_id": pair_id,
                "pair_class": ab["pair_class"],
                "domain": ab["domain"],
                "valence_status": ab["valence_status"],
                "opposition_status": ab["opposition_status"],
                "word_A": ab["word_A"],
                "word_B": ab["word_B"],
                "semantic_orientation_sign": sign,
                "alpha_norm": alpha_norm,
                "alpha_raw": float(ab["alpha_raw"]),
                "d_AB": d_ab,
                "d_BA": d_ba,
                "semantic_component_source_order": semantic_source,
                "semantic_component": sign * semantic_source,
                "position_component": (d_ab - d_ba) / 2,
                "first_token_d_AB": first_ab,
                "first_token_d_BA": first_ba,
                "first_token_semantic_component_source_order": first_semantic_source,
                "first_token_semantic_component": sign * first_semantic_source,
                "first_token_position_component": (first_ab - first_ba) / 2,
            }
        )
    return output


def decompose_initial_state(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        pair_id = str(row["pair_id"])
        order = str(row["order"])
        if order in grouped[pair_id]:
            raise ValueError(f"Duplicate initial-state row for {pair_id}/{order}")
        grouped[pair_id][order] = row

    output = []
    for pair_id, by_order in grouped.items():
        if set(by_order) != {"AB", "BA"}:
            raise ValueError(f"Incomplete initial-state AB/BA rows for {pair_id}")
        ab = by_order["AB"]
        ba = by_order["BA"]
        sign = semantic_orientation_sign(ab)
        record = {
            "model_key": ab["model_key"],
            "target_layer": ab["target_layer"],
            "pair_id": pair_id,
            "pair_class": ab["pair_class"],
            "domain": ab["domain"],
            "valence_status": ab["valence_status"],
            "opposition_status": ab["opposition_status"],
            "word_A": ab["word_A"],
            "word_B": ab["word_B"],
            "semantic_orientation_sign": sign,
            "alpha_norm": 0.0,
            "alpha_raw": 0.0,
        }
        for metric in ("projection_raw", "projection_unit", "projection_cosine"):
            value_ab = float(ab[metric])
            value_ba = float(ba[metric])
            source_delta = value_ab - value_ba
            record[f"{metric}_AB"] = value_ab
            record[f"{metric}_BA"] = value_ba
            record[f"{metric}_delta_source_order"] = source_delta
            record[f"{metric}_delta"] = sign * source_delta
        record["activation_norm_AB"] = float(ab["activation_norm"])
        record["activation_norm_BA"] = float(ba["activation_norm"])
        output.append(record)
    return output
