"""Shared runner for natural feedback effects and VAA intervention curves."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from vaa.config import REPOSITORY_ROOT
from vaa.generation import (
    generate_text_batch_with_state_capture,
    set_generation_seed,
)
from vaa.io import write_json_atomic, write_jsonl_atomic
from vaa.models import load_model_bundle
from vaa.parsing import parse_strong_weak_verdict
from vaa.prompts import get_prompt_spec, load_prompt_registry
from vaa.steering import normalized_alpha_to_raw
from vaa.sycophancy import add_baseline_state_zscores, build_sycophancy_prompts
from vaa.sycophancy_config import (
    load_sycophancy_config,
    load_sycophancy_items,
    load_sycophancy_selection,
)


PROTOCOLS = {"feedback_effect", "intervention"}


def batches(values: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def resolve_protocol_design(
    protocol: str,
    all_items: list[dict[str, Any]],
    configured_alpha_grid: tuple[float, ...],
    alpha_override: list[float] | None,
) -> tuple[list[dict[str, Any]], tuple[float, ...]]:
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown sycophancy protocol: {protocol}")
    if protocol == "feedback_effect":
        if alpha_override is not None and tuple(alpha_override) != (0.0,):
            raise ValueError("Natural feedback effects are defined only at alpha=0")
        return list(all_items), (0.0,)

    items = [row for row in all_items if row["in_intervention_subset"]]
    alpha_grid = (
        configured_alpha_grid
        if alpha_override is None
        else tuple(sorted(set(float(value) for value in alpha_override)))
    )
    if not alpha_grid or any(value < -1 or value > 1 for value in alpha_grid):
        raise ValueError("Intervention alpha values must lie in [-1, 1]")
    if 0.0 not in alpha_grid:
        raise ValueError("Intervention alpha values must include zero")
    return items, alpha_grid


def run_generation(
    bundle,
    prompt_rows: list[dict[str, Any]],
    alpha_grid: tuple[float, ...],
    *,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    seed: int,
) -> list[dict[str, Any]]:
    set_generation_seed(seed)
    rows = []
    for alpha_norm in alpha_grid:
        alpha_raw = round(
            normalized_alpha_to_raw(alpha_norm, bundle.spec.raw_alpha_range),
            6,
        )
        completed = 0
        for prompt_batch in batches(prompt_rows, batch_size):
            generated = generate_text_batch_with_state_capture(
                bundle.model,
                bundle.tokenizer,
                [row["prompt"] for row in prompt_batch],
                target_layer=bundle.spec.target_layer,
                steering_vector=bundle.vaa_vector,
                alpha_raw=alpha_raw,
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
            )
            for prompt_row, generation in zip(prompt_batch, generated):
                verdict = parse_strong_weak_verdict(generation["generated_text"])
                rows.append(
                    {
                        "model_key": bundle.spec.key,
                        "target_layer": bundle.spec.target_layer,
                        **prompt_row,
                        "activation_endpoint": "assistant_start_boundary",
                        "alpha_norm": float(alpha_norm),
                        "alpha_raw": alpha_raw,
                        "seed": seed,
                        **generation,
                        **verdict,
                    }
                )
            completed += len(prompt_batch)
        print(
            f"Feedback-Induced Sycophancy: alpha={alpha_norm:g}, "
            f"generated {completed}/{len(prompt_rows)} prompts"
        )
    return rows


def run_protocol(protocol: str, args: argparse.Namespace) -> None:
    config = load_sycophancy_config()
    if config.stimulus_file.is_file():
        all_items = load_sycophancy_items(config.stimulus_file)
    elif args.dry_run:
        all_items = load_sycophancy_selection(config.selection_manifest)["items"]
    else:
        raise FileNotFoundError(
            "The Feedback-Induced Sycophancy arguments are third-party data and "
            "are not redistributed in this repository. Prepare them with "
            "`python analysis/python/prepare_feedback_stimuli.py --upstream-file "
            "/path/to/sycophancy-eval/datasets/feedback.jsonl`."
        )
    items, alpha_grid = resolve_protocol_design(
        protocol,
        all_items,
        config.normalized_alpha_grid,
        args.alpha_values,
    )
    if args.max_items is not None:
        if args.max_items <= 0:
            raise ValueError("max-items must be positive")
        items = items[: args.max_items]
    generation = config.generation
    batch_size = args.batch_size or generation.batch_size
    max_new_tokens = args.max_new_tokens or generation.max_new_tokens
    seed = generation.seed if args.seed is None else args.seed
    if batch_size <= 0 or max_new_tokens <= 0:
        raise ValueError("batch size and token limit must be positive")

    if args.dry_run:
        n_prompt_rows = len(items) * len(config.conditions)
        print(
            {
                "model_key": args.model,
                "task": "feedback_induced_sycophancy",
                "protocol": protocol,
                "n_items_configured": len(all_items),
                "n_items_selected": len(items),
                "n_prompt_rows": n_prompt_rows,
                "normalized_alpha_grid": list(alpha_grid),
                "n_generation_rows": n_prompt_rows * len(alpha_grid),
                "conditions": list(config.conditions),
                "generation": {
                    "batch_size": batch_size,
                    "max_input_tokens": generation.max_input_tokens,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": generation.do_sample,
                    "seed": seed,
                },
                "activation_endpoint": config.activation_endpoint,
            }
        )
        return

    prompt_rows = build_sycophancy_prompts(items, config.conditions)
    bundle = load_model_bundle(
        args.model,
        device_map=args.device_map,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    rows = run_generation(
        bundle,
        prompt_rows,
        alpha_grid,
        batch_size=batch_size,
        max_input_tokens=generation.max_input_tokens,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    rows, baseline_reference = add_baseline_state_zscores(rows)
    output_dir = args.output_dir or (
        REPOSITORY_ROOT
        / "results"
        / "generated"
        / "feedback_induced_sycophancy"
        / protocol
        / args.model
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_dir / "generations.jsonl", rows)

    prompt_spec = get_prompt_spec(config.prompt_key)
    prompt_registry = load_prompt_registry()
    parse_count = sum(bool(row["verdict_valid"]) for row in rows)
    write_json_atomic(
        output_dir / "metadata.json",
        {
            "schema_version": 1,
            "model_key": bundle.spec.key,
            "task": "feedback_induced_sycophancy",
            "display_name": config.display_name,
            "protocol": protocol,
            "target_layer": bundle.spec.target_layer,
            "raw_alpha_range": list(bundle.spec.raw_alpha_range),
            "normalized_alpha_grid": list(alpha_grid),
            "n_items_configured": len(all_items),
            "n_items_selected": len(items),
            "n_prompt_rows": len(prompt_rows),
            "n_generation_rows": len(rows),
            "conditions": list(config.conditions),
            "feedback_sentences": {
                condition: prompt_registry.feedback_conditions[feedback_key]
                for condition, feedback_key in config.conditions.items()
            },
            "prompt_key": config.prompt_key,
            "prompt_template": prompt_spec.template_for_model(bundle.spec.key),
            "generation": {
                "batch_size": batch_size,
                "max_input_tokens": generation.max_input_tokens,
                "max_new_tokens": max_new_tokens,
                "do_sample": generation.do_sample,
                "seed": seed,
            },
            "activation_endpoint": config.activation_endpoint,
            "state_metrics": [
                "pre_addition_vaa_projection_unit",
                "post_addition_vaa_projection_unit",
            ],
            "baseline_state_reference": baseline_reference,
            "primary_behavioral_outcome": config.primary_behavioral_outcome,
            "verdict_parse_rate": parse_count / len(rows),
        },
    )
    print(f"Wrote {len(rows)} rows to {output_dir}")


def parse_args(protocol: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure natural feedback effects at alpha zero."
            if protocol == "feedback_effect"
            else "Run VAA intervention curves under each feedback condition."
        )
    )
    parser.add_argument("--model", default="qwen25_14b")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--alpha-values", nargs="+", type=float)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()
