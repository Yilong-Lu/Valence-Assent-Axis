"""Shared runner for the manuscript's open-ended generation tasks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from vaa.config import REPOSITORY_ROOT
from vaa.generation import generate_text_batch, set_generation_seed
from vaa.generative_config import (
    GenerativeTaskSpec,
    load_generative_config,
    load_json_records,
)
from vaa.generative_tasks import (
    build_alphabetical_prompts,
    build_factual_prompts,
    build_stance_prompts,
    parse_task_response,
)
from vaa.io import write_json_atomic, write_jsonl_atomic
from vaa.models import load_model_bundle
from vaa.prompts import get_prompt_spec
from vaa.steering import normalized_alpha_to_raw


def batches(values: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def resolve_alpha_grid(
    configured: tuple[float, ...],
    override: list[float] | None,
) -> tuple[float, ...]:
    values = configured if override is None else tuple(float(value) for value in override)
    if not values or len(values) != len(set(values)):
        raise ValueError("alpha values must be nonempty and unique")
    if any(value < -1 or value > 1 for value in values):
        raise ValueError("alpha values must lie within [-1, 1]")
    return tuple(sorted(values))


def select_conditions(
    task: GenerativeTaskSpec,
    requested: list[str] | None,
) -> dict[str, str]:
    if not task.conditions:
        if requested:
            raise ValueError("conditions are available only for Alphabetical Order")
        return {}
    selected = list(task.conditions) if requested is None else requested
    unknown = set(selected) - set(task.conditions)
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")
    return {condition: task.conditions[condition] for condition in selected}


def build_prompt_rows(
    task: GenerativeTaskSpec,
    items: list[dict[str, Any]],
    *,
    model_key: str,
    conditions: dict[str, str],
) -> list[dict[str, Any]]:
    if task.key == "alphabetical_order":
        return build_alphabetical_prompts(items, conditions, model_key=model_key)
    if task.key == "factual_judgment":
        assert task.prompt_key is not None
        return build_factual_prompts(items, task.prompt_key, model_key=model_key)
    if task.key == "stance_taking":
        assert task.prompt_key is not None
        return build_stance_prompts(items, task.prompt_key, model_key=model_key)
    raise ValueError(f"Unknown generative task: {task.key}")


def run_generation(
    bundle,
    task: GenerativeTaskSpec,
    prompt_rows: list[dict[str, Any]],
    alpha_grid: tuple[float, ...],
    *,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
) -> list[dict[str, Any]]:
    set_generation_seed(seed)
    output_rows = []
    for alpha_norm in alpha_grid:
        alpha_raw = round(
            normalized_alpha_to_raw(alpha_norm, bundle.spec.raw_alpha_range),
            6,
        )
        completed = 0
        for prompt_batch in batches(prompt_rows, batch_size):
            generations = generate_text_batch(
                bundle.model,
                bundle.tokenizer,
                [row["prompt"] for row in prompt_batch],
                target_layer=bundle.spec.target_layer,
                steering_vector=bundle.vaa_vector,
                alpha_raw=alpha_raw,
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            for prompt_row, generation in zip(prompt_batch, generations):
                parsed = parse_task_response(
                    task.key,
                    generation["generated_text"],
                    prompt_row.get("correct_answer"),
                )
                output = {
                    "model_key": bundle.spec.key,
                    "target_layer": bundle.spec.target_layer,
                    **prompt_row,
                    "alpha_norm": float(alpha_norm),
                    "alpha_raw": alpha_raw,
                    "seed": seed,
                    **generation,
                    **parsed,
                }
                if "truth_direction" in prompt_row:
                    direction = int(prompt_row["truth_direction"])
                    output["alignment_pressure_norm"] = float(alpha_norm * direction)
                    output["alignment_pressure_raw"] = float(alpha_raw * direction)
                output_rows.append(output)
            completed += len(prompt_batch)
        print(
            f"{task.display_name}: alpha={alpha_norm:g}, "
            f"generated {completed}/{len(prompt_rows)} prompts"
        )
    return output_rows


def prompt_metadata(
    task: GenerativeTaskSpec,
    model_key: str,
    conditions: dict[str, str],
) -> dict[str, Any]:
    if task.conditions:
        return {
            condition: {
                "prompt_key": prompt_key,
                "template": get_prompt_spec(prompt_key).template_for_model(model_key),
            }
            for condition, prompt_key in conditions.items()
        }
    assert task.prompt_key is not None
    return {
        "prompt_key": task.prompt_key,
        "template": get_prompt_spec(task.prompt_key).template_for_model(model_key),
    }


def run_task(task_key: str, args: argparse.Namespace) -> None:
    config = load_generative_config()
    task = config.tasks[task_key]
    all_items = load_json_records(task.stimulus_file)
    if args.max_items is not None and args.max_items <= 0:
        raise ValueError("max-items must be positive")
    items = all_items if args.max_items is None else all_items[: args.max_items]
    conditions = select_conditions(task, args.conditions)
    alpha_grid = resolve_alpha_grid(config.normalized_alpha_grid, args.alpha_values)
    prompt_rows = build_prompt_rows(
        task,
        items,
        model_key=args.model,
        conditions=conditions,
    )
    generation = config.generation
    batch_size = args.batch_size or generation.batch_size
    max_new_tokens = args.max_new_tokens or generation.max_new_tokens
    seed = generation.seed if args.seed is None else args.seed
    temperature = generation.temperature if args.temperature is None else args.temperature
    if batch_size <= 0 or max_new_tokens <= 0 or temperature <= 0:
        raise ValueError("batch size, token limit, and temperature must be positive")

    if args.dry_run:
        print(
            {
                "model_key": args.model,
                "task": task.key,
                "display_name": task.display_name,
                "n_items_configured": len(all_items),
                "n_items_selected": len(items),
                "n_prompt_rows": len(prompt_rows),
                "normalized_alpha_grid": list(alpha_grid),
                "n_generation_rows": len(prompt_rows) * len(alpha_grid),
                "conditions": list(conditions),
                "generation": {
                    "batch_size": batch_size,
                    "max_input_tokens": generation.max_input_tokens,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": generation.do_sample,
                    "temperature": temperature,
                    "top_p": generation.top_p,
                    "top_k": generation.top_k,
                    "seed": seed,
                },
            }
        )
        return

    bundle = load_model_bundle(
        args.model,
        device_map=args.device_map,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    prompt_rows = build_prompt_rows(
        task,
        items,
        model_key=bundle.spec.key,
        conditions=conditions,
    )
    rows = run_generation(
        bundle,
        task,
        prompt_rows,
        alpha_grid,
        batch_size=batch_size,
        max_input_tokens=generation.max_input_tokens,
        max_new_tokens=max_new_tokens,
        do_sample=generation.do_sample,
        temperature=temperature,
        top_p=generation.top_p,
        top_k=generation.top_k,
        seed=seed,
    )
    output_dir = args.output_dir or (
        REPOSITORY_ROOT / "results" / "generated" / task.key / args.model
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_dir / "generations.jsonl", rows)
    write_json_atomic(
        output_dir / "metadata.json",
        {
            "schema_version": 1,
            "model_key": bundle.spec.key,
            "task": task.key,
            "display_name": task.display_name,
            "target_layer": bundle.spec.target_layer,
            "raw_alpha_range": list(bundle.spec.raw_alpha_range),
            "normalized_alpha_grid": list(alpha_grid),
            "n_items_configured": len(all_items),
            "n_items_selected": len(items),
            "n_prompt_rows": len(prompt_rows),
            "n_generation_rows": len(rows),
            "conditions": list(conditions),
            "prompts": prompt_metadata(task, bundle.spec.key, conditions),
            "generation": {
                "batch_size": batch_size,
                "max_input_tokens": generation.max_input_tokens,
                "max_new_tokens": max_new_tokens,
                "do_sample": generation.do_sample,
                "temperature": temperature,
                "top_p": generation.top_p,
                "top_k": generation.top_k,
                "seed": seed,
            },
            "response_parsing": (
                "strict_json_and_unambiguous_answer_field"
                if task.answer_labels
                else "strict_json_object"
            ),
        },
    )
    print(f"Wrote {len(rows)} rows to {output_dir}")


def parse_args(task_key: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Run the {task_key.replace('_', ' ')} experiment."
    )
    parser.add_argument("--model", default="qwen25_14b")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--alpha-values", nargs="+", type=float)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()
