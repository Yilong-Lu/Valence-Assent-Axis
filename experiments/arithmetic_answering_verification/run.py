"""Run matched arithmetic answering and verification under VAA intervention."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from vaa.arithmetic import (
    build_arithmetic_prompts,
    calculate_arithmetic_metrics,
    estimate_arithmetic_item_slopes,
    summarize_arithmetic_by_alpha,
)
from vaa.arithmetic_config import load_arithmetic_config, load_arithmetic_items
from vaa.config import REPOSITORY_ROOT
from vaa.io import write_json_atomic, write_jsonl_atomic
from vaa.models import load_model_bundle
from vaa.prompts import get_prompt_spec
from vaa.scoring import candidate_token_metadata, score_candidate_pair
from vaa.steering import normalized_alpha_to_raw


def resolve_alpha_grid(
    configured: tuple[float, ...],
    override: list[float] | None,
) -> tuple[float, ...]:
    values = (
        configured
        if override is None
        else tuple(float(value) for value in override)
    )
    if not values or len(set(values)) != len(values):
        raise ValueError("alpha values must be nonempty and unique")
    if any(value < -1 or value > 1 for value in values):
        raise ValueError("alpha values must lie within [-1, 1]")
    return tuple(sorted(values))


def build_tokenization_audit(
    prompt_rows: list[dict[str, Any]],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    rows = []
    for prompt_row in prompt_rows:
        metadata_a = candidate_token_metadata(
            tokenizer,
            prompt_row["candidate_A"],
        )
        metadata_b = candidate_token_metadata(
            tokenizer,
            prompt_row["candidate_B"],
        )
        if not metadata_a["token_ids"] or not metadata_b["token_ids"]:
            raise ValueError(f"Empty candidate tokenization for {prompt_row['prompt_id']}")
        rows.append(
            {
                "item_id": prompt_row["item_id"],
                "prompt_id": prompt_row["prompt_id"],
                "mode": prompt_row["mode"],
                "statement_truth": prompt_row["statement_truth"],
                "candidate_A": prompt_row["candidate_A"],
                "candidate_B": prompt_row["candidate_B"],
                "candidate_A_metadata": metadata_a,
                "candidate_B_metadata": metadata_b,
                "same_first_token": (
                    metadata_a["token_ids"][0] == metadata_b["token_ids"][0]
                ),
            }
        )
    return rows


def score_prompts(
    bundle,
    prompt_rows: list[dict[str, Any]],
    alpha_grid: tuple[float, ...],
    *,
    progress_every: int,
) -> list[dict[str, Any]]:
    rows = []
    for prompt_index, prompt_row in enumerate(prompt_rows, start=1):
        for alpha_norm in alpha_grid:
            alpha_raw = round(
                normalized_alpha_to_raw(alpha_norm, bundle.spec.raw_alpha_range),
                6,
            )
            scores = score_candidate_pair(
                bundle.model,
                bundle.tokenizer,
                prompt_row["prompt"],
                prompt_row["candidate_A"],
                prompt_row["candidate_B"],
                target_layer=bundle.spec.target_layer,
                steering_vector=bundle.vaa_vector,
                alpha_raw=alpha_raw,
            )
            margin = float(scores["logprob_margin_a_minus_b"])
            metrics = calculate_arithmetic_metrics(
                prompt_row["mode"],
                prompt_row["statement_truth"],
                margin,
            )
            rows.append(
                {
                    "model_key": bundle.spec.key,
                    "target_layer": bundle.spec.target_layer,
                    **prompt_row,
                    "alpha_norm": float(alpha_norm),
                    "alpha_raw": alpha_raw,
                    "candidate_A_score": scores["candidate_a"],
                    "candidate_B_score": scores["candidate_b"],
                    "logprob_margin_A_minus_B": margin,
                    **metrics,
                }
            )
        if progress_every > 0 and prompt_index % progress_every == 0:
            print(f"Scored {prompt_index}/{len(prompt_rows)} arithmetic prompts")
    return rows


def run(args: argparse.Namespace) -> None:
    config = load_arithmetic_config()
    all_items = load_arithmetic_items(config.stimulus_file)
    if args.max_items is not None and args.max_items <= 0:
        raise ValueError("max-items must be positive")
    items = all_items if args.max_items is None else all_items[: args.max_items]
    alpha_grid = resolve_alpha_grid(config.normalized_alpha_grid, args.alpha_values)
    prompt_rows = build_arithmetic_prompts(
        items,
        config.direct_prompt_key,
        config.verification_prompt_key,
        model_key=args.model,
    )
    if args.dry_run:
        print(
            {
                "model_key": args.model,
                "task": "arithmetic_answering_verification",
                "n_items_configured": len(all_items),
                "n_items_selected": len(items),
                "n_prompt_rows": len(prompt_rows),
                "normalized_alpha_grid": alpha_grid,
                "n_sequence_score_rows": len(prompt_rows) * len(alpha_grid),
                "primary_outcome": config.primary_outcome,
                "candidate_scoring": config.candidate_scoring,
            }
        )
        return

    bundle = load_model_bundle(
        args.model,
        device_map=args.device_map,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    prompt_rows = build_arithmetic_prompts(
        items,
        config.direct_prompt_key,
        config.verification_prompt_key,
        model_key=bundle.spec.key,
    )
    tokenization_audit = build_tokenization_audit(prompt_rows, bundle.tokenizer)
    raw_rows = score_prompts(
        bundle,
        prompt_rows,
        alpha_grid,
        progress_every=args.progress_every,
    )
    slope_rows = estimate_arithmetic_item_slopes(raw_rows)
    summary_rows = summarize_arithmetic_by_alpha(raw_rows)

    output_dir = args.output_dir or (
        REPOSITORY_ROOT
        / "results"
        / "generated"
        / "arithmetic_answering_verification"
        / args.model
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_dir / "sequence_scores.jsonl", raw_rows)
    write_jsonl_atomic(output_dir / "item_slopes.jsonl", slope_rows)
    write_jsonl_atomic(output_dir / "alpha_summary.jsonl", summary_rows)
    write_jsonl_atomic(output_dir / "tokenization_audit.jsonl", tokenization_audit)

    direct_prompt = get_prompt_spec(config.direct_prompt_key)
    verification_prompt = get_prompt_spec(config.verification_prompt_key)
    metadata = {
        "schema_version": 1,
        "model_key": bundle.spec.key,
        "task": "arithmetic_answering_verification",
        "display_name": config.display_name,
        "prompt_keys": {
            "direct_numeric": config.direct_prompt_key,
            "verification": config.verification_prompt_key,
        },
        "prompt_templates": {
            "direct_numeric": direct_prompt.template_for_model(bundle.spec.key),
            "verification": verification_prompt.template_for_model(bundle.spec.key),
        },
        "target_layer": bundle.spec.target_layer,
        "normalized_alpha_grid": list(alpha_grid),
        "raw_alpha_range": list(bundle.spec.raw_alpha_range),
        "primary_outcome": config.primary_outcome,
        "candidate_scoring": config.candidate_scoring,
        "continuous_margin_role": "auditable_scoring_diagnostic",
        "n_items_configured": len(all_items),
        "n_items_selected": len(items),
        "n_prompt_rows": len(prompt_rows),
        "n_sequence_score_rows": len(raw_rows),
        "n_item_slope_rows": len(slope_rows),
        "n_alpha_summary_rows": len(summary_rows),
    }
    write_json_atomic(output_dir / "metadata.json", metadata)
    print(f"Wrote arithmetic outputs to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen25_14b")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--alpha-values", nargs="+", type=float)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
