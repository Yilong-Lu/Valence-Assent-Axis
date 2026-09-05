"""Run the Subjective Preference lexical and option-order controls."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from vaa.activations import capture_assistant_start_activations, project_onto_vaa
from vaa.config import REPOSITORY_ROOT
from vaa.io import write_json_atomic, write_jsonl_atomic
from vaa.models import load_model_bundle
from vaa.preference import (
    build_ordered_prompts,
    decompose_initial_state,
    decompose_order_effects,
    limit_pairs_per_class,
)
from vaa.preference_config import load_preference_config, load_preference_pairs
from vaa.prompts import get_prompt_spec
from vaa.scoring import (
    candidate_token_metadata,
    render_chat_prompt,
    score_candidate_pair,
    score_first_token_candidates,
)
from vaa.steering import normalized_alpha_to_raw


def build_tokenization_audit(
    pairs: list[dict[str, str]],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    rows = []
    for pair in pairs:
        metadata_a = candidate_token_metadata(tokenizer, pair["word_A"])
        metadata_b = candidate_token_metadata(tokenizer, pair["word_B"])
        token_id_a = metadata_a["token_ids"][0] if metadata_a["token_ids"] else None
        token_id_b = metadata_b["token_ids"][0] if metadata_b["token_ids"] else None
        if token_id_a is None or token_id_b is None:
            raise ValueError(f"Empty candidate tokenization for {pair['pair_id']}")
        rows.append(
            {
                **pair,
                "candidate_A": metadata_a,
                "candidate_B": metadata_b,
                "same_first_token": token_id_a == token_id_b,
            }
        )
    return rows


def resolve_alpha_grid(
    configured: tuple[float, ...],
    override: list[float] | None,
) -> tuple[float, ...]:
    values = configured if override is None else tuple(float(value) for value in override)
    if not values or len(set(values)) != len(values):
        raise ValueError("alpha values must be nonempty and unique")
    if any(value < -1 or value > 1 for value in values):
        raise ValueError("alpha values must lie within [-1, 1]")
    return tuple(sorted(values))


def score_prompts(
    bundle,
    prompt_rows: list[dict[str, Any]],
    alpha_grid: tuple[float, ...],
    *,
    progress_every: int,
) -> list[dict[str, Any]]:
    raw_rows = []
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
                prompt_row["word_A"],
                prompt_row["word_B"],
                target_layer=bundle.spec.target_layer,
                steering_vector=bundle.vaa_vector,
                alpha_raw=alpha_raw,
            )
            rendered_prompt = render_chat_prompt(bundle.tokenizer, prompt_row["prompt"])
            first_token_scores = score_first_token_candidates(
                bundle.model,
                bundle.tokenizer,
                rendered_prompt,
                (prompt_row["word_A"], prompt_row["word_B"]),
                target_layer=bundle.spec.target_layer,
                steering_vector=bundle.vaa_vector,
                alpha_raw=alpha_raw,
            )
            first_token_margin = (
                first_token_scores["candidate_logits"][prompt_row["word_A"]]
                - first_token_scores["candidate_logits"][prompt_row["word_B"]]
            )
            raw_rows.append(
                {
                    "model_key": bundle.spec.key,
                    "target_layer": bundle.spec.target_layer,
                    **prompt_row,
                    "alpha_norm": float(alpha_norm),
                    "alpha_raw": alpha_raw,
                    "candidate_A_score": scores["candidate_a"],
                    "candidate_B_score": scores["candidate_b"],
                    "first_token_scores": first_token_scores,
                    "logprob_margin_A_minus_B": float(
                        scores["logprob_margin_a_minus_b"]
                    ),
                    "first_token_margin_A_minus_B": first_token_margin,
                }
            )
        if progress_every > 0 and prompt_index % progress_every == 0:
            print(f"Scored {prompt_index}/{len(prompt_rows)} ordered prompts")
    return raw_rows


def measure_initial_state(
    bundle,
    prompt_rows: list[dict[str, Any]],
    *,
    batch_size: int,
    max_input_tokens: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    activations = capture_assistant_start_activations(
        bundle.model,
        bundle.tokenizer,
        [row["prompt"] for row in prompt_rows],
        [bundle.spec.target_layer],
        batch_size=batch_size,
        max_input_tokens=max_input_tokens,
    )[bundle.spec.target_layer]
    projections = project_onto_vaa(activations, bundle.vaa_vector)
    rows = []
    for index, prompt_row in enumerate(prompt_rows):
        rows.append(
            {
                "model_key": bundle.spec.key,
                "target_layer": bundle.spec.target_layer,
                **prompt_row,
                "alpha_norm": 0.0,
                "alpha_raw": 0.0,
                "projection_raw": float(projections["projection_raw"][index]),
                "projection_unit": float(projections["projection_unit"][index]),
                "projection_cosine": float(
                    projections["projection_cosine"][index]
                ),
                "activation_norm": float(projections["activation_norm"][index]),
            }
        )
    return rows, activations


def class_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["pair_class"] for row in rows).items()))


def run(args: argparse.Namespace) -> None:
    config = load_preference_config()
    all_pairs = load_preference_pairs(config.stimulus_file)
    pairs = limit_pairs_per_class(all_pairs, args.max_pairs_per_class)
    alpha_grid = resolve_alpha_grid(config.normalized_alpha_grid, args.alpha_values)
    if not args.sequence_scores and not args.initial_state:
        raise ValueError("At least one of sequence_scores or initial_state is required")
    if args.dry_run:
        print(
            {
                "model_key": args.model,
                "task": "subjective_preference",
                "prompt_key": config.prompt_key,
                "n_pairs_configured": len(all_pairs),
                "n_pairs_selected_before_token_audit": len(pairs),
                "pair_class_counts": class_counts(pairs),
                "n_ordered_prompts_before_token_audit": 2 * len(pairs),
                "normalized_alpha_grid": alpha_grid,
                "sequence_scores": args.sequence_scores,
                "measure_initial_state": args.initial_state,
            }
        )
        return

    bundle = load_model_bundle(
        args.model,
        device_map=args.device_map,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    tokenization_audit = build_tokenization_audit(pairs, bundle.tokenizer)
    excluded_ids = {
        row["pair_id"]
        for row in tokenization_audit
        if config.exclude_same_first_token and row["same_first_token"]
    }
    usable_pairs = [row for row in pairs if row["pair_id"] not in excluded_ids]
    prompt_rows = build_ordered_prompts(
        usable_pairs,
        config.prompt_key,
        model_key=bundle.spec.key,
    )
    output_dir = args.output_dir or (
        REPOSITORY_ROOT
        / "results"
        / "generated"
        / "subjective_preference"
        / args.model
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_dir / "tokenization_audit.jsonl", tokenization_audit)

    initial_prompt_rows = None
    initial_activations = None
    if args.initial_state:
        initial_prompt_rows, initial_activations = measure_initial_state(
            bundle,
            prompt_rows,
            batch_size=args.activation_batch_size,
            max_input_tokens=args.max_input_tokens,
        )
        write_jsonl_atomic(
            output_dir / "initial_state_prompts.jsonl",
            initial_prompt_rows,
        )
        write_jsonl_atomic(
            output_dir / "initial_state_pairs.jsonl",
            decompose_initial_state(initial_prompt_rows),
        )
        if args.save_activations:
            np.save(
                output_dir / "initial_state_activations.npy",
                initial_activations,
                allow_pickle=False,
            )

    raw_rows = []
    component_rows = []
    if args.sequence_scores:
        raw_rows = score_prompts(
            bundle,
            prompt_rows,
            alpha_grid,
            progress_every=args.progress_every,
        )
        component_rows = decompose_order_effects(raw_rows)
        write_jsonl_atomic(output_dir / "sequence_scores.jsonl", raw_rows)
        write_jsonl_atomic(output_dir / "order_components.jsonl", component_rows)

    prompt_spec = get_prompt_spec(config.prompt_key)
    metadata = {
        "schema_version": 1,
        "model_key": bundle.spec.key,
        "task": "subjective_preference",
        "display_name": config.display_name,
        "prompt_key": config.prompt_key,
        "prompt_template": prompt_spec.template_for_model(bundle.spec.key),
        "target_layer": bundle.spec.target_layer,
        "normalized_alpha_grid": list(alpha_grid),
        "raw_alpha_range": list(bundle.spec.raw_alpha_range),
        "neutral_semantic_orientation": config.neutral_orientation,
        "primary_score": "full_sequence_candidate_log_probability",
        "first_token_score_role": "robustness",
        "sequence_scores_measured": args.sequence_scores,
        "n_pairs_configured": len(all_pairs),
        "n_pairs_selected_before_token_audit": len(pairs),
        "n_pairs_usable": len(usable_pairs),
        "pair_class_counts_usable": class_counts(usable_pairs),
        "excluded_same_first_token_pair_ids": sorted(excluded_ids),
        "n_ordered_prompts": len(prompt_rows),
        "n_sequence_score_rows": len(raw_rows),
        "n_order_component_rows": len(component_rows),
        "initial_state_measured": args.initial_state,
        "n_initial_state_prompt_rows": (
            len(initial_prompt_rows) if initial_prompt_rows is not None else 0
        ),
    }
    write_json_atomic(output_dir / "metadata.json", metadata)
    print(f"Wrote Subjective Preference outputs to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen25_14b")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--alpha-values", nargs="+", type=float)
    parser.add_argument("--max-pairs-per-class", type=int)
    parser.add_argument(
        "--sequence-scores",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--initial-state",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--save-activations", action="store_true")
    parser.add_argument("--activation-batch-size", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
