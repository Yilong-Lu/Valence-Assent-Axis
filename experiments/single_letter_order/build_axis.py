"""Derive Objective Truth Axes under right/wrong and true/false labels."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from vaa.activations import capture_assistant_start_activations
from vaa.axis_control_config import (
    AnswerLabelSpec,
    load_axis_control_config,
    load_record_array,
)
from vaa.config import REPOSITORY_ROOT
from vaa.io import write_json_atomic, write_jsonl_atomic
from vaa.models import load_model_bundle
from vaa.parsing import parse_label_response
from vaa.prompts import render_prompt
from vaa.representation import compare_task_axis_to_vaa
from vaa.scoring import render_chat_prompt, score_next_token_candidates_batch


def batches(values: Sequence[Any], batch_size: int):
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def build_items(
    records: list[dict[str, str]],
    condition: AnswerLabelSpec,
    model_key: str,
) -> list[dict]:
    items = []
    for record in records:
        earlier = record["earlier_letter"]
        later = record["later_letter"]
        for order, option1, option2, is_true in (
            ("earlier_first", earlier, later, True),
            ("later_first", later, earlier, False),
        ):
            items.append(
                {
                    "item_id": f"{record['pair_id']}::{order}",
                    "pair_id": record["pair_id"],
                    "order": order,
                    "option1": option1,
                    "option2": option2,
                    "statement_true": is_true,
                    "true_answer": (
                        condition.true_label if is_true else condition.false_label
                    ),
                    "prompt": render_prompt(
                        condition.prompt_key,
                        model_key=model_key,
                        option1=option1,
                        option2=option2,
                    ),
                }
            )
    return items


def score_candidates(bundle, prompts, condition, batch_size):
    rendered = [render_chat_prompt(bundle.tokenizer, prompt) for prompt in prompts]
    rows = []
    for prompt_batch in batches(rendered, batch_size):
        rows.extend(
            score_next_token_candidates_batch(
                bundle.model,
                bundle.tokenizer,
                prompt_batch,
                (condition.true_label, condition.false_label),
                target_layer=bundle.spec.target_layer,
                steering_vector=bundle.vaa_vector,
                alpha_raw=0.0,
            )
        )
    return rows


def sample_one_token_responses(bundle, prompts, batch_size, temperature, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    responses = []
    rendered = [render_chat_prompt(bundle.tokenizer, prompt) for prompt in prompts]
    previous_padding_side = bundle.tokenizer.padding_side
    bundle.tokenizer.padding_side = "left"
    try:
        for prompt_batch in batches(rendered, batch_size):
            inputs = bundle.tokenizer(
                prompt_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=False,
            ).to(bundle.model.device)
            input_length = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                generated = bundle.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=True,
                    temperature=temperature,
                    top_p=1.0,
                    top_k=50,
                )
            responses.extend(
                bundle.tokenizer.batch_decode(
                    generated[:, input_length:],
                    skip_special_tokens=True,
                )
            )
    finally:
        bundle.tokenizer.padding_side = previous_padding_side
    return responses


def binary_auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    values = np.asarray(scores, dtype=np.float64)
    outcomes = np.asarray(labels, dtype=bool)
    positive = values[outcomes]
    negative = values[~outcomes]
    if len(positive) == 0 or len(negative) == 0:
        return None
    comparisons = positive[:, None] - negative[None, :]
    raw_auc = float(
        (np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0))
        / comparisons.size
    )
    return max(raw_auc, 1.0 - raw_auc)


def representation_summary(result) -> dict:
    return {
        "pc1_vaa_projection_correlation": result.pc1_vaa_projection_correlation,
        "absolute_pc1_vaa_projection_correlation": abs(
            result.pc1_vaa_projection_correlation
        ),
        "axis_pearson": result.axis_pearson,
        "absolute_axis_pearson": abs(result.axis_pearson),
        "axis_cosine": result.axis_cosine,
        "absolute_axis_cosine": abs(result.axis_cosine),
        "vaa_variance_ratio": result.vaa_variance_ratio,
        "pc1_explained_variance_ratio": float(result.explained_variance_ratio[0]),
        "pc1_oriented_toward_true": True,
        "pc1_sign_flipped": result.sign_flipped,
    }


def run_condition(bundle, records, condition, args, output_dir):
    items = build_items(records, condition, bundle.spec.key)
    if args.max_items is not None:
        items = items[: args.max_items]
    prompts = [item["prompt"] for item in items]
    layer = bundle.spec.target_layer
    activations = capture_assistant_start_activations(
        bundle.model,
        bundle.tokenizer,
        prompts,
        [layer],
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
    )[layer]
    truth = np.asarray([float(item["statement_true"]) for item in items])
    result = compare_task_axis_to_vaa(
        activations,
        bundle.vaa_vector.numpy(),
        orient_toward=truth,
    )
    candidate_scores = score_candidates(bundle, prompts, condition, args.batch_size)

    sampled_responses = None
    if args.sample_responses:
        sampled_responses = sample_one_token_responses(
            bundle,
            prompts,
            args.generation_batch_size,
            args.temperature,
            args.seed,
        )

    rows = []
    for index, (item, score) in enumerate(zip(items, candidate_scores)):
        probabilities = score["candidate_probabilities"]
        candidate_answer = max(probabilities, key=probabilities.__getitem__)
        sampled_response = (
            sampled_responses[index] if sampled_responses is not None else None
        )
        parsed_response = (
            parse_label_response(
                sampled_response,
                condition.true_label,
                condition.false_label,
            )
            if sampled_response is not None
            else None
        )
        rows.append(
            {
                "model_key": bundle.spec.key,
                "task": "single_letter_order",
                "answer_label_condition": condition.key,
                "prompt_protocol": condition.protocol,
                "prompt_key": condition.prompt_key,
                "target_layer": layer,
                **item,
                "pc1_score": float(result.pc1_scores[index]),
                "pc2_score": float(result.pc2_scores[index]),
                "vaa_projection": float(result.vaa_projections[index]),
                **score,
                "candidate_answer": candidate_answer,
                "candidate_correct": candidate_answer == item["true_answer"],
                "sampled_response": sampled_response,
                "parsed_sampled_response": parsed_response,
                "sampled_response_valid": (
                    parsed_response != "other" if parsed_response is not None else None
                ),
                "sampled_response_correct": (
                    parsed_response == item["true_answer"]
                    if parsed_response is not None
                    else None
                ),
            }
        )

    valid_sampled = [
        row for row in rows if row["sampled_response_valid"] is True
    ]
    model_answer_auc = (
        binary_auc(
            [row["pc1_score"] for row in valid_sampled],
            [
                row["parsed_sampled_response"] == condition.true_label
                for row in valid_sampled
            ],
        )
        if valid_sampled
        else None
    )
    summary = {
        "schema_version": 1,
        "model_key": bundle.spec.key,
        "task": "single_letter_order",
        "answer_label_condition": condition.key,
        "prompt_protocol": condition.protocol,
        "prompt_key": condition.prompt_key,
        "true_label": condition.true_label,
        "false_label": condition.false_label,
        "target_layer": layer,
        "n_letter_pairs": len(records),
        "n_items": len(items),
        "n_true_statements": int(truth.sum()),
        "n_false_statements": int(len(truth) - truth.sum()),
        "representation": representation_summary(result),
        "behavior": {
            "candidate_accuracy": float(
                np.mean([row["candidate_correct"] for row in rows])
            ),
            "sampled_response_parse_rate": (
                float(len(valid_sampled) / len(rows))
                if sampled_responses is not None
                else None
            ),
            "sampled_response_accuracy": (
                float(
                    np.mean([row["sampled_response_correct"] for row in rows])
                )
                if sampled_responses is not None
                else None
            ),
            "model_answer_auc_from_pc1": model_answer_auc,
            "truth_auc_from_pc1": binary_auc(result.pc1_scores, truth.astype(bool)),
            "sample_temperature": args.temperature if args.sample_responses else None,
            "sample_seed": args.seed if args.sample_responses else None,
        },
        "singular_values": result.singular_values.tolist(),
        "explained_variance_ratio": result.explained_variance_ratio.tolist(),
    }

    condition_dir = output_dir / condition.key
    condition_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        condition_dir / "objective_truth_axis.npy",
        result.axis_vector,
        allow_pickle=False,
    )
    write_jsonl_atomic(condition_dir / "items.jsonl", rows)
    write_json_atomic(condition_dir / "summary.json", summary)
    if args.save_activations:
        np.save(
            condition_dir / "assistant_start_activations.npy",
            activations,
            allow_pickle=False,
        )
    print(f"{condition.key}: wrote {len(rows)} rows to {condition_dir}")
    return summary


def run(args: argparse.Namespace) -> None:
    if args.sample_responses and args.temperature <= 0:
        raise ValueError("temperature must be positive when sampling responses")
    config = load_axis_control_config().single_letter_order
    records = load_record_array(config.stimulus_file)
    condition_keys = args.answer_labels or list(config.answer_label_conditions)
    unknown = set(condition_keys) - set(config.answer_label_conditions)
    if unknown:
        raise ValueError(f"Unknown answer-label conditions: {sorted(unknown)}")
    if args.dry_run:
        print(
            {
                "model_key": args.model,
                "task": "single_letter_order",
                "n_letter_pairs": len(records),
                "n_items_per_condition": min(
                    len(records) * 2,
                    args.max_items if args.max_items is not None else len(records) * 2,
                ),
                "answer_label_conditions": condition_keys,
                "sample_responses": args.sample_responses,
            }
        )
        return

    bundle = load_model_bundle(
        args.model,
        device_map=args.device_map,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    output_dir = args.output_dir or (
        REPOSITORY_ROOT
        / "results"
        / "generated"
        / "single_letter_order"
        / args.model
    )
    summaries = []
    for key in condition_keys:
        summaries.append(
            run_condition(
                bundle,
                records,
                config.answer_label_conditions[key],
                args,
                output_dir,
            )
        )
    write_json_atomic(
        output_dir / "metadata.json",
        {
            "schema_version": 1,
            "model_key": args.model,
            "task": "single_letter_order",
            "target_layer": bundle.spec.target_layer,
            "answer_label_conditions": condition_keys,
            "n_items": sum(summary["n_items"] for summary in summaries),
            "sample_responses": args.sample_responses,
            "temperature": args.temperature if args.sample_responses else None,
            "seed": args.seed if args.sample_responses else None,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen25_14b")
    parser.add_argument(
        "--answer-labels",
        nargs="+",
        choices=("right_wrong", "true_false"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--save-activations", action="store_true")
    parser.add_argument(
        "--sample-responses",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
