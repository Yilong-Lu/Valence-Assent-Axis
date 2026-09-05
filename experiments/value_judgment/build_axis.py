"""Rebuild the Value Judgment PC1 axes from the frozen extraction set."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from vaa.activations import capture_assistant_start_activations
from vaa.axis import cosine_axis_alignment, derive_layer_axes
from vaa.config import REPOSITORY_ROOT
from vaa.experiment_config import (
    load_judgment_experiment_config,
    load_task_stimuli,
)
from vaa.io import write_json_atomic
from vaa.models import get_transformer_layers, load_model_bundle
from vaa.prompts import render_prompt
from vaa.scoring import render_chat_prompt, score_next_token_candidates_batch
from vaa.steering import load_vaa_vector


def batches(values, batch_size):
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def binary_support_probabilities(bundle, prompts, batch_size):
    probabilities = []
    rendered = [render_chat_prompt(bundle.tokenizer, prompt) for prompt in prompts]
    for prompt_batch in batches(rendered, batch_size):
        scores = score_next_token_candidates_batch(
            bundle.model,
            bundle.tokenizer,
            prompt_batch,
            ("A", "B"),
            target_layer=bundle.spec.target_layer,
            steering_vector=bundle.vaa_vector,
            alpha_raw=0.0,
        )
        probabilities.extend(row["candidate_probabilities"]["A"] for row in scores)
    return np.asarray(probabilities, dtype=np.float64)


def dry_run_summary(model_key: str, all_layers: bool) -> dict:
    config = load_judgment_experiment_config()
    task = config.tasks["value_judgment_binary"]
    extraction_task = replace(
        task,
        stimulus_files=(task.stimulus_files[0],),
    )
    stimuli = load_task_stimuli(extraction_task)
    return {
        "model_key": model_key,
        "n_extraction_statements": len(stimuli),
        "layer_scope": "all" if all_layers else "selected",
        "prompt_key": task.prompt_key,
        "candidates": list(task.candidates),
    }


def run(args: argparse.Namespace) -> None:
    if args.dry_run:
        print(dry_run_summary(args.model, args.all_layers))
        return

    config = load_judgment_experiment_config()
    task = config.tasks["value_judgment_binary"]
    extraction_task = replace(
        task,
        stimulus_files=(task.stimulus_files[0],),
    )
    stimuli = load_task_stimuli(extraction_task)
    prompts = [
        render_prompt(task.prompt_key, statement=row["statement"])
        for row in stimuli
    ]
    bundle = load_model_bundle(
        args.model,
        device_map=args.device_map,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        load_vector=False,
    )
    layer_numbers = (
        list(range(len(get_transformer_layers(bundle.model))))
        if args.all_layers
        else [bundle.spec.target_layer]
    )
    activations = capture_assistant_start_activations(
        bundle.model,
        bundle.tokenizer,
        prompts,
        layer_numbers,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
    )
    support_probability = binary_support_probabilities(
        bundle,
        prompts,
        args.batch_size,
    )
    axes = derive_layer_axes(activations, support_probability)
    frozen_vector = (
        load_vaa_vector(bundle.spec) if bundle.spec.vector_path.is_file() else None
    )

    output_dir = args.output_dir or (
        REPOSITORY_ROOT / "results" / "generated" / "value_judgment" / args.model
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for layer, result in axes.items():
        np.save(output_dir / f"axis_layer{layer}.npy", result.vector, allow_pickle=False)
        frozen_cosine = None
        if layer == bundle.spec.target_layer and frozen_vector is not None:
            frozen_cosine = cosine_axis_alignment(
                result.vector,
                frozen_vector.numpy(),
            )
        summaries.append(
            {
                "layer": layer,
                "pc1_explained_variance": float(result.explained_variance_ratio[0]),
                "pc1_support_correlation": result.response_correlation,
                "sign_flipped": result.sign_flipped,
                "frozen_vector_cosine": frozen_cosine,
            }
        )
    if args.save_activations:
        np.savez_compressed(
            output_dir / "assistant_start_activations.npz",
            **{f"layer_{layer}": values for layer, values in activations.items()},
        )
    write_json_atomic(
        output_dir / "metadata.json",
        {
            "model_key": args.model,
            "prompt_key": task.prompt_key,
            "n_extraction_statements": len(stimuli),
            "target_layer": bundle.spec.target_layer,
            "layers": summaries,
        },
    )
    print(f"Wrote {len(summaries)} layer axes to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen25_14b")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--save-activations", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
