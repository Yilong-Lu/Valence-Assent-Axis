"""Derive the Valence Axis and compare it with the frozen VAA."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vaa.activations import capture_assistant_start_activations
from vaa.axis_control_config import load_axis_control_config, load_record_array
from vaa.config import REPOSITORY_ROOT
from vaa.io import write_json_atomic, write_jsonl_atomic
from vaa.models import load_model_bundle
from vaa.prompts import render_prompt
from vaa.representation import compare_task_axis_to_vaa


def build_items(records: list[dict[str, str]], prompt_key: str) -> list[dict]:
    items = []
    for record in records:
        for label, field, numeric_label in (
            ("positive", "positive_word", 1.0),
            ("negative", "negative_word", 0.0),
        ):
            word = record[field]
            items.append(
                {
                    "item_id": f"{record['pair_id']}::{label}",
                    "pair_id": record["pair_id"],
                    "domain": record["domain"],
                    "word": word,
                    "valence_label": label,
                    "valence_numeric": numeric_label,
                    "prompt": render_prompt(prompt_key, word=word.lower()),
                }
            )
    return items


def summary_from_result(result) -> dict:
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
        "pc1_oriented_toward_positive": True,
        "pc1_sign_flipped": result.sign_flipped,
    }


def run(args: argparse.Namespace) -> None:
    config = load_axis_control_config().valence
    records = load_record_array(config.stimulus_file)
    items = build_items(records, config.prompt_key)
    if args.max_items is not None:
        items = items[: args.max_items]
    if args.dry_run:
        print(
            {
                "model_key": args.model,
                "task": "valence_axis_extraction",
                "prompt_key": config.prompt_key,
                "n_word_pairs": len(records),
                "n_items": len(items),
                "target_layer": "registered model layer",
            }
        )
        return

    bundle = load_model_bundle(
        args.model,
        device_map=args.device_map,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    layer = bundle.spec.target_layer
    activations = capture_assistant_start_activations(
        bundle.model,
        bundle.tokenizer,
        [item["prompt"] for item in items],
        [layer],
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
    )[layer]
    labels = np.asarray([item["valence_numeric"] for item in items])
    result = compare_task_axis_to_vaa(
        activations,
        bundle.vaa_vector.numpy(),
        orient_toward=labels,
    )

    output_dir = args.output_dir or (
        REPOSITORY_ROOT / "results" / "generated" / "valence" / args.model
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "valence_axis.npy", result.axis_vector, allow_pickle=False)
    output_rows = []
    for index, item in enumerate(items):
        output_rows.append(
            {
                "model_key": args.model,
                "task": "valence_axis_extraction",
                "target_layer": layer,
                "prompt_key": config.prompt_key,
                **item,
                "pc1_score": float(result.pc1_scores[index]),
                "pc2_score": float(result.pc2_scores[index]),
                "vaa_projection": float(result.vaa_projections[index]),
            }
        )
    write_jsonl_atomic(output_dir / "items.jsonl", output_rows)
    if args.save_activations:
        np.save(
            output_dir / "assistant_start_activations.npy",
            activations,
            allow_pickle=False,
        )
    write_json_atomic(
        output_dir / "summary.json",
        {
            "schema_version": 1,
            "model_key": args.model,
            "task": "valence_axis_extraction",
            "display_name": config.display_name,
            "prompt_key": config.prompt_key,
            "target_layer": layer,
            "n_word_pairs": len(records),
            "n_items": len(items),
            "metrics": summary_from_result(result),
            "singular_values": result.singular_values.tolist(),
            "explained_variance_ratio": result.explained_variance_ratio.tolist(),
        },
    )
    print(f"Wrote Valence Axis outputs to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen25_14b")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--save-activations", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
