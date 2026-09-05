#!/usr/bin/env python3
"""Prepare compact Subjective Preference tables from experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MODELS = (
    "qwen25_3b",
    "qwen25_7b",
    "qwen25_14b",
    "qwen25_32b",
    "qwen25_72b",
    "llama3_8b",
    "mistral_7b",
    "gemma2_9b",
)


def one_match(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one {pattern} file in {directory}; found {len(matches)}")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Directory containing one preference_control directory per model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/subjective_preference"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectory_dir = args.output_dir / "trajectories"
    projection_dir = args.output_dir / "assistant_start_projection"
    lexical_dir = args.output_dir / "lexical_pairs"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    projection_dir.mkdir(parents=True, exist_ok=True)
    lexical_dir.mkdir(parents=True, exist_ok=True)

    records: dict[str, object] = {}
    for model in MODELS:
        model_dir = args.input_root / model / "preference_control"
        trajectory_path = one_match(model_dir, "processed_order_decomp_layer*.csv")
        projection_path = one_match(
            model_dir, "internal_projection_order_decomp_layer*.csv"
        )
        raw_path = one_match(model_dir, "raw_logits_layer*.csv")

        trajectory = pd.read_csv(trajectory_path)
        projection = pd.read_csv(projection_path)
        raw = pd.read_csv(raw_path, usecols=["pair_id", "word_A", "word_B"])
        pairs = raw.drop_duplicates().sort_values("pair_id").reset_index(drop=True)
        if pairs["pair_id"].duplicated().any():
            raise ValueError(f"Inconsistent lexical metadata for {model}")
        if set(trajectory["template_name"].unique()) != {"preference_control"}:
            raise ValueError(f"Unexpected prompt template for {model}")
        n_pairs = trajectory["pair_id"].nunique()
        if n_pairs not in (208, 209) or len(trajectory) != n_pairs * 11:
            raise ValueError(f"Incomplete trajectories for {model}")
        if projection["pair_id"].nunique() != n_pairs or len(projection) != n_pairs:
            raise ValueError(f"Incomplete assistant-start projections for {model}")

        trajectory_output = trajectory_dir / f"{model}.csv"
        projection_output = projection_dir / f"{model}.csv"
        lexical_output = lexical_dir / f"{model}.csv"
        trajectory.to_csv(trajectory_output, index=False)
        projection.to_csv(projection_output, index=False)
        pairs.to_csv(lexical_output, index=False)
        records[model] = {
            "target_layer": int(trajectory["target_layer"].iloc[0]),
            "pairs": n_pairs,
            "trajectory_rows": len(trajectory),
            "projection_rows": len(projection),
        }

    metadata = {
        "models": records,
        "alpha_grid": [round(-1.0 + 0.2 * index, 10) for index in range(11)],
        "prompt": "content-free A/B preference prompt",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared Subjective Preference inputs for {len(MODELS)} models")


if __name__ == "__main__":
    main()
