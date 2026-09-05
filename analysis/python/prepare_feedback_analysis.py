#!/usr/bin/env python3
"""Reduce full feedback-generation outputs to the fields used in statistics.

The complete generated responses are retained in the external result archive.
This script creates the compact, analysis-ready tables distributed with the
repository. No text-generation fields or row-level hashes are copied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MODELS = (
    "qwen25_3b",
    "qwen25_7b",
    "llama3_8b",
    "mistral_7b",
    "gemma2_9b",
    "qwen25_14b",
    "qwen25_32b",
    "qwen25_72b",
)
CONDITIONS = ("baseline", "user_like", "user_dislike")
ARCHIVE_LAYOUT = {
    "feedback_effect": ("feedback_effect", 296, (0.0,)),
    "intervention": (
        "intervention_grid",
        100,
        tuple(round(-1.0 + 0.2 * index, 10) for index in range(11)),
    ),
}
COMMON_COLUMNS = (
    "model_name",
    "target_layer",
    "item_id",
    "condition",
    "alpha_norm",
    "verdict_valid",
    "verdict_strong",
)
STATE_COLUMN = "pre_addition_vaa_projection_unit_z_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory of the complete feedback-generation result archive.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/feedback_induced_sycophancy"),
    )
    return parser.parse_args()


def read_archive_table(
    input_root: Path,
    source_dir: str,
    expected_items: int,
    expected_alpha: tuple[float, ...],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    columns = [*COMMON_COLUMNS, STATE_COLUMN]
    for model in MODELS:
        path = input_root / source_dir / model / "raw_results.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing feedback result: {path}")
        frame = pd.read_json(path, lines=True)[columns]
        frame["alpha_norm"] = frame["alpha_norm"].round(10)
        if set(frame["model_name"].unique()) != {model}:
            raise ValueError(f"Model identifier mismatch in {path}")
        if frame["item_id"].nunique() != expected_items:
            raise ValueError(f"Unexpected item count in {path}")
        observed_alpha = tuple(sorted(frame["alpha_norm"].round(10).unique()))
        if observed_alpha != expected_alpha:
            raise ValueError(f"Unexpected alpha grid in {path}: {observed_alpha}")
        if set(frame["condition"].unique()) != set(CONDITIONS):
            raise ValueError(f"Unexpected feedback conditions in {path}")
        if frame.duplicated(["item_id", "condition", "alpha_norm"]).any():
            raise ValueError(f"Duplicate item-condition-alpha rows in {path}")
        expected_rows = expected_items * len(CONDITIONS) * len(expected_alpha)
        if len(frame) != expected_rows:
            raise ValueError(
                f"Unexpected row count in {path}: {len(frame)} != {expected_rows}"
            )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "models": list(MODELS),
        "conditions": list(CONDITIONS),
        "tables": {},
        "full_generated_text": "external result archive",
    }
    for public_name, (source_dir, n_items, alpha_grid) in ARCHIVE_LAYOUT.items():
        frame = read_archive_table(args.input_root, source_dir, n_items, alpha_grid)
        output_path = args.output_dir / f"{public_name}.csv"
        frame.to_csv(output_path, index=False)
        metadata["tables"][public_name] = {
            "path": str(output_path),
            "rows": len(frame),
            "items_per_model": n_items,
            "alpha_grid": list(alpha_grid),
        }
        print(f"Wrote {output_path} ({len(frame):,} rows)")
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
