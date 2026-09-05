"""Extract cross-domain intervention rows used by the Supplementary Information."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MODEL_DIRECTORIES = {
    "qwen25_3b": "qwen25_3B_it",
    "mistral_7b": "mistral_7B_it",
    "qwen25_7b": "qwen25_7B_it",
    "llama3_8b": "llama3_8B_it",
    "gemma2_9b": "gemma2_9B_it",
    "qwen25_14b": "qwen25_14B_it",
    "qwen25_32b": "qwen25_32B_it",
    "qwen25_72b": "qwen25_72B_it",
}
TARGET_LAYERS = {
    "qwen25_3b": 26,
    "qwen25_7b": 18,
    "qwen25_14b": 28,
    "qwen25_32b": 43,
    "qwen25_72b": 52,
    "llama3_8b": 13,
    "mistral_7b": 14,
    "gemma2_9b": 22,
}
GROUPS = ("continuous09", "binarySentiment", "continuousSentiment")


def normalize_alpha(value: float, lower: float, upper: float) -> float:
    scale = -lower if value < 0 else upper
    return round(value / scale, 3) if scale else 0.0


def extract_model(steering_root: Path, model: str) -> pd.DataFrame:
    run_dir = steering_root / MODEL_DIRECTORIES[model] / "train_survey_binaryAB_stmt_train"
    frame = pd.read_csv(run_dir / "steer_results_raw.csv")
    frame["alpha"] = frame["alpha"].round(3)
    frame["Group"] = frame["test_template"].str.split("_").str[2]
    frame["statement_split"] = frame["test_template"].str.rsplit("_", n=1).str[-1]
    frame["statement_id"] = (
        frame["statement_split"].astype(str)
        + ":"
        + frame["statement_idx"].astype(str)
    )
    frame["expected_prob"] = frame["expected_mean"].astype(float)
    continuous = frame["Group"].str.contains("continuous")
    frame.loc[continuous, "expected_prob"] /= 9.0

    layer = TARGET_LAYERS[model]
    valid = json.loads((run_dir / "valid_layer.json").read_text(encoding="utf-8"))
    lower, upper = valid[str(layer)]
    frame = frame[
        frame["layer"].eq(layer)
        & frame["alpha"].between(lower, upper)
        & frame["Group"].isin(GROUPS)
    ].copy()
    binary = frame[frame["Group"].eq("binarySentiment")]
    alpha_low, alpha_high = binary["alpha"].min(), binary["alpha"].max()
    frame["alpha_norm"] = frame["alpha"].map(
        lambda value: normalize_alpha(value, alpha_low, alpha_high)
    )
    frame["model_name"] = model
    return frame[
        [
            "model_name",
            "Group",
            "statement_split",
            "statement_idx",
            "statement_id",
            "alpha",
            "alpha_norm",
            "expected_prob",
        ]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steering-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/source_data/supplement_cross_domain_intervention.csv"),
    )
    args = parser.parse_args()
    frames = [extract_model(args.steering_root, model) for model in MODEL_DIRECTORIES]
    result = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result):,} rows to {args.output}")


if __name__ == "__main__":
    main()
