"""Extract the compact Qwen2.5-14B Source Data used by Figure 2.

The intervention input must be the complete ``steer_results_raw.csv`` table.
This script selects the reported layer, task conditions, and valid alpha range
and writes the compact table tracked in the code and data releases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


GROUPS = ("continuous09", "binarySentiment", "continuousSentiment")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pca-results", type=Path, required=True)
    parser.add_argument(
        "--steering-results",
        type=Path,
        required=True,
        help="Complete steer_results_raw.csv file, including total_prob.",
    )
    parser.add_argument("--valid-alpha", type=Path, required=True)
    parser.add_argument("--target-layer", type=int, default=28)
    parser.add_argument("--output-dir", type=Path, default=Path("data/source_data"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    layer_destination = args.output_dir / "figure2_layer_profiles_qwen25_14b.json"
    layer_destination.write_text(args.pca_results.read_text(encoding="utf-8"), encoding="utf-8")

    frame = pd.read_csv(args.steering_results)
    required_columns = {
        "statement_idx",
        "layer",
        "test_template",
        "alpha",
        "expected_mean",
        "total_prob",
    }
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"The Figure 2 extraction requires steer_results_raw.csv; "
            f"missing columns: {missing}"
        )
    frame["alpha"] = frame["alpha"].round(3)
    frame["group"] = frame["test_template"].map(lambda value: value.split("_")[2])
    frame["statement_split"] = frame["test_template"].str.rsplit("_", n=1).str[-1]
    frame["statement_id"] = (
        frame["statement_split"].astype(str)
        + ":"
        + frame["statement_idx"].astype(str)
    )
    frame["expected_probability"] = frame["expected_mean"].astype(float)
    continuous = frame["group"].str.contains("continuous")
    frame.loc[continuous, "expected_probability"] /= 9.0

    valid = json.loads(args.valid_alpha.read_text(encoding="utf-8"))
    alpha_low, alpha_high = valid[str(args.target_layer)]
    frame = frame[
        frame["layer"].eq(args.target_layer)
        & frame["alpha"].between(alpha_low, alpha_high)
        & frame["group"].isin(GROUPS)
    ].copy()
    frame["alpha_norm"] = frame["alpha"].map(
        lambda value: value / -alpha_low if value < 0 else value / alpha_high
    ).round(3)
    columns = [
        "group",
        "statement_split",
        "statement_idx",
        "statement_id",
        "alpha",
        "alpha_norm",
        "expected_probability",
    ]
    destination = args.output_dir / "figure2_intervention_qwen25_14b.csv"
    frame[columns].to_csv(destination, index=False)
    print(f"Wrote {len(frame)} rows to {destination}")


if __name__ == "__main__":
    main()
