#!/usr/bin/env python
"""Reproduce cross- and within-domain intervention coefficients.

The reported analysis standardized raw intervention coefficients and expected
responses within each model and task, then fitted a random-intercept MixedLM by
statement. This script preserves that analysis for Figure 2 and Supplementary
Table B2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


GROUP_LABELS = {
    "binarySentiment": "Sentiment Analysis: Binary",
    "continuousSentiment": "Sentiment Analysis: Continuous",
    "continuous09": "Value Judgment: Continuous",
}
MODEL_ORDER = (
    "gemma2_9b",
    "llama3_8b",
    "mistral_7b",
    "qwen25_14b",
    "qwen25_32b",
    "qwen25_3b",
    "qwen25_72b",
    "qwen25_7b",
)


def fit_cell(frame: pd.DataFrame, model_name: str, group_name: str) -> dict[str, object]:
    frame = frame.copy()
    frame["z_alpha"] = (frame["alpha"] - frame["alpha"].mean()) / frame["alpha"].std()
    frame["z_expected_prob"] = (
        frame["expected_prob"] - frame["expected_prob"].mean()
    ) / frame["expected_prob"].std()
    fit = smf.mixedlm(
        "z_expected_prob ~ z_alpha",
        frame,
        groups=frame["statement_id"],
    ).fit()
    interval = fit.conf_int().loc["z_alpha"]
    return {
        "task": GROUP_LABELS[group_name],
        "model_name": model_name,
        "coefficient": fit.params["z_alpha"],
        "standard_error": fit.bse["z_alpha"],
        "statistic": fit.tvalues["z_alpha"],
        "p_value": fit.pvalues["z_alpha"],
        "ci_low": interval.iloc[0],
        "ci_high": interval.iloc[1],
        "converged": bool(fit.converged),
        "n_rows": len(frame),
        "n_items": frame["statement_id"].nunique(),
    }


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=repository_root
        / "data/source_data/supplement_cross_domain_intervention.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "results/summaries/cross_domain_control.csv",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input).rename(
        columns={"group": "Group", "expected_probability": "expected_prob"}
    )
    if "model_name" not in data:
        data["model_name"] = "qwen25_14b"
    required = {"model_name", "Group", "statement_id", "alpha", "expected_prob"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    rows = []
    available_models = [
        model_name for model_name in MODEL_ORDER if model_name in set(data["model_name"])
    ]
    for group_name in GROUP_LABELS:
        for model_name in available_models:
            frame = data[
                data["Group"].eq(group_name) & data["model_name"].eq(model_name)
            ]
            if frame.empty:
                raise ValueError(f"No rows for {model_name} / {group_name}")
            rows.append(fit_cell(frame, model_name, group_name))

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} cross-domain coefficients to {args.output}")


if __name__ == "__main__":
    main()
