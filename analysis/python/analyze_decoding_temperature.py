#!/usr/bin/env python
"""Analyze the compact temperature sensitivity with item-paired inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_prompt_spelling as format_analysis
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.stats.multitest import multipletests

ARCHIVE_RESULT_ROOT = (
    REPOSITORY_ROOT
    / "data"
    / "raw"
    / "model_outputs"
    / "decoding_temperature_sensitivity"
)
STAGING_RESULT_ROOT = REPOSITORY_ROOT / "data" / "raw_external" / "decoding_temperature"
DEFAULT_RESULT_ROOT = (
    ARCHIVE_RESULT_ROOT if ARCHIVE_RESULT_ROOT.is_dir() else STAGING_RESULT_ROOT
)
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "results"
    / "summaries"
    / "generation_robustness"
    / "decoding_temperature"
)
TEMPERATURES = (0.0, 0.2, 1.0)
REFERENCE_TEMPERATURE = 0.2


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_results(
    result_root: Path, model_name: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model_dir = result_root / model_name
    metadata_path = model_dir / "metadata.json"
    result_path = model_dir / "raw_results.jsonl"
    if not metadata_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(f"Incomplete result directory: {model_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise ValueError(f"Run is not complete: {model_dir}")
    if metadata.get("raw_results_sha256") != sha256_file(result_path):
        raise ValueError(f"Raw-result hash mismatch: {result_path}")
    rows = format_analysis.read_jsonl(result_path)
    if len(rows) != int(metadata["n_rows"]):
        raise ValueError(
            f"Row-count mismatch for {model_name}: "
            f"{len(rows)} vs {metadata['n_rows']}"
        )
    return rows, metadata


def prepare_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = format_analysis.prepare_dataframe(rows)
    if set(frame["prompt_version"]) != {"legacy_anwer_v1"}:
        raise ValueError("Temperature analysis must use the exact legacy prompt")
    observed_temperatures = {
        round(float(value), 8) for value in frame["temperature"].unique()
    }
    if observed_temperatures != set(TEMPERATURES):
        raise ValueError(f"Temperature mismatch: {sorted(observed_temperatures)}")
    frame["temperature"] = frame["temperature"].astype(float)
    frame["temperature_label"] = frame["temperature"].map(
        {0.0: "0", 0.2: "0.2", 1.0: "1"}
    )
    expected_sampling = frame["temperature"] > 0
    if not np.array_equal(
        frame["do_sample"].astype(bool).to_numpy(),
        expected_sampling.to_numpy(),
    ):
        raise ValueError("Sampling mode does not match temperature")
    key_columns = [
        "model_name",
        "task",
        "item_id",
        "temperature",
        "alpha_norm",
        "seed",
    ]
    if frame.duplicated(key_columns).any():
        raise ValueError("Duplicate temperature-result keys")
    return frame


def summarize_cells(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_name", "task", "temperature", "alpha_norm"]
    rows: list[dict[str, Any]] = []
    for values, group in frame.groupby(keys, sort=True):
        rows.append(
            {
                **dict(zip(keys, values)),
                "n_rows": len(group),
                "n_items": group["item_id"].nunique(),
                "n_seeds": group["seed"].nunique(),
                "strict_json_rate": group["strict_valid_numeric"].mean(),
                "answer_parse_rate": group["answer_valid_numeric"].mean(),
                "accuracy_all": group["correct_all"].mean(),
                "accuracy_valid": group["correct_valid"].mean(),
                "positive_rate_valid": group["positive_numeric"].mean(),
                "mean_generated_tokens": group["generated_n_tokens"].mean(),
                "truncation_rate": group["truncated"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _ols_slope(alpha_means: pd.Series) -> float:
    if len(alpha_means) < 2:
        return np.nan
    return float(
        np.polyfit(
            alpha_means.index.to_numpy(dtype=float),
            alpha_means.to_numpy(dtype=float),
            deg=1,
        )[0]
    )


def fit_item_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_name", "task", "temperature", "item_id"]
    rows: list[dict[str, Any]] = []
    for values, group in frame.groupby(keys, sort=True):
        alpha_means = (
            group.groupby("alpha_norm", sort=True)["positive_numeric"].mean().dropna()
        )
        slope = _ols_slope(alpha_means)
        # The inner subset reproduces the original -0.2/0/+0.2 estimand so that a
        # wider grid stays comparable with the frozen three-point results.
        inner_means = alpha_means[
            np.abs(alpha_means.index.to_numpy(dtype=float)) <= 0.2 + 1e-9
        ]
        inner_slope = _ols_slope(inner_means)
        endpoint_delta = np.nan
        if len(alpha_means) >= 2:
            endpoint_delta = float(alpha_means.iloc[-1] - alpha_means.iloc[0])
        inner_endpoint_delta = np.nan
        if len(inner_means) >= 2:
            inner_endpoint_delta = float(inner_means.iloc[-1] - inner_means.iloc[0])
        saturation_ratio = np.nan
        if (
            len(alpha_means) >= 2
            and len(inner_means) >= 2
            and np.isfinite(inner_endpoint_delta)
            and abs(inner_endpoint_delta) > 0
        ):
            outer_span = float(alpha_means.index[-1] - alpha_means.index[0])
            inner_span = float(inner_means.index[-1] - inner_means.index[0])
            if inner_span > 0:
                expected = inner_endpoint_delta * (outer_span / inner_span)
                if abs(expected) > 0:
                    saturation_ratio = float(endpoint_delta / expected)
        alpha0 = group[np.isclose(group["alpha_norm"], 0.0)]
        rows.append(
            {
                **dict(zip(keys, values)),
                "positive_slope": slope,
                "inner_slope": inner_slope,
                "endpoint_delta": endpoint_delta,
                "inner_endpoint_delta": inner_endpoint_delta,
                "saturation_ratio": saturation_ratio,
                "n_alpha_levels": int(len(alpha_means)),
                "alpha0_accuracy_all": alpha0["correct_all"].mean(),
                "alpha0_strict_json_rate": alpha0["strict_valid_numeric"].mean(),
                "alpha0_answer_parse_rate": alpha0["answer_valid_numeric"].mean(),
                "alpha0_mean_generated_tokens": alpha0["generated_n_tokens"].mean(),
                "alpha0_truncation_rate": alpha0["truncated"].mean(),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return np.nan, np.nan
    indices = rng.integers(0, len(clean), size=(n_bootstrap, len(clean)))
    means = clean[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_slopes(
    item_metrics: pd.DataFrame,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(bootstrap_seed)
    rows: list[dict[str, Any]] = []
    keys = ["model_name", "task", "temperature"]
    for values, group in item_metrics.groupby(keys, sort=True):
        slopes = group["positive_slope"].dropna().to_numpy(dtype=float)
        low, high = bootstrap_mean_ci(slopes, n_bootstrap, rng)
        rows.append(
            {
                **dict(zip(keys, values)),
                "n_items": len(slopes),
                "mean_positive_slope": float(np.mean(slopes)),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "ci_excludes_zero": bool(low > 0 or high < 0),
            }
        )
    return pd.DataFrame(rows)


def summarize_metric_cis(
    item_metrics: pd.DataFrame,
    metrics: tuple[str, ...],
    n_bootstrap: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(bootstrap_seed)
    rows: list[dict[str, Any]] = []
    keys = ["model_name", "task", "temperature"]
    for values, group in item_metrics.groupby(keys, sort=True):
        for metric in metrics:
            if metric not in group.columns:
                continue
            observed = group[metric].dropna().to_numpy(dtype=float)
            if len(observed) == 0:
                continue
            low, high = bootstrap_mean_ci(observed, n_bootstrap, rng)
            rows.append(
                {
                    **dict(zip(keys, values)),
                    "metric": metric,
                    "n_items": len(observed),
                    "mean": float(np.mean(observed)),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "ci_excludes_zero": bool(low > 0 or high < 0),
                }
            )
    return pd.DataFrame(rows)


def build_alpha_curve_source(
    frame: pd.DataFrame,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """Per-alpha dose-response source data for the steering curve figure."""
    rng = np.random.default_rng(bootstrap_seed)
    keys = ["model_name", "task", "temperature", "alpha_norm"]
    rows: list[dict[str, Any]] = []
    for values, group in frame.groupby(keys, sort=True):
        per_item = group.groupby("item_id")["positive_numeric"].mean().dropna()
        observed = per_item.to_numpy(dtype=float)
        low, high = bootstrap_mean_ci(observed, n_bootstrap, rng)
        rows.append(
            {
                **dict(zip(keys, values)),
                "n_items": len(observed),
                "n_rows": len(group),
                "alpha_raw": (
                    float(group["alpha_raw"].iloc[0])
                    if "alpha_raw" in group.columns
                    else np.nan
                ),
                "positive_rate": float(np.mean(observed)) if len(observed) else np.nan,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "accuracy_all": float(group["correct_all"].mean()),
                "strict_json_rate": float(group["strict_valid_numeric"].mean()),
                "answer_parse_rate": float(group["answer_valid_numeric"].mean()),
                "truncation_rate": float(group["truncated"].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_alpha_curves(curve_source: pd.DataFrame, figures_dir: Path) -> None:
    """Dose-response curves over whatever normalized alpha grid was run."""
    if curve_source.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted(curve_source["model_name"].unique())
    tasks = sorted(curve_source["task"].unique())
    n_rows, n_cols = len(models), len(tasks)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.6 * n_cols, 2.9 * n_rows),
        squeeze=False,
        sharex=True,
    )
    for i, model in enumerate(models):
        for j, task in enumerate(tasks):
            ax = axes[i][j]
            cell = curve_source[
                (curve_source["model_name"] == model) & (curve_source["task"] == task)
            ]
            for temperature in sorted(cell["temperature"].unique()):
                sub = cell[cell["temperature"] == temperature].sort_values("alpha_norm")
                x = sub["alpha_norm"].to_numpy(dtype=float)
                y = sub["positive_rate"].to_numpy(dtype=float)
                ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=1.2,
                    markersize=3.5,
                    label=f"T={temperature:g}",
                )
                ax.fill_between(
                    x,
                    sub["bootstrap_ci_low"].to_numpy(dtype=float),
                    sub["bootstrap_ci_high"].to_numpy(dtype=float),
                    alpha=0.15,
                    linewidth=0,
                )
            ax.axhline(0.5, color="0.6", linewidth=0.6, linestyle=":")
            ax.axvline(0.0, color="0.6", linewidth=0.6, linestyle=":")
            ax.set_title(f"{model} | {task}", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == n_rows - 1:
                ax.set_xlabel("normalized alpha", fontsize=8)
            if j == 0:
                ax.set_ylabel("P(positive)", fontsize=8)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(
            figures_dir / f"alpha_curves.{suffix}",
            dpi=350 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def contrast_threshold(metric: str, reference_mean: float) -> float:
    if metric in {
        "alpha0_accuracy_all",
        "alpha0_answer_parse_rate",
        "alpha0_strict_json_rate",
        "alpha0_truncation_rate",
    }:
        return 0.05
    if metric in {
        "positive_slope",
        "inner_slope",
        "endpoint_delta",
        "inner_endpoint_delta",
    }:
        return 0.25 * abs(reference_mean)
    return np.inf


def paired_temperature_contrasts(
    item_metrics: pd.DataFrame,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(bootstrap_seed)
    metrics = (
        "positive_slope",
        "inner_slope",
        "endpoint_delta",
        "inner_endpoint_delta",
        "alpha0_accuracy_all",
        "alpha0_strict_json_rate",
        "alpha0_answer_parse_rate",
        "alpha0_mean_generated_tokens",
        "alpha0_truncation_rate",
    )
    rows: list[dict[str, Any]] = []
    for (model_name, task), group in item_metrics.groupby(
        ["model_name", "task"], sort=True
    ):
        reference = group[
            np.isclose(group["temperature"], REFERENCE_TEMPERATURE)
        ].set_index("item_id")
        for temperature in (0.0, 1.0):
            candidate = group[np.isclose(group["temperature"], temperature)].set_index(
                "item_id"
            )
            common = reference.index.intersection(candidate.index)
            if len(common) != group["item_id"].nunique():
                raise ValueError(
                    f"Unpaired items for {model_name}/{task}/T={temperature}"
                )
            for metric in metrics:
                paired = pd.DataFrame(
                    {
                        "reference": reference.loc[common, metric],
                        "candidate": candidate.loc[common, metric],
                    }
                ).dropna()
                differences = (paired["candidate"] - paired["reference"]).to_numpy(
                    dtype=float
                )
                low, high = bootstrap_mean_ci(differences, n_bootstrap, rng)
                reference_mean = float(paired["reference"].mean())
                candidate_mean = float(paired["candidate"].mean())
                difference = float(differences.mean())
                threshold = contrast_threshold(metric, reference_mean)
                ci_excludes_zero = bool(low > 1e-12 or high < -1e-12)
                rows.append(
                    {
                        "model_name": model_name,
                        "task": task,
                        "temperature": temperature,
                        "reference_temperature": REFERENCE_TEMPERATURE,
                        "metric": metric,
                        "n_items": len(paired),
                        "reference_mean": reference_mean,
                        "candidate_mean": candidate_mean,
                        "candidate_minus_reference": difference,
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                        "ci_excludes_zero": ci_excludes_zero,
                        "material_threshold": threshold,
                        "material_supported_change": bool(
                            ci_excludes_zero
                            and np.isfinite(threshold)
                            and abs(difference) >= threshold
                        ),
                    }
                )
    return pd.DataFrame(rows)


def summarize_seed_slopes(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["model_name", "task", "temperature", "seed"]
    for values, group in frame.groupby(keys, sort=True):
        alpha_means = group.groupby("alpha_norm")["positive_numeric"].mean().dropna()
        slope = np.nan
        if len(alpha_means) >= 2:
            slope = float(
                np.polyfit(
                    alpha_means.index.to_numpy(dtype=float),
                    alpha_means.to_numpy(dtype=float),
                    1,
                )[0]
            )
        rows.append(
            {
                **dict(zip(keys, values)),
                "positive_slope": slope,
                "answer_parse_rate": group["answer_valid_numeric"].mean(),
                "truncation_rate": group["truncated"].mean(),
            }
        )
    return pd.DataFrame(rows)


def summarize_alpha0_breakdowns(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    alpha0 = frame[np.isclose(frame["alpha_norm"], 0.0)].copy()
    seed_summary = (
        alpha0.groupby(["model_name", "task", "temperature", "seed"], sort=True)
        .agg(
            n_rows=("item_id", "size"),
            accuracy_all=("correct_all", "mean"),
            positive_rate_valid=("positive_numeric", "mean"),
            answer_parse_rate=("answer_valid_numeric", "mean"),
            strict_json_rate=("strict_valid_numeric", "mean"),
            mean_generated_tokens=("generated_n_tokens", "mean"),
            truncation_rate=("truncated", "mean"),
        )
        .reset_index()
    )
    label_summary = (
        alpha0.groupby(
            [
                "model_name",
                "task",
                "temperature",
                "correct_answer",
            ],
            sort=True,
        )
        .agg(
            n_rows=("item_id", "size"),
            accuracy_all=("correct_all", "mean"),
            positive_rate_valid=("positive_numeric", "mean"),
            answer_parse_rate=("answer_valid_numeric", "mean"),
        )
        .reset_index()
    )
    return seed_summary, label_summary


def fit_gee_models(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    coefficient_rows: list[dict[str, Any]] = []
    summaries: list[str] = []
    valid = frame.dropna(subset=["positive_numeric"]).copy()
    for (model_name, task), group in valid.groupby(["model_name", "task"], sort=True):
        formula = (
            "positive_numeric ~ alpha_norm * "
            "C(temperature_label, Treatment(reference='0.2'))"
        )
        try:
            result = smf.gee(
                formula,
                groups="item_id",
                data=group,
                family=sm.families.Gaussian(),
                cov_struct=Exchangeable(),
            ).fit()
            summaries.append(f"\n=== {model_name} | {task} ===\n{result.summary()}\n")
            intervals = result.conf_int()
            for term in result.params.index:
                coefficient_rows.append(
                    {
                        "model_name": model_name,
                        "task": task,
                        "term": term,
                        "estimate": float(result.params[term]),
                        "std_error": float(result.bse[term]),
                        "ci_low": float(intervals.loc[term, 0]),
                        "ci_high": float(intervals.loc[term, 1]),
                        "p_value": float(result.pvalues[term]),
                        "n_rows": int(result.nobs),
                        "n_item_clusters": int(group["item_id"].nunique()),
                    }
                )
        except Exception as error:
            summaries.append(f"\n=== {model_name} | {task} | FAILED ===\n{error}\n")
    return pd.DataFrame(coefficient_rows), "".join(summaries)


def write_report(
    path: Path,
    frame: pd.DataFrame,
    slope_summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    gee_coefficients: pd.DataFrame,
    extended_metric_cis: pd.DataFrame | None = None,
) -> None:
    n_rows = len(frame)
    n_parse = int(frame["answer_valid_numeric"].sum())
    n_truncated = int(frame["truncated"].sum())
    slope_lines: list[str] = []
    for row in slope_summary.itertuples(index=False):
        slope_lines.append(
            f"| {row.model_name} | {row.task} | {row.temperature:g} | "
            f"{row.mean_positive_slope:.3f} "
            f"[{row.bootstrap_ci_low:.3f}, {row.bootstrap_ci_high:.3f}] |"
        )
    temperature_lines: list[str] = []
    for temperature, group in frame.groupby("temperature", sort=True):
        temperature_lines.append(
            f"| {temperature:g} | {len(group)} | "
            f"{group['answer_valid_numeric'].mean():.3f} | "
            f"{group['strict_valid_numeric'].mean():.3f} | "
            f"{group['truncated'].mean():.4f} |"
        )
    slope_contrasts = contrasts[contrasts["metric"] == "positive_slope"]
    accuracy_flags = contrasts[
        (contrasts["metric"] == "alpha0_accuracy_all")
        & contrasts["material_supported_change"]
    ]
    parse_flags = contrasts[
        (contrasts["metric"] == "alpha0_answer_parse_rate")
        & contrasts["material_supported_change"]
    ]
    interaction_terms = gee_coefficients[
        gee_coefficients["term"].str.startswith("alpha_norm:C(")
    ]
    interaction_p_column = (
        "p_holm_temperature_interactions"
        if "p_holm_temperature_interactions" in interaction_terms.columns
        else "p_value"
    )
    interaction_significant = bool(
        (interaction_terms[interaction_p_column] < 0.05).any()
    )
    slope_triggered = bool(
        slope_contrasts["material_supported_change"].any() or interaction_significant
    )
    all_positive = bool((slope_summary["mean_positive_slope"] > 0).all())
    steering_decision = (
        "The steering robustness gate was not triggered."
        if all_positive and not slope_triggered
        else "The steering robustness gate requires targeted inspection."
    )
    n_slope_cells = int(len(slope_summary))
    n_slope_positive_excl = int(
        (
            (slope_summary["mean_positive_slope"] > 0)
            & slope_summary["ci_excludes_zero"]
        ).sum()
    )
    alpha_grid = sorted(float(v) for v in frame["alpha_norm"].unique())
    alpha_grid_text = ", ".join(f"{v:+g}" for v in alpha_grid)
    slope_sentence = (
        f"{n_slope_positive_excl} of {n_slope_cells} temperature-specific slope "
        "intervals exclude zero in the positive direction over the normalized "
        f"alpha grid [{alpha_grid_text}]."
    )
    n_slope_contrasts = int(len(slope_contrasts))
    n_slope_contrast_excl = int(slope_contrasts["ci_excludes_zero"].sum())
    n_interaction = int(len(interaction_terms))
    n_interaction_sig = int((interaction_terms[interaction_p_column] < 0.05).sum())
    n_gee_models = (
        int(gee_coefficients.groupby(["model_name", "task"]).ngroups)
        if len(gee_coefficients)
        else 0
    )
    slope_contrast_clause = (
        f"None of the {n_slope_contrasts} item-paired slope contrasts against "
        "T=0.2 excludes zero"
        if n_slope_contrast_excl == 0
        else f"{n_slope_contrast_excl} of {n_slope_contrasts} item-paired slope "
        "contrasts against T=0.2 exclude zero"
    )
    interaction_clause = (
        f"none of the {n_interaction} temperature-contrast interaction "
        f"coefficients across the {n_gee_models} clustered GEEs is significant "
        "at p < 0.05"
        if n_interaction_sig == 0
        else f"{n_interaction_sig} of {n_interaction} temperature-contrast "
        f"interaction coefficients across the {n_gee_models} clustered GEEs is "
        "significant at p < 0.05"
    )
    contrast_sentence = f"{slope_contrast_clause}, and {interaction_clause}."
    endpoint_section: list[str] = []
    if extended_metric_cis is not None and not extended_metric_cis.empty:
        endpoint_section = [
            "## Endpoint and Inner-Grid Sensitivity",
            "",
            "`endpoint_delta` is P(positive) at the largest normalized alpha minus "
            "P(positive) at the smallest. `inner_slope` and `inner_endpoint_delta` "
            "are restricted to |alpha| <= 0.2 and therefore reproduce the "
            "three-point estimand used in the main analysis. "
            "`saturation_ratio` near 1 indicates a linear dose-response; values "
            "below 1 indicate the outer alpha points are saturating.",
            "",
            "| Model | Task | Temperature | Metric | Mean [95% CI] |",
            "|---|---|---:|---|---:|",
        ]
        for row in extended_metric_cis.itertuples(index=False):
            endpoint_section.append(
                f"| {row.model_name} | {row.task} | {row.temperature:g} | "
                f"{row.metric} | {row.mean:.3f} "
                f"[{row.bootstrap_ci_low:.3f}, {row.bootstrap_ci_high:.3f}] |"
            )
        endpoint_section.append("")
    accuracy_lines = [
        (
            f"- {row.model_name}, {row.task}, T={row.temperature:g}: "
            f"{row.candidate_minus_reference:+.3f} "
            f"[{row.bootstrap_ci_low:.3f}, {row.bootstrap_ci_high:.3f}] "
            "relative to T=0.2."
        )
        for row in accuracy_flags.itertuples(index=False)
    ]
    if not accuracy_lines:
        accuracy_lines = ["- No supported material alpha-zero accuracy shift."]
    parse_lines = [
        (
            f"- {row.model_name}, {row.task}, T={row.temperature:g}: "
            f"{row.candidate_minus_reference:+.3f} "
            f"[{row.bootstrap_ci_low:.3f}, {row.bootstrap_ci_high:.3f}]."
        )
        for row in parse_flags.itertuples(index=False)
    ]
    if not parse_lines:
        parse_lines = ["- No supported material answer-parseability shift."]
    lines = [
        "# Temperature Sensitivity Report",
        "",
        "## Completeness",
        "",
        f"- Total rows: {n_rows}.",
        f"- Answers extracted with the manuscript scoring rule: {n_parse}/{n_rows}.",
        f"- Truncated generations: {n_truncated}/{n_rows}.",
        f"- Models: {frame['model_name'].nunique()}; tasks: {frame['task'].nunique()}.",
        "",
        "| Temperature | Rows | Answer parse rate | Strict JSON rate | Truncation rate |",
        "|---:|---:|---:|---:|---:|",
        *temperature_lines,
        "",
        "## Steering Slopes",
        "",
        "Item-bootstrap 95% confidence intervals are shown.",
        "",
        "| Model | Task | Temperature | Mean slope [95% CI] |",
        "|---|---|---:|---:|",
        *slope_lines,
        "",
        slope_sentence,
        contrast_sentence,
        "",
        *endpoint_section,
        "## Secondary Temperature Differences",
        "",
        "Supported material alpha-zero accuracy contrasts:",
        "",
        *accuracy_lines,
        "",
        "Supported material answer-parseability contrasts:",
        "",
        *parse_lines,
        "",
        "Strict JSON validity is a format diagnostic and is not treated as verdict",
        "loss when the answer field remains extractable with the manuscript scoring rule.",
        "",
        "## Screening Decision",
        "",
        steering_decision,
        "",
        "Secondary accuracy or format shifts must still be reported and inspected,",
        "but they do not by themselves imply a VAA slope reversal. Expansion should",
        "be limited to cases where the steering direction or central interpretation",
        "changes after that inspection.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    for model_name in args.models:
        model_rows, model_metadata = load_model_results(
            args.result_root.resolve(), model_name
        )
        rows.extend(model_rows)
        metadata[model_name] = model_metadata
    frame = prepare_dataframe(rows)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_cells(frame)
    item_metrics = fit_item_metrics(frame)
    slope_summary = summarize_slopes(
        item_metrics, args.n_bootstrap, args.bootstrap_seed
    )
    contrasts = paired_temperature_contrasts(
        item_metrics, args.n_bootstrap, args.bootstrap_seed + 1
    )
    seed_slopes = summarize_seed_slopes(frame)
    extended_metric_cis = summarize_metric_cis(
        item_metrics,
        (
            "endpoint_delta",
            "inner_endpoint_delta",
            "inner_slope",
            "saturation_ratio",
        ),
        args.n_bootstrap,
        args.bootstrap_seed + 2,
    )
    alpha_curve_source = build_alpha_curve_source(
        frame, args.n_bootstrap, args.bootstrap_seed + 3
    )
    alpha0_seed_summary, alpha0_label_summary = summarize_alpha0_breakdowns(frame)
    gee_coefficients, gee_summaries = fit_gee_models(frame)
    interaction_mask = gee_coefficients["term"].str.startswith("alpha_norm:C(")
    gee_coefficients["p_holm_temperature_interactions"] = np.nan
    gee_coefficients["degenerate_zero_effect"] = False
    if interaction_mask.any():
        interaction_p_values = gee_coefficients.loc[interaction_mask, "p_value"].copy()
        degenerate_mask = (
            gee_coefficients.loc[interaction_mask, "estimate"].abs() < 1e-12
        )
        interaction_p_values.loc[degenerate_mask] = 1.0
        gee_coefficients.loc[interaction_p_values.index, "degenerate_zero_effect"] = (
            degenerate_mask
        )
        gee_coefficients.loc[interaction_mask, "p_holm_temperature_interactions"] = (
            multipletests(interaction_p_values, method="holm")[1]
        )

    summary.to_csv(output_dir / "summary_by_cell.csv", index=False)
    item_metrics.to_csv(output_dir / "item_level_metrics.csv", index=False)
    slope_summary.to_csv(output_dir / "temperature_slope_summary.csv", index=False)
    contrasts.to_csv(output_dir / "paired_temperature_contrasts.csv", index=False)
    seed_slopes.to_csv(output_dir / "seed_level_summary.csv", index=False)
    extended_metric_cis.to_csv(
        output_dir / "endpoint_and_inner_metric_cis.csv", index=False
    )
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    alpha_curve_source.to_csv(figures_dir / "source_data_alpha_curves.csv", index=False)
    plot_alpha_curves(alpha_curve_source, figures_dir)
    alpha0_seed_summary.to_csv(output_dir / "alpha0_seed_summary.csv", index=False)
    alpha0_label_summary.to_csv(output_dir / "alpha0_label_summary.csv", index=False)
    gee_coefficients.to_csv(
        output_dir / "gee_temperature_coefficients.csv", index=False
    )
    (output_dir / "gee_temperature_summaries.txt").write_text(
        gee_summaries, encoding="utf-8"
    )
    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(
        output_dir / "temperature_sensitivity_report.md",
        frame,
        slope_summary,
        contrasts,
        gee_coefficients,
        extended_metric_cis,
    )
    print(f"Wrote temperature analysis to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", nargs="+", default=("qwen25_7b", "llama3_8b"))
    parser.add_argument("--n_bootstrap", type=int, default=10_000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260724)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
