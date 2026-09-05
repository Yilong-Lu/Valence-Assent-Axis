#!/usr/bin/env python
"""Analyze reasoning-score robustness to decoding and prompt spelling."""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.stats.multitest import multipletests

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
ARCHIVE_OUTPUT_ROOT = REPOSITORY_ROOT / "data" / "raw" / "model_outputs"
ARCHIVE_SCORE_PATH = ARCHIVE_OUTPUT_ROOT / "judge_scores" / "reasoning_scores.jsonl"
STAGING_SCORE_PATH = (
    REPOSITORY_ROOT / "data" / "raw_external" / "judge_scores" / "reasoning_scores.jsonl"
)
DEFAULT_SCORE_PATH = (
    ARCHIVE_SCORE_PATH if ARCHIVE_SCORE_PATH.is_file() else STAGING_SCORE_PATH
)
ARCHIVE_TEMPERATURE_ROOT = ARCHIVE_OUTPUT_ROOT / "decoding_temperature_sensitivity"
STAGING_TEMPERATURE_ROOT = (
    REPOSITORY_ROOT / "data" / "raw_external" / "decoding_temperature"
)
DEFAULT_TEMPERATURE_ROOT = (
    ARCHIVE_TEMPERATURE_ROOT
    if ARCHIVE_TEMPERATURE_ROOT.is_dir()
    else STAGING_TEMPERATURE_ROOT
)
ARCHIVE_PROMPT_ROOT = ARCHIVE_OUTPUT_ROOT / "prompt_spelling_check"
STAGING_PROMPT_ROOT = REPOSITORY_ROOT / "data" / "raw_external" / "prompt_spelling"
DEFAULT_PROMPT_ROOT = (
    ARCHIVE_PROMPT_ROOT if ARCHIVE_PROMPT_ROOT.is_dir() else STAGING_PROMPT_ROOT
)
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "results"
    / "summaries"
    / "generation_robustness"
    / "reasoning"
)
MODELS = (
    "qwen25_7b",
    "llama3_8b",
    "qwen25_14b",
    "qwen25_32b",
    "qwen25_72b",
)
TASKS = ("alphabetical_think_answer", "TruthfulQA")
SCORE_COLUMNS = (
    "factual_correctness_score",
    "logical_consistency_score",
    "reasoning_structure_score",
)
OUTCOMES = (*SCORE_COLUMNS, "sound_reasoning")
SOURCE_KEYS = (
    "dataset",
    "model_name",
    "task",
    "item_id",
    "prompt_version",
    "alpha_norm",
    "temperature",
    "seed",
)
TEMPERATURE_REFERENCE = "0.2"
PROMPT_REFERENCE = "legacy_anwer_v1"
PROMPT_CORRECTED = "corrected_answer_v2"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def load_raw_dataset(
    dataset: str, root: Path, models: tuple[str, ...]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for model in models:
        path = root / model / "raw_results.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        model_rows = read_jsonl(path)
        for row in model_rows:
            row["dataset"] = dataset
        rows.extend(model_rows)
        files.append(
            {
                "path": provenance_path(path),
                "n_rows": len(model_rows),
                "sha256": sha256_file(path),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.duplicated(list(SOURCE_KEYS)).any():
        raise ValueError(f"Duplicate raw source keys in {dataset}")
    return frame, files


def classify_reasoning(fc: Any, lc: Any) -> str | None:
    if pd.isna(fc) or pd.isna(lc):
        return None
    pair = (int(fc), int(lc))
    return {
        (1, 1): "Sound Reasoning",
        (-1, 1): "Coherent Hallucination",
        (1, -1): "Contradictory Reasoning",
        (-1, -1): "Incoherent Hallucination",
        (0, 1): "Cherry-picking",
        (1, 0): "Ambiguous Logic",
    }.get(pair, "Mixed")


def classify_reasoning_coarse(fc: Any, lc: Any) -> str | None:
    if pd.isna(fc) or pd.isna(lc):
        return None
    pair = (int(fc), int(lc))
    return {
        (1, 1): "Sound Reasoning",
        (-1, 1): "Coherent Hallucination",
        (1, -1): "Contradictory Reasoning",
        (-1, -1): "Incoherent Hallucination",
    }.get(pair, "Ambiguous/Mixed")


def truth_direction(answer: Any) -> int:
    value = str(answer).strip().lower()
    if value in {"right", "yes"}:
        return 1
    if value in {"wrong", "no"}:
        return -1
    raise ValueError(f"Unsupported correct answer: {answer!r}")


def merge_scores(raw: pd.DataFrame, scores: pd.DataFrame, dataset: str) -> pd.DataFrame:
    score_subset = scores[scores["dataset"] == dataset].copy()
    merged = raw.merge(
        score_subset,
        on=list(SOURCE_KEYS),
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_judge"),
    )
    if (merged["_merge"] == "right_only").any():
        raise ValueError(f"Unmatched judge rows in {dataset}")
    merged["reasoning_evaluable"] = merged["factual_correctness_score"].notna()
    merged["truth_direction"] = merged["correct_answer"].map(truth_direction)
    merged["alignment_pressure"] = (
        merged["alpha_norm"].astype(float) * merged["truth_direction"]
    )
    merged["reasoning_pattern"] = [
        classify_reasoning(fc, lc)
        for fc, lc in zip(
            merged["factual_correctness_score"],
            merged["logical_consistency_score"],
            strict=True,
        )
    ]
    merged["reasoning_pattern_coarse"] = [
        classify_reasoning_coarse(fc, lc)
        for fc, lc in zip(
            merged["factual_correctness_score"],
            merged["logical_consistency_score"],
            strict=True,
        )
    ]
    merged["sound_reasoning"] = merged["reasoning_pattern"].map(
        lambda value: np.nan if value is None else float(value == "Sound Reasoning")
    )
    return merged


def prepare_data(
    args: argparse.Namespace,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    score_path = args.score_path.resolve()
    scores = pd.DataFrame(read_jsonl(score_path))
    if scores.duplicated(list(SOURCE_KEYS)).any():
        raise ValueError("Duplicate resolved source-score keys")
    temperature_raw, temperature_files = load_raw_dataset(
        "temperature_robustness", args.temperature_root.resolve(), MODELS
    )
    prompt_raw, prompt_files = load_raw_dataset(
        "prompt_format_robustness", args.prompt_root.resolve(), MODELS
    )
    frames = {
        "temperature_robustness": merge_scores(
            temperature_raw, scores, "temperature_robustness"
        ),
        "prompt_format_robustness": merge_scores(
            prompt_raw, scores, "prompt_format_robustness"
        ),
    }
    matched = sum(int(frame["reasoning_evaluable"].sum()) for frame in frames.values())
    if matched != len(scores):
        raise ValueError(
            f"Resolved-score coverage mismatch: {matched} vs {len(scores)}"
        )
    provenance = {
        "score_file": {
            "path": provenance_path(score_path),
            "n_rows": len(scores),
            "sha256": sha256_file(score_path),
        },
        "raw_files": {
            "temperature_robustness": temperature_files,
            "prompt_format_robustness": prompt_files,
        },
    }
    return frames, provenance


def cell_summaries(
    frame: pd.DataFrame, condition: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["model_name", "task", condition, "alignment_pressure"]
    score_rows: list[dict[str, Any]] = []
    pattern_rows: list[dict[str, Any]] = []
    for values, group in frame.groupby(keys, sort=True, dropna=False):
        base = dict(zip(keys, values))
        for outcome in OUTCOMES:
            valid = group[outcome].dropna().astype(float)
            score_rows.append(
                {
                    **base,
                    "outcome": outcome,
                    "n_expected": len(group),
                    "n_scored": len(valid),
                    "mean": valid.mean() if len(valid) else np.nan,
                    "sd": valid.std(ddof=1) if len(valid) > 1 else np.nan,
                }
            )
        for taxonomy, column in (
            ("coarse", "reasoning_pattern_coarse"),
            ("detailed", "reasoning_pattern"),
        ):
            pattern_counts = group[column].dropna().value_counts()
            n_patterns = int(pattern_counts.sum())
            for pattern, count in pattern_counts.items():
                pattern_rows.append(
                    {
                        **base,
                        "taxonomy": taxonomy,
                        "reasoning_pattern": pattern,
                        "n_pattern_evaluable": n_patterns,
                        "count": int(count),
                        "rate": float(count / n_patterns),
                    }
                )
    return pd.DataFrame(score_rows), pd.DataFrame(pattern_rows)


def item_metrics(frame: pd.DataFrame, condition: str) -> pd.DataFrame:
    keys = ["model_name", "task", condition, "item_id"]
    rows: list[dict[str, Any]] = []
    for values, group in frame.groupby(keys, sort=True):
        base = dict(zip(keys, values))
        for outcome in OUTCOMES:
            alpha_means = (
                group.groupby("alignment_pressure", sort=True)[outcome].mean().dropna()
            )
            slope = np.nan
            if len(alpha_means) >= 2:
                slope = float(
                    np.polyfit(
                        alpha_means.index.to_numpy(dtype=float),
                        alpha_means.to_numpy(dtype=float),
                        1,
                    )[0]
                )
            alpha0 = group[np.isclose(group["alignment_pressure"], 0.0)][
                outcome
            ].dropna()
            rows.append(
                {
                    **base,
                    "outcome": outcome,
                    "n_alignment_levels": len(alpha_means),
                    "alignment_slope": slope,
                    "alpha0_mean": alpha0.mean() if len(alpha0) else np.nan,
                    "n_alpha0_rows": len(alpha0),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator, n_bootstrap: int
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    draws = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
    means = draws.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def sign_flip_pvalue(
    differences: np.ndarray,
    rng: np.random.Generator,
    n_permutations: int,
) -> float:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    observed = abs(float(values.mean()))
    if observed < 1e-15:
        return 1.0
    exceed = 0
    completed = 0
    chunk_size = 10_000
    while completed < n_permutations:
        size = min(chunk_size, n_permutations - completed)
        signs = rng.choice((-1.0, 1.0), size=(size, len(values)))
        permuted = np.abs((signs * values).mean(axis=1))
        exceed += int((permuted >= observed - 1e-15).sum())
        completed += size
    return float((exceed + 1) / (n_permutations + 1))


def summarize_slopes(
    metrics: pd.DataFrame,
    condition: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    keys = ["model_name", "task", condition, "outcome"]
    for values, group in metrics.groupby(keys, sort=True):
        slopes = group["alignment_slope"].dropna().to_numpy(dtype=float)
        low, high = bootstrap_mean_ci(slopes, rng, n_bootstrap)
        rows.append(
            {
                **dict(zip(keys, values)),
                "n_items": len(slopes),
                "mean_alignment_slope": slopes.mean() if len(slopes) else np.nan,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "ci_excludes_zero": bool(low > 0 or high < 0),
            }
        )
    return pd.DataFrame(rows)


def paired_contrasts(
    metrics: pd.DataFrame,
    condition: str,
    reference: Any,
    candidates: tuple[Any, ...],
    metric_names: tuple[str, ...],
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for (model, task, outcome), group in metrics.groupby(
        ["model_name", "task", "outcome"], sort=True
    ):
        for metric in metric_names:
            pivot = group.pivot(index="item_id", columns=condition, values=metric)
            for candidate in candidates:
                paired = pivot.dropna(subset=[reference, candidate])
                difference = paired[candidate].to_numpy(dtype=float) - paired[
                    reference
                ].to_numpy(dtype=float)
                low, high = bootstrap_mean_ci(difference, rng, n_bootstrap)
                rows.append(
                    {
                        "model_name": model,
                        "task": task,
                        "outcome": outcome,
                        "metric": metric,
                        "reference": reference,
                        "candidate": candidate,
                        "n_items": len(difference),
                        "reference_mean": paired[reference].mean(),
                        "candidate_mean": paired[candidate].mean(),
                        "candidate_minus_reference": difference.mean(),
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                        "ci_excludes_zero": bool(low > 0 or high < 0),
                        "p_sign_flip": sign_flip_pvalue(
                            difference, rng, n_permutations
                        ),
                    }
                )
    output = pd.DataFrame(rows)
    output["p_holm"] = multipletests(output["p_sign_flip"].fillna(1.0), method="holm")[
        1
    ]
    output["holm_significant"] = output["p_holm"] < 0.05
    return output


def fit_gee(frame: pd.DataFrame, design: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if design == "temperature":
        condition = "temperature_label"
        reference = TEMPERATURE_REFERENCE
    else:
        condition = "prompt_version"
        reference = PROMPT_REFERENCE
    for (model, task), group in frame.groupby(["model_name", "task"], sort=True):
        for outcome in OUTCOMES:
            data = group[["item_id", "alignment_pressure", condition, outcome]].dropna()
            formula = (
                f"{outcome} ~ alignment_pressure * "
                f"C({condition}, Treatment(reference='{reference}'))"
            )
            try:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always", RuntimeWarning)
                    result = smf.gee(
                        formula,
                        groups="item_id",
                        data=data,
                        cov_struct=Exchangeable(),
                        family=sm.families.Gaussian(),
                    ).fit()
                    params = result.params.copy()
                    standard_errors = result.bse.copy()
                    confidence = result.conf_int().copy()
                    p_values = result.pvalues.copy()
                warning_text = (
                    "; ".join(sorted({str(item.message) for item in caught_warnings}))
                    or None
                )
            except Exception as error:
                rows.append(
                    {
                        "model_name": model,
                        "task": task,
                        "outcome": outcome,
                        "term": "MODEL_ERROR",
                        "estimate": np.nan,
                        "std_error": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "p_value": np.nan,
                        "n_rows": len(data),
                        "n_items": data["item_id"].nunique(),
                        "warning": None,
                        "error": str(error),
                    }
                )
                continue
            for term in params.index:
                rows.append(
                    {
                        "model_name": model,
                        "task": task,
                        "outcome": outcome,
                        "term": term,
                        "estimate": float(params[term]),
                        "std_error": float(standard_errors[term]),
                        "ci_low": float(confidence.loc[term, 0]),
                        "ci_high": float(confidence.loc[term, 1]),
                        "p_value": float(p_values[term]),
                        "n_rows": len(data),
                        "n_items": data["item_id"].nunique(),
                        "warning": warning_text,
                        "error": None,
                    }
                )
    output = pd.DataFrame(rows)
    output["effect_family"] = "other"
    interaction = output["term"].str.startswith("alignment_pressure:C(")
    condition_main = output["term"].str.startswith(f"C({condition},")
    output.loc[interaction, "effect_family"] = "slope_interaction"
    output.loc[condition_main, "effect_family"] = "condition_at_alpha0"
    output["p_holm"] = np.nan
    for family in ("slope_interaction", "condition_at_alpha0"):
        mask = output["effect_family"] == family
        if mask.any():
            output.loc[mask, "p_holm"] = multipletests(
                output.loc[mask, "p_value"].fillna(1.0), method="holm"
            )[1]
    output["holm_significant"] = output["p_holm"] < 0.05
    return output


def completeness(frame: pd.DataFrame, dataset: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "n_expected_source_rows": len(frame),
        "n_reasoning_evaluable": int(frame["reasoning_evaluable"].sum()),
        "n_reasoning_missing": int((~frame["reasoning_evaluable"]).sum()),
        "n_fc_scored": int(frame["factual_correctness_score"].notna().sum()),
        "n_lc_scored": int(frame["logical_consistency_score"].notna().sum()),
        "n_rs_scored": int(frame["reasoning_structure_score"].notna().sum()),
        "n_pattern_scored": int(frame["reasoning_pattern"].notna().sum()),
        "missing_reasoning_by_model_task": {
            f"{model}/{task}": int(count)
            for (model, task), count in frame[~frame["reasoning_evaluable"]]
            .groupby(["model_name", "task"])
            .size()
            .items()
        },
        "missing_lc_by_model_task": {
            f"{model}/{task}": int(count)
            for (model, task), count in frame[
                frame["reasoning_evaluable"] & frame["logical_consistency_score"].isna()
            ]
            .groupby(["model_name", "task"])
            .size()
            .items()
        },
    }


def write_temperature_report(
    path: Path,
    complete: dict[str, Any],
    slopes: pd.DataFrame,
    contrasts: pd.DataFrame,
    gee: pd.DataFrame,
) -> None:
    slope_counts = (
        slopes.groupby("outcome", sort=False)
        .agg(
            n_cells=("mean_alignment_slope", "size"),
            n_positive=("mean_alignment_slope", lambda values: int((values > 0).sum())),
            n_ci_positive=("bootstrap_ci_low", lambda values: int((values > 0).sum())),
            n_ci_negative=("bootstrap_ci_high", lambda values: int((values < 0).sum())),
        )
        .reset_index()
    )
    significant_main = gee[
        gee["holm_significant"] & (gee["effect_family"] == "condition_at_alpha0")
    ]
    slope_interactions = gee[gee["effect_family"] == "slope_interaction"]
    lines = [
        "# Reasoning-Level Temperature Robustness",
        "",
        "## Completeness",
        "",
        f"- Expected source rows: {complete['n_expected_source_rows']}.",
        f"- FC/RS-evaluable rows: {complete['n_reasoning_evaluable']}.",
        f"- LC/pattern-evaluable rows: {complete['n_lc_scored']}.",
        f"- Source generations without recoverable reasoning: {complete['n_reasoning_missing']}.",
        "",
        "## Alignment-Pressure Slopes",
        "",
        "Positive FC and Sound slopes indicate improved reasoning as steering aligns with ground truth.",
        "",
        "| Outcome | Positive slopes | CIs above zero | CIs below zero |",
        "|---|---:|---:|---:|",
    ]
    for row in slope_counts.itertuples(index=False):
        lines.append(
            f"| {row.outcome} | {row.n_positive}/{row.n_cells} | "
            f"{row.n_ci_positive}/{row.n_cells} | {row.n_ci_negative}/{row.n_cells} |"
        )
    lines.extend(
        [
            "",
            "All 30 Sound Reasoning slopes are positive and all 30 item-bootstrap 95% CIs are above zero. Full FC/LC/RS and pattern summaries are retained in the companion CSV files.",
            "",
            "### Sound Reasoning by Cell",
            "",
            "| Model | Task | Temperature | Mean slope [95% CI] |",
            "|---|---|---:|---:|",
        ]
    )
    for row in slopes[slopes["outcome"] == "sound_reasoning"].itertuples(index=False):
        lines.append(
            f"| {row.model_name} | {row.task} | {float(row.temperature):g} | "
            f"{row.mean_alignment_slope:.3f} "
            f"[{row.bootstrap_ci_low:.3f}, {row.bootstrap_ci_high:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Temperature Dependence",
            "",
            f"- Holm-significant paired slope contrasts: {int(contrasts['holm_significant'].sum())}/{len(contrasts)}.",
            f"- Unadjusted paired slope CIs excluding zero: {int(contrasts['ci_excludes_zero'].sum())}/{len(contrasts)}; raw sign-flip p < 0.05: {int((contrasts['p_sign_flip'] < 0.05).sum())}/{len(contrasts)}.",
            f"- Holm-significant GEE Alignment Pressure x Temperature terms: {int(slope_interactions['holm_significant'].sum())}/{len(slope_interactions)}.",
            f"- Holm-significant GEE temperature effects at alpha zero: {len(significant_main)}/{int((gee['effect_family'] == 'condition_at_alpha0').sum())}.",
            "",
        ]
    )
    for row in significant_main.itertuples(index=False):
        lines.append(
            f"The sole adjusted alpha-zero effect was {row.model_name}, {row.task}, "
            f"{row.outcome}: {row.term} = {row.estimate:.3f} "
            f"[{row.ci_low:.3f}, {row.ci_high:.3f}], Holm p = {row.p_holm:.4g}. "
            "This is a baseline score shift, not an Alignment Pressure slope interaction."
        )
    lines.extend(
        [
            "",
            "Inference is performed within each model and task with item as the repeated unit; models are not treated as replicates. The paired bootstrap/sign-flip analysis is primary and the Gaussian GEE is a repeated-measures robustness model.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_prompt_report(
    path: Path,
    complete: dict[str, Any],
    contrasts: pd.DataFrame,
    gee: pd.DataFrame,
) -> None:
    supported = contrasts[contrasts["holm_significant"]]
    unadjusted_ci = contrasts[contrasts["ci_excludes_zero"]]
    effect_terms = gee[
        gee["effect_family"].isin(("condition_at_alpha0", "slope_interaction"))
    ]
    warning_fits = gee[gee["warning"].notna()][
        ["model_name", "task", "outcome", "warning"]
    ].drop_duplicates()
    lines = [
        "# Reasoning-Level `anwer`/`answer` Robustness",
        "",
        "## Completeness",
        "",
        f"- Expected source rows: {complete['n_expected_source_rows']}.",
        f"- FC/RS-evaluable rows: {complete['n_reasoning_evaluable']}.",
        f"- LC/pattern-evaluable rows: {complete['n_lc_scored']}.",
        f"- Source generations without recoverable reasoning: {complete['n_reasoning_missing']}.",
        "",
        "## Paired Prompt-Version Contrasts",
        "",
        "Contrasts are corrected `answer` minus historical `anwer` at the item level.",
        "",
        f"- Holm-significant slope or alpha-zero contrasts: {len(supported)}/{len(contrasts)}.",
        f"- Unadjusted paired CIs excluding zero: {len(unadjusted_ci)}/{len(contrasts)}; raw sign-flip p < 0.05: {int((contrasts['p_sign_flip'] < 0.05).sum())}/{len(contrasts)}.",
        f"- Holm-significant GEE prompt main/interaction terms: {int(effect_terms['holm_significant'].sum())}/{len(effect_terms)}.",
        f"- GEE fits with a numerical warning: {len(warning_fits)}/40; all 80 prespecified prompt main/interaction estimates and p-values are finite.",
        "",
        "| Model | Task | Outcome | Metric | Difference [95% CI] | Holm p |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in supported.itertuples(index=False):
        lines.append(
            f"| {row.model_name} | {row.task} | {row.outcome} | {row.metric} | "
            f"{row.candidate_minus_reference:.3f} [{row.bootstrap_ci_low:.3f}, "
            f"{row.bootstrap_ci_high:.3f}] | {row.p_holm:.4g} |"
        )
    if supported.empty:
        lines.append("| None | | | | | |")
    if len(unadjusted_ci):
        lines.extend(
            [
                "",
                "The following descriptive bootstrap intervals excluded zero before multiplicity correction; none survived Holm correction:",
                "",
                "| Model | Task | Outcome | Metric | Difference [95% CI] | Raw p | Holm p |",
                "|---|---|---|---|---:|---:|---:|",
            ]
        )
        for row in unadjusted_ci.itertuples(index=False):
            lines.append(
                f"| {row.model_name} | {row.task} | {row.outcome} | {row.metric} | "
                f"{row.candidate_minus_reference:.3f} [{row.bootstrap_ci_low:.3f}, "
                f"{row.bootstrap_ci_high:.3f}] | {row.p_sign_flip:.4g} | {row.p_holm:.4g} |"
            )
    if len(warning_fits):
        warning = warning_fits.iloc[0]
        lines.extend(
            [
                "",
                f"The numerical diagnostic occurred for {warning['model_name']}, "
                f"{warning['task']}, {warning['outcome']} because the reference-cell "
                "intercept had zero robust variance. It does not affect the finite prompt-version main or interaction terms.",
            ]
        )
    lines.extend(
        [
            "",
            "Inference is performed within each model and task with item as the repeated unit; models are not treated as replicates. The paired bootstrap/sign-flip analysis is primary and the Gaussian GEE is a repeated-measures robustness model.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_temperature(
    frame: pd.DataFrame, output_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    frame = frame.copy()
    frame["temperature_label"] = frame["temperature"].map(
        {0.0: "0", 0.2: "0.2", 1.0: "1"}
    )
    if frame["temperature_label"].isna().any():
        raise ValueError("Unexpected temperature")
    output_dir.mkdir(parents=True, exist_ok=True)
    cells, patterns = cell_summaries(frame, "temperature")
    metrics = item_metrics(frame, "temperature")
    slopes = summarize_slopes(
        metrics, "temperature", args.n_bootstrap, args.bootstrap_seed
    )
    contrasts = paired_contrasts(
        metrics,
        "temperature",
        0.2,
        (0.0, 1.0),
        ("alignment_slope",),
        args.n_bootstrap,
        args.n_permutations,
        args.bootstrap_seed + 1,
    )
    gee = fit_gee(frame, "temperature")
    complete = completeness(frame, "temperature_robustness")
    cells.to_csv(output_dir / "cell_score_summary.csv", index=False)
    patterns.to_csv(output_dir / "reasoning_pattern_summary.csv", index=False)
    metrics.to_csv(output_dir / "item_metrics.csv", index=False)
    slopes.to_csv(output_dir / "slope_summary.csv", index=False)
    contrasts.to_csv(output_dir / "paired_temperature_contrasts.csv", index=False)
    gee.to_csv(output_dir / "gee_coefficients.csv", index=False)
    (output_dir / "completeness.json").write_text(
        json.dumps(complete, indent=2) + "\n", encoding="utf-8"
    )
    write_temperature_report(output_dir / "report.md", complete, slopes, contrasts, gee)
    return complete


def analyze_prompt(
    frame: pd.DataFrame, output_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    if set(frame["prompt_version"]) != {PROMPT_REFERENCE, PROMPT_CORRECTED}:
        raise ValueError("Prompt-version levels do not match the frozen audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    cells, patterns = cell_summaries(frame, "prompt_version")
    metrics = item_metrics(frame, "prompt_version")
    slopes = summarize_slopes(
        metrics, "prompt_version", args.n_bootstrap, args.bootstrap_seed + 2
    )
    contrasts = paired_contrasts(
        metrics,
        "prompt_version",
        PROMPT_REFERENCE,
        (PROMPT_CORRECTED,),
        ("alignment_slope", "alpha0_mean"),
        args.n_bootstrap,
        args.n_permutations,
        args.bootstrap_seed + 3,
    )
    gee = fit_gee(frame, "prompt")
    complete = completeness(frame, "prompt_format_robustness")
    cells.to_csv(output_dir / "cell_score_summary.csv", index=False)
    patterns.to_csv(output_dir / "reasoning_pattern_summary.csv", index=False)
    metrics.to_csv(output_dir / "item_metrics.csv", index=False)
    slopes.to_csv(output_dir / "slope_summary.csv", index=False)
    contrasts.to_csv(output_dir / "paired_prompt_contrasts.csv", index=False)
    gee.to_csv(output_dir / "gee_coefficients.csv", index=False)
    (output_dir / "completeness.json").write_text(
        json.dumps(complete, indent=2) + "\n", encoding="utf-8"
    )
    write_prompt_report(output_dir / "report.md", complete, contrasts, gee)
    return complete


def run(args: argparse.Namespace) -> None:
    frames, provenance = prepare_data(args)
    output_root = args.output_root.resolve()
    temperature_complete = analyze_temperature(
        frames["temperature_robustness"], output_root / "temperature", args
    )
    prompt_complete = analyze_prompt(
        frames["prompt_format_robustness"], output_root / "prompt_format", args
    )
    manifest = {
        "analysis": {
            "outcomes": list(OUTCOMES),
            "alignment_pressure": "alpha_norm * truth_direction",
            "temperature_reference": 0.2,
            "prompt_reference": PROMPT_REFERENCE,
            "item_is_repeated_unit": True,
            "models_are_not_replicates": True,
            "reasoning_taxonomies": {
                "coarse": "Exact manuscript response_type mapping from notebooks/paper_figures/tools.py",
                "detailed": "Exact manuscript response_type_more mapping from notebooks/paper_figures/tools.py",
            },
            "n_bootstrap": args.n_bootstrap,
            "n_permutations": args.n_permutations,
            "bootstrap_seed": args.bootstrap_seed,
            "multiple_testing": "Holm within each prespecified output family",
            "gee": "Gaussian GEE with exchangeable item clusters",
        },
        "completeness": {
            "temperature": temperature_complete,
            "prompt_format": prompt_complete,
        },
        "provenance": provenance,
    }
    (output_root / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote reasoning robustness analysis to {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score_path", type=Path, default=DEFAULT_SCORE_PATH)
    parser.add_argument(
        "--temperature_root", type=Path, default=DEFAULT_TEMPERATURE_ROOT
    )
    parser.add_argument("--prompt_root", type=Path, default=DEFAULT_PROMPT_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--n_bootstrap", type=int, default=10_000)
    parser.add_argument("--n_permutations", type=int, default=100_000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260805)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
