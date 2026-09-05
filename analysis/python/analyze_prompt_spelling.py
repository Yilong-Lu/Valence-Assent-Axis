#!/usr/bin/env python
"""Analyse the paired legacy-typo versus corrected-prompt robustness check."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


def _preload_conda_libstdcxx() -> None:
    candidates: list[Path] = []
    if os.environ.get("CONDA_PREFIX"):
        candidates.append(Path(os.environ["CONDA_PREFIX"]) / "lib" / "libstdc++.so.6")
    candidates.append(Path(sys.executable).resolve().parents[1] / "lib" / "libstdc++.so.6")
    for path in candidates:
        if not path.exists():
            continue
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue


_preload_conda_libstdcxx()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
ARCHIVE_RESULT_ROOT = (
    REPOSITORY_ROOT
    / "data"
    / "raw"
    / "model_outputs"
    / "prompt_spelling_check"
)
STAGING_RESULT_ROOT = REPOSITORY_ROOT / "data" / "raw_external" / "prompt_spelling"
DEFAULT_RESULT_ROOT = (
    ARCHIVE_RESULT_ROOT if ARCHIVE_RESULT_ROOT.is_dir() else STAGING_RESULT_ROOT
)
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "results" / "summaries" / "generation_robustness" / "prompt_spelling"
)
VERSIONS = ("legacy_anwer_v1", "corrected_answer_v2")
ANSWER_FIELD_PATTERN = re.compile(
    r"""["']answer["']\s*:\s*["'](right|wrong|yes|no)["']""",
    flags=re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def load_model_results(result_root: Path, model_name: str) -> list[dict[str, Any]]:
    model_dir = result_root / model_name
    metadata_path = model_dir / "metadata.json"
    result_path = model_dir / "raw_results.jsonl"
    if not metadata_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(f"Incomplete result directory: {model_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise ValueError(f"Run is not complete: {model_dir}")
    rows = read_jsonl(result_path)
    if len(rows) != int(metadata["n_rows"]):
        raise ValueError(
            f"Row-count mismatch for {model_name}: "
            f"{len(rows)} vs {metadata['n_rows']}"
        )
    return rows


def prepare_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    required = {
        "model_name",
        "task",
        "item_id",
        "prompt_version",
        "alpha_norm",
        "seed",
        "strict_json_valid",
        "answer_valid",
        "correct",
        "positive_response",
        "generated_n_tokens",
        "generation_stop_reason",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")
    frame["strict_valid_numeric"] = frame["strict_json_valid"].astype(float)
    relaxed_answers = frame["generated_text"].map(extract_answer_field)
    frame["analysis_answer_canonical"] = relaxed_answers
    frame["answer_valid_numeric"] = relaxed_answers.notna().astype(float)
    frame["correct_valid"] = (
        relaxed_answers == frame["correct_answer"].str.lower()
    ).where(relaxed_answers.notna())
    frame["correct_all"] = frame["correct_valid"].eq(True).astype(float)
    frame["positive_numeric"] = relaxed_answers.map(
        {"right": 1.0, "wrong": 0.0, "yes": 1.0, "no": 0.0}
    )
    frame["truncated"] = (
        frame["generation_stop_reason"] == "max_new_tokens"
    ).astype(float)
    return frame


def extract_answer_field(text: str) -> str | None:
    matches = ANSWER_FIELD_PATTERN.findall(str(text))
    canonical = {match.lower() for match in matches}
    if len(canonical) != 1:
        return None
    return next(iter(canonical))


def summarize_cells(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_name", "task", "prompt_version", "alpha_norm"]
    summaries: list[dict[str, Any]] = []
    for values, group in frame.groupby(keys, sort=True):
        summaries.append(
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
    return pd.DataFrame(summaries)


def fit_item_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_name", "task", "prompt_version", "item_id"]
    metrics: list[dict[str, Any]] = []
    for values, group in frame.groupby(keys, sort=True):
        alpha_means = (
            group.groupby("alpha_norm", sort=True)["positive_numeric"]
            .mean()
            .dropna()
        )
        slope = np.nan
        if len(alpha_means) >= 2:
            slope = float(
                np.polyfit(
                    alpha_means.index.to_numpy(dtype=float),
                    alpha_means.to_numpy(dtype=float),
                    deg=1,
                )[0]
            )
        alpha0 = group[np.isclose(group["alpha_norm"], 0.0)]
        metrics.append(
            {
                **dict(zip(keys, values)),
                "positive_slope": slope,
                "alpha0_accuracy_all": alpha0["correct_all"].mean(),
                "alpha0_strict_json_rate": alpha0["strict_valid_numeric"].mean(),
                "alpha0_answer_parse_rate": alpha0["answer_valid_numeric"].mean(),
                "alpha0_mean_generated_tokens": alpha0[
                    "generated_n_tokens"
                ].mean(),
            }
        )
    return pd.DataFrame(metrics)


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float]:
    if len(values) == 0:
        return np.nan, np.nan
    draws = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
    means = draws.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_version_contrasts(
    item_metrics: pd.DataFrame,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    metric_names = (
        "positive_slope",
        "alpha0_accuracy_all",
        "alpha0_strict_json_rate",
        "alpha0_answer_parse_rate",
        "alpha0_mean_generated_tokens",
    )
    rng = np.random.default_rng(bootstrap_seed)
    results: list[dict[str, Any]] = []
    for (model_name, task), group in item_metrics.groupby(
        ["model_name", "task"], sort=True
    ):
        for metric in metric_names:
            pivot = group.pivot(
                index="item_id", columns="prompt_version", values=metric
            )
            pivot = pivot.dropna(subset=list(VERSIONS))
            legacy = pivot[VERSIONS[0]].to_numpy(dtype=float)
            corrected = pivot[VERSIONS[1]].to_numpy(dtype=float)
            difference = corrected - legacy
            ci_low, ci_high = bootstrap_mean_ci(
                difference, rng, n_bootstrap
            )
            results.append(
                {
                    "model_name": model_name,
                    "task": task,
                    "metric": metric,
                    "n_items": len(difference),
                    "legacy_mean": legacy.mean() if len(legacy) else np.nan,
                    "corrected_mean": corrected.mean() if len(corrected) else np.nan,
                    "corrected_minus_legacy": (
                        difference.mean() if len(difference) else np.nan
                    ),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                }
            )
    return pd.DataFrame(results)


def material_effect_flags(contrasts: pd.DataFrame) -> pd.DataFrame:
    output = contrasts.copy()
    supported = (output["bootstrap_ci_low"] > 0) | (
        output["bootstrap_ci_high"] < 0
    )
    output["ci_excludes_zero"] = supported
    output["material_threshold"] = np.select(
        [
            output["metric"].isin(
                [
                    "alpha0_accuracy_all",
                    "alpha0_strict_json_rate",
                    "alpha0_answer_parse_rate",
                ]
            ),
            output["metric"] == "positive_slope",
        ],
        [0.05, 0.20 * output["legacy_mean"].abs().clip(lower=0.10)],
        default=np.inf,
    )
    output["material_supported_change"] = (
        supported
        & (
            output["corrected_minus_legacy"].abs()
            > output["material_threshold"]
        )
    )
    return output


def format_number(value: float, digits: int = 3) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def write_report(
    output_path: Path,
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> None:
    model_names = ", ".join(sorted(frame["model_name"].astype(str).unique()))
    lines = [
        "# Prompt Spelling Robustness",
        "",
        "## Design",
        "",
        f"- Models: {model_names}.",
        "- Tasks: 30 balanced Alphabetical Think-then-Answer statements and the existing 30-item TruthfulQA set.",
        "- Prompt versions differ only by `anwer` versus `answer`.",
        "- Temperature: 0.2; seeds: 0, 1, 2.",
        "- Normalized steering alpha: -0.2, 0, +0.2, mapped separately to each direction's calibrated raw range.",
        "- Strict JSON validity is reported separately. Behavioral endpoints use the manuscript scoring rule, which conservatively extracts an unambiguous quoted `answer` field because models often emit unescaped newlines inside the JSON-like `think` string.",
        "- Missing or conflicting answer fields count as incorrect for `accuracy_all`; valid-only accuracy is also retained.",
        "",
        "## Completeness",
        "",
        f"- Total rows: {len(frame)}.",
        f"- Models: {frame['model_name'].nunique()}; tasks: {frame['task'].nunique()}; items: {frame.groupby(['model_name', 'task'])['item_id'].nunique().min()} per model-task.",
        f"- Overall strict JSON rate: {frame['strict_valid_numeric'].mean():.3f}.",
        f"- Overall answer parse rate: {frame['answer_valid_numeric'].mean():.3f}.",
        f"- Overall truncation rate: {frame['truncated'].mean():.3f}.",
        "",
        "## Paired Version Contrasts",
        "",
        "Values are corrected minus legacy; 95% CIs use item bootstrap resampling.",
        "",
        "| Model | Task | Metric | Legacy | Corrected | Difference [95% CI] | Material? |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    selected_metrics = {
        "positive_slope": "steering slope",
        "alpha0_accuracy_all": "alpha=0 accuracy",
        "alpha0_answer_parse_rate": "alpha=0 parse rate",
        "alpha0_strict_json_rate": "alpha=0 strict JSON",
    }
    for _, row in contrasts[
        contrasts["metric"].isin(selected_metrics)
    ].iterrows():
        lines.append(
            "| {model} | {task} | {metric} | {legacy} | {corrected} | "
            "{delta} [{low}, {high}] | {material} |".format(
                model=row["model_name"],
                task=row["task"],
                metric=selected_metrics[row["metric"]],
                legacy=format_number(float(row["legacy_mean"])),
                corrected=format_number(float(row["corrected_mean"])),
                delta=format_number(float(row["corrected_minus_legacy"])),
                low=format_number(float(row["bootstrap_ci_low"])),
                high=format_number(float(row["bootstrap_ci_high"])),
                material="yes" if row["material_supported_change"] else "no",
            )
        )
    material = contrasts["material_supported_change"].any()
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                "The compact check detected at least one supported material prompt-version change. "
                "Expand only the affected task before manuscript integration."
                if material
                else "No supported material prompt-version change was detected in the registered model set. Retain the historical prompts for reproducibility and disclose the typo."
            ),
            "",
            "The automatic decision is a screening rule, not a replacement for inspecting the cell summaries and raw completions.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    for model_name in args.models:
        rows.extend(load_model_results(args.result_root.resolve(), model_name))
    frame = prepare_dataframe(rows)
    expected_versions = set(VERSIONS)
    if set(frame["prompt_version"]) != expected_versions:
        raise ValueError(
            f"Prompt-version mismatch: {sorted(frame['prompt_version'].unique())}"
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_cells(frame)
    item_metrics = fit_item_metrics(frame)
    contrasts = paired_version_contrasts(
        item_metrics, args.n_bootstrap, args.bootstrap_seed
    )
    contrasts = material_effect_flags(contrasts)
    summary.to_csv(output_dir / "summary_by_cell.csv", index=False)
    item_metrics.to_csv(output_dir / "item_level_metrics.csv", index=False)
    contrasts.to_csv(output_dir / "paired_version_contrasts.csv", index=False)
    write_report(
        output_dir / "prompt_spelling_report.md",
        frame,
        summary,
        contrasts,
    )
    print(f"Wrote prompt-format analysis to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--models", nargs="+", default=("qwen25_7b", "llama3_8b")
    )
    parser.add_argument("--n_bootstrap", type=int, default=10_000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260723)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
