"""Reproduce human-expert and LLM-judge agreement statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from vaa.config import REPOSITORY_ROOT


DEFAULT_DATA_DIR = REPOSITORY_ROOT / "data" / "judge_validation"
REASONING_JUDGES = {
    "DeepSeek V3": "deepseek_v3",
    "DeepSeek R1": "deepseek_r1",
    "Qwen3-30B-A3B-Instruct": "qwen3_32b",
}
SCORE_TYPES = ("FC", "LC", "RS")


def icc_2_1(values: np.ndarray) -> float:
    """Two-way random-effects, absolute-agreement, single-measure ICC."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("ICC input must contain at least two targets and raters")
    n_targets, n_raters = values.shape
    grand_mean = values.mean()
    target_means = values.mean(axis=1)
    rater_means = values.mean(axis=0)
    ms_targets = (
        n_raters * np.square(target_means - grand_mean).sum() / (n_targets - 1)
    )
    ms_raters = (
        n_targets * np.square(rater_means - grand_mean).sum() / (n_raters - 1)
    )
    residual = values - target_means[:, None] - rater_means[None, :] + grand_mean
    ms_error = np.square(residual).sum() / (
        (n_targets - 1) * (n_raters - 1)
    )
    denominator = (
        ms_targets
        + (n_raters - 1) * ms_error
        + n_raters * (ms_raters - ms_error) / n_targets
    )
    return float((ms_targets - ms_error) / denominator)


def reasoning_expert_reliability(data_dir: Path) -> list[dict[str, Any]]:
    experts = pd.read_csv(data_dir / "evaluated_results.csv")
    results = []
    for score_type in SCORE_TYPES:
        first = experts[f"{score_type}_Score1"]
        second = experts[f"{score_type}_Score2"]
        agreement = first == second
        results.append(
            {
                "score_type": score_type,
                "n_rated": int(len(experts)),
                "n_agreement": int(agreement.sum()),
                "n_disagreement_excluded": int((~agreement).sum()),
                "exact_percent_agreement": float(agreement.mean()),
                "quadratic_weighted_kappa": float(
                    cohen_kappa_score(first, second, weights="quadratic")
                ),
                "unweighted_kappa": float(cohen_kappa_score(first, second)),
            }
        )
    return results


def reasoning_judge_validation(data_dir: Path) -> list[dict[str, Any]]:
    experts = pd.read_csv(data_dir / "evaluated_results.csv")
    score_columns = {
        "FC": "factual_correctness_score",
        "LC": "logical_consistency_score",
        "RS": "reasoning_structure_score",
    }
    results = []
    for display_name, file_key in REASONING_JUDGES.items():
        path = data_dir / f"eval_sample_sample_models_{file_key}.json"
        judged = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
        judged = judged.sort_values("i", ignore_index=True)
        if not judged["question"].equals(experts["question"]):
            raise ValueError(f"Question order mismatch for {display_name}")
        if not judged["answer"].equals(experts["answer"]):
            raise ValueError(f"Answer order mismatch for {display_name}")
        for score_type in SCORE_TYPES:
            first = experts[f"{score_type}_Score1"]
            second = experts[f"{score_type}_Score2"]
            consensus = first.where(first == second)
            model_scores = judged[score_columns[score_type]]
            mask = consensus.notna() & model_scores.notna()
            results.append(
                {
                    "judge": display_name,
                    "score_type": score_type,
                    "n_consensus": int(mask.sum()),
                    "quadratic_weighted_kappa": float(
                        cohen_kappa_score(
                            consensus[mask],
                            model_scores[mask],
                            weights="quadratic",
                        )
                    ),
                    "percent_agreement": float(
                        (consensus[mask] == model_scores[mask]).mean()
                    ),
                }
            )
    return results


def stance_judge_validation(data_dir: Path) -> list[dict[str, Any]]:
    results = []
    for measure, filename in (
        ("final_stance", "answer_stance.csv"),
        ("reasoning_stance", "think_stance.csv"),
    ):
        frame = pd.read_csv(data_dir / filename)
        expert_mean = frame[["S1", "S2"]].mean(axis=1).to_numpy()
        results.append(
            {
                "measure": measure,
                "judge": "Human experts",
                "n": int(len(frame)),
                "icc_2_1": icc_2_1(frame[["S1", "S2"]].to_numpy()),
            }
        )
        for display_name, column in REASONING_JUDGES.items():
            results.append(
                {
                    "measure": measure,
                    "judge": display_name,
                    "n": int(len(frame)),
                    "icc_2_1": icc_2_1(
                        np.column_stack([expert_mean, frame[column].to_numpy()])
                    ),
                }
            )
    return results


def build_report(data_dir: Path) -> dict[str, Any]:
    return {
        "reasoning_expert_reliability": reasoning_expert_reliability(data_dir),
        "reasoning_judge_validation": reasoning_judge_validation(data_dir),
        "stance_judge_validation": stance_judge_validation(data_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.data_dir)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
