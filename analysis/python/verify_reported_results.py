#!/usr/bin/env python3
"""Verify the analysis data and numerical conclusions used in the manuscript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ALL_MODELS = {
    "qwen25_3b",
    "qwen25_7b",
    "qwen25_14b",
    "qwen25_32b",
    "qwen25_72b",
    "llama3_8b",
    "mistral_7b",
    "gemma2_9b",
}
ROBUSTNESS_MODELS = {
    "qwen25_7b",
    "qwen25_14b",
    "qwen25_32b",
    "qwen25_72b",
    "llama3_8b",
}


def read_csv(relative_path: str) -> pd.DataFrame:
    return pd.read_csv(REPOSITORY_ROOT / relative_path)


def assert_models(frame: pd.DataFrame, expected: set[str]) -> None:
    observed = set(frame["model_name"].dropna().unique())
    if observed != expected:
        raise AssertionError(f"Model coverage mismatch: {observed} != {expected}")


def validate_manifest() -> dict[str, object]:
    manifest = yaml.safe_load(
        (REPOSITORY_ROOT / "manifest/results.yaml").read_text(encoding="utf-8")
    )
    for record in manifest["tracked_analysis_data"].values():
        if "path" in record:
            frame = read_csv(record["path"])
            if len(frame) != record["rows"]:
                raise AssertionError(f"Row count mismatch for {record['path']}")
        if "pattern" in record:
            files = sorted(REPOSITORY_ROOT.glob(record["pattern"]))
            if len(files) != record["files"]:
                raise AssertionError(f"File count mismatch for {record['pattern']}")
            if "rows_per_file" in record:
                for path in files:
                    if len(pd.read_csv(path)) != record["rows_per_file"]:
                        raise AssertionError(f"Row count mismatch for {path}")
    return manifest


def verify_answer_labels() -> dict[str, float]:
    frame = read_csv("data/processed/answer_label_control/model_summary.csv")
    assert_models(frame, ALL_MODELS)
    if set(frame["answer_pair"]) != {"right_wrong", "true_false"}:
        raise AssertionError("Answer-label conditions are incomplete")
    if not (frame.groupby("model_name").size() == 2).all():
        raise AssertionError("Each model must have both answer-label conditions")
    minimum = float(frame["abs_pc1_d_projection_r"].min())
    if minimum <= 0.70:
        raise AssertionError("Single-Letter Order projection alignment is below 0.70")
    qwen14_true_false = frame[
        (frame["model_name"] == "qwen25_14b")
        & (frame["answer_pair"] == "true_false")
    ]["abs_pc1_d_projection_r"].iloc[0]
    if not np.isclose(qwen14_true_false, 0.9933774, atol=1e-6):
        raise AssertionError("Qwen2.5-14B true/false projection result changed")
    return {
        "minimum_absolute_projection_correlation": minimum,
        "qwen25_14b_true_false_projection_correlation": float(qwen14_true_false),
    }


def verify_subjective_preference() -> dict[str, float]:
    trends = read_csv(
        "results/summaries/subjective_preference/semantic_valence_marginal_trends.csv"
    )
    primary = trends[
        (trends["score_mode"] == "sequence")
        & (trends["valence_status"] == "valenced")
    ]
    assert_models(primary, ALL_MODELS)
    if not (primary["estimate"] > 0).all() or not (primary["ci_low"] > 0).all():
        raise AssertionError("Valenced semantic slopes are not uniformly positive")

    contrasts = read_csv(
        "results/summaries/subjective_preference/planned_slope_contrasts.csv"
    )
    valence = contrasts[
        (contrasts["score_mode"] == "sequence")
        & (contrasts["component"] == "semantic")
        & (contrasts["contrast_family"] == "valenced_minus_neutral_within_opposition")
    ]
    if len(valence) != 16 or not (valence["ci_low"] > 0).all():
        raise AssertionError("Valence-specific semantic contrasts are incomplete")
    qwen14_slope = primary[primary["model_name"] == "qwen25_14b"]["estimate"].iloc[0]
    if not np.isclose(qwen14_slope, 9.8479848, atol=1e-6):
        raise AssertionError("Qwen2.5-14B valenced semantic slope changed")
    return {
        "minimum_valenced_semantic_slope": float(primary["estimate"].min()),
        "minimum_valence_contrast_ci_low": float(valence["ci_low"].min()),
        "qwen25_14b_valenced_semantic_slope": float(qwen14_slope),
    }


def verify_feedback_sycophancy() -> dict[str, float]:
    state = read_csv(
        "results/summaries/feedback_induced_sycophancy/feedback_state_contrasts.csv"
    )
    verdict = read_csv(
        "results/summaries/feedback_induced_sycophancy/feedback_verdict_contrasts.csv"
    )
    state_difference = state[state["contrast"] == "user_like_minus_user_dislike"]
    verdict_difference = verdict[
        verdict["contrast"] == "user_like_minus_user_dislike"
    ]
    assert_models(state_difference, ALL_MODELS)
    assert_models(verdict_difference, ALL_MODELS)
    if not (state_difference["ci_low"] > 0).all():
        raise AssertionError("Feedback-induced VAA-state differences are not uniform")
    if not (verdict_difference["probability_ci_low"] > 0).all():
        raise AssertionError("Feedback-induced verdict differences are not uniform")

    slopes = read_csv(
        "results/summaries/feedback_induced_sycophancy/intervention_condition_slopes.csv"
    )
    assert_models(slopes, ALL_MODELS)
    if len(slopes) != 24 or not (slopes["ci_low"] > 0).all():
        raise AssertionError("VAA intervention slopes are not positive in all conditions")

    raw = read_csv("data/processed/feedback_induced_sycophancy/feedback_effect.csv")
    qwen14 = raw[
        (raw["model_name"] == "qwen25_14b")
        & (raw["condition"] == "baseline")
        & raw["verdict_valid"].astype(bool)
    ]
    baseline_rate = float(qwen14["verdict_strong"].mean())
    if not np.isclose(baseline_rate, 0.3614864865, atol=1e-10):
        raise AssertionError("Qwen2.5-14B baseline Strong-verdict rate changed")
    qwen14_like = verdict[
        (verdict["model_name"] == "qwen25_14b")
        & (verdict["contrast"] == "user_like_minus_baseline")
    ]["probability_difference"].iloc[0]
    if not np.isclose(qwen14_like, 0.0945945946, atol=1e-10):
        raise AssertionError("Qwen2.5-14B user-like verdict contrast changed")
    return {
        "qwen25_14b_baseline_strong_rate": baseline_rate,
        "qwen25_14b_user_like_verdict_difference": float(qwen14_like),
        "minimum_state_like_dislike_ci_low": float(state_difference["ci_low"].min()),
        "minimum_verdict_like_dislike_ci_low": float(
            verdict_difference["probability_ci_low"].min()
        ),
    }


def verify_arithmetic() -> dict[str, float]:
    frame = read_csv(
        "results/summaries/arithmetic_answering_verification/accuracy_slope_condition_estimates.csv"
    )
    assert_models(frame, ALL_MODELS)
    if len(frame) != 24:
        raise AssertionError("Arithmetic condition estimates are incomplete")
    qwen14 = frame[frame["model_name"] == "qwen25_14b"].set_index("condition")
    direct = qwen14.loc["direct_numeric"]
    true = qwen14.loc["verification_true"]
    false = qwen14.loc["verification_false"]
    if not (direct["ci_low"] <= 0 <= direct["ci_high"]):
        raise AssertionError("Qwen2.5-14B direct-answer accuracy is not stable")
    if true["ci_low"] <= 0 or false["ci_high"] >= 0:
        raise AssertionError("Verification accuracy slopes have unexpected directions")
    return {
        "qwen25_14b_direct_accuracy_slope": float(direct["estimate"]),
        "qwen25_14b_true_statement_slope": float(true["estimate"]),
        "qwen25_14b_false_statement_slope": float(false["estimate"]),
    }


def verify_generality_and_robustness() -> dict[str, float | int]:
    alignment = read_csv("data/processed/cross_model/cross_model_layer_alignment.csv")
    target = alignment[alignment["is_target_layer"].astype(str).str.lower() == "true"]
    assert_models(target, ALL_MODELS)
    minimum_alignment = float(target["alignment"].min())
    maximum_alignment = float(target["alignment"].max())
    if not np.isclose(minimum_alignment, 0.703216, atol=1e-6):
        raise AssertionError("Minimum selected-layer alignment changed")

    cross_domain = read_csv(
        "data/processed/cross_model/cross_model_intervention_coefficients.csv"
    )
    assert_models(cross_domain, ALL_MODELS)
    if len(cross_domain) != 16 or not (cross_domain["p_value"] < 0.001).all():
        raise AssertionError("Cross-domain intervention estimates are incomplete")
    fitted = read_csv("results/summaries/cross_domain_control.csv")
    task_by_domain = {
        "Value Judgment": "Value Judgment: Continuous",
        "Sentiment Analysis": "Sentiment Analysis: Continuous",
    }
    for domain, task in task_by_domain.items():
        compact = cross_domain[cross_domain["domain"].eq(domain)].set_index("model_name")
        full = fitted[fitted["task"].eq(task)].set_index("model_name")
        for column in ("coefficient", "ci_low", "ci_high", "p_value"):
            if not np.allclose(compact.loc[full.index, column], full[column]):
                raise AssertionError(
                    f"Cross-model processed table is stale for {domain}: {column}"
                )

    hallucination = read_csv(
        "results/summaries/cross_model_factual_judgment.csv"
    )
    assert_models(hallucination, ALL_MODELS)
    if not (hallucination["coefficient"] < 0).all():
        raise AssertionError("Truth-aligned pressure has an unexpected direction")
    if not (hallucination["p_value"] < 0.01).all():
        raise AssertionError("A cross-model coherent-hallucination test is not significant")

    temperature = read_csv(
        "results/summaries/generation_robustness/decoding_temperature_slope_summary.csv"
    )
    assert_models(temperature, ROBUSTNESS_MODELS)
    if set(temperature["temperature"]) != {0.0, 0.2, 1.0}:
        raise AssertionError("Temperature grid is incomplete")
    if not temperature["ci_excludes_zero"].astype(bool).all():
        raise AssertionError("A decoding-temperature slope CI includes zero")

    spelling = read_csv(
        "results/summaries/generation_robustness/prompt_spelling_answer_contrasts.csv"
    )
    assert_models(spelling, ROBUSTNESS_MODELS)
    if spelling["material_supported_change"].astype(bool).any():
        raise AssertionError("Prompt spelling produced a material behavioral change")
    return {
        "minimum_target_layer_alignment": minimum_alignment,
        "maximum_target_layer_alignment": maximum_alignment,
        "cross_domain_intervention_estimates": len(cross_domain),
        "cross_model_factual_judgment_estimates": len(hallucination),
        "temperature_estimates": len(temperature),
        "material_prompt_spelling_changes": int(
            spelling["material_supported_change"].astype(bool).sum()
        ),
    }


def verify_original_statistical_models() -> dict[str, float | int]:
    cross_domain = read_csv("results/summaries/cross_domain_control.csv")
    assert_models(cross_domain, ALL_MODELS)
    if len(cross_domain) != 24:
        raise AssertionError("Cross-domain MixedLM summary is incomplete")
    qwen14_cross = cross_domain[
        cross_domain["model_name"] == "qwen25_14b"
    ].set_index("task")
    expected_cross = {
        "Sentiment Analysis: Binary": 0.7335881764,
        "Sentiment Analysis: Continuous": 0.8388068966,
        "Value Judgment: Continuous": 0.8389735936,
    }
    for task, expected in expected_cross.items():
        if not np.isclose(qwen14_cross.loc[task, "coefficient"], expected, atol=1e-8):
            raise AssertionError(f"Qwen2.5-14B cross-domain coefficient changed: {task}")
    if not (cross_domain["n_items"] == 175).all():
        raise AssertionError("Cross-domain item identifiers are not unique across splits")
    if not cross_domain["converged"].astype(bool).all():
        raise AssertionError("A cross-domain MixedLM did not converge")

    accuracy = read_csv(
        "results/summaries/reasoning_subordination/answer_accuracy.csv"
    ).set_index("task")
    expected_accuracy = {
        "Alphabetical Order: Think-then-Answer": 8.3817082526,
        "Alphabetical Order: Answer-then-Think": 4.5779759340,
        "Factual Judgment": 6.0125377964,
    }
    for task, expected in expected_accuracy.items():
        if not np.isclose(accuracy.loc[task, "coefficient"], expected, atol=1e-8):
            raise AssertionError(f"Answer-accuracy coefficient changed: {task}")

    stance = read_csv("results/summaries/stance_taking.csv").set_index("endpoint")
    expected_stance = {
        "Answer Stance": 0.7847177871,
        "Reasoning Stance": 0.4665299896,
        "Sound Reasoning": -0.9877329013,
    }
    for endpoint, expected in expected_stance.items():
        if not np.isclose(stance.loc[endpoint, "coefficient"], expected, atol=1e-8):
            raise AssertionError(f"Stance-Taking coefficient changed: {endpoint}")

    return {
        "cross_domain_coefficients": len(cross_domain),
        "answer_accuracy_coefficients": len(accuracy),
        "stance_taking_coefficients": len(stance),
    }


def run_verification() -> dict[str, object]:
    validate_manifest()
    return {
        "answer_label_control": verify_answer_labels(),
        "subjective_preference": verify_subjective_preference(),
        "feedback_induced_sycophancy": verify_feedback_sycophancy(),
        "arithmetic_answering_verification": verify_arithmetic(),
        "generality_and_robustness": verify_generality_and_robustness(),
        "original_statistical_models": verify_original_statistical_models(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"status": "passed", "checks": run_verification()}
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
