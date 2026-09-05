import pytest

from vaa.arithmetic import (
    build_arithmetic_prompts,
    calculate_arithmetic_metrics,
    estimate_arithmetic_item_slopes,
    summarize_arithmetic_by_alpha,
)


ITEM = {
    "item_id": "arith_000",
    "a": 17,
    "b": 24,
    "correct": 41,
    "incorrect": 43,
    "offset": 2,
}


def test_build_arithmetic_prompts_creates_matched_three_condition_design():
    rows = build_arithmetic_prompts(
        [ITEM],
        "arithmetic_direct_numeric_v1",
        "arithmetic_verification_v1",
    )
    assert [row["mode"] for row in rows] == [
        "direct_numeric",
        "verification_true",
        "verification_false",
    ]
    assert rows[0]["candidate_A"] == "41"
    assert rows[0]["candidate_B"] == "43"
    assert rows[0]["prompt"] == "What is 17 + 24? Answer with a number only."
    assert rows[1]["prompt"] == (
        '17 + 24 = 41. Is this true or false? Answer only "True" or "False".'
    )
    assert rows[2]["prompt"] == (
        '17 + 24 = 43. Is this true or false? Answer only "True" or "False".'
    )


def test_metrics_separate_assent_from_verification_correctness():
    direct = calculate_arithmetic_metrics("direct_numeric", None, 2.0)
    true_statement = calculate_arithmetic_metrics("verification_true", True, 2.0)
    false_statement = calculate_arithmetic_metrics("verification_false", False, 2.0)
    assert direct["candidate_accuracy"] == 1
    assert direct["numeric_correct_margin"] == 2.0
    assert direct["assent_margin"] is None
    assert true_statement["verification_correct_margin"] == 2.0
    assert true_statement["candidate_accuracy"] == 1
    assert false_statement["assent_margin"] == 2.0
    assert false_statement["verification_correct_margin"] == -2.0
    assert false_statement["candidate_accuracy"] == 0


def test_item_slopes_and_alpha_summary_preserve_accuracy_estimand():
    rows = []
    for alpha, accuracy in ((-1.0, 0), (0.0, 1), (1.0, 1)):
        rows.append(
            {
                "model_key": "model",
                "target_layer": 1,
                **ITEM,
                "prompt_id": "arith_000::direct_numeric",
                "mode": "direct_numeric",
                "statement_truth": None,
                "alpha_norm": alpha,
                "alpha_raw": alpha,
                "primary_margin": 2 * alpha + 1,
                "numeric_correct_margin": 2 * alpha + 1,
                "assent_margin": None,
                "verification_correct_margin": None,
                "candidate_accuracy": accuracy,
            }
        )
    slopes = estimate_arithmetic_item_slopes(rows)
    summary = summarize_arithmetic_by_alpha(rows)
    assert len(slopes) == 1
    assert slopes[0]["primary_margin_slope"] == pytest.approx(2.0)
    assert slopes[0]["candidate_accuracy_slope"] == pytest.approx(0.5)
    assert len(summary) == 3
    assert {row["candidate_accuracy"] for row in summary} == {0.0, 1.0}
