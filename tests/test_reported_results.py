from analysis.python.verify_reported_results import run_verification


def test_reported_results_pass_scientific_checks():
    report = run_verification()
    assert report["feedback_induced_sycophancy"][
        "qwen25_14b_baseline_strong_rate"
    ] > 0
    assert report["generality_and_robustness"][
        "material_prompt_spelling_changes"
    ] == 0
