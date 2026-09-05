import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "python"
    / "judge_validation.py"
)
SPEC = importlib.util.spec_from_file_location("judge_validation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_icc_is_one_for_identical_raters():
    values = np.array([[1, 1], [3, 3], [7, 7]], dtype=float)
    assert MODULE.icc_2_1(values) == 1.0


def test_judge_validation_reproduces_reported_values():
    report = MODULE.build_report(MODULE.DEFAULT_DATA_DIR)
    experts = {
        row["score_type"]: row
        for row in report["reasoning_expert_reliability"]
    }
    assert [experts[key]["n_agreement"] for key in ("FC", "LC", "RS")] == [
        113,
        113,
        117,
    ]
    reasoning = {
        (row["judge"], row["score_type"]): row["quadratic_weighted_kappa"]
        for row in report["reasoning_judge_validation"]
    }
    assert reasoning[("DeepSeek R1", "FC")] == pytest.approx(0.870991178)
    assert reasoning[("DeepSeek R1", "LC")] == pytest.approx(0.828910670)
    assert reasoning[("DeepSeek R1", "RS")] == pytest.approx(0.620129870)
    stance = {
        (row["judge"], row["measure"]): row["icc_2_1"]
        for row in report["stance_judge_validation"]
    }
    assert stance[("Qwen3-30B-A3B-Instruct", "final_stance")] == pytest.approx(
        0.984941323
    )
    assert stance[("Qwen3-30B-A3B-Instruct", "reasoning_stance")] == pytest.approx(
        0.969084219
    )
