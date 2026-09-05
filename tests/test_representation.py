import numpy as np
import pytest

from vaa.representation import compare_task_axis_to_vaa


def test_task_axis_comparison_orients_pc1_and_reports_variance():
    states = np.array(
        [
            [-3.0, 0.2, 1.0],
            [-2.0, -0.1, 1.0],
            [-1.0, 0.1, 1.0],
            [1.0, -0.1, 1.0],
            [2.0, 0.1, 1.0],
            [3.0, -0.2, 1.0],
        ]
    )
    labels = np.array([0, 0, 0, 1, 1, 1])
    result = compare_task_axis_to_vaa(
        states,
        np.array([1.0, 0.0, 0.0]),
        orient_toward=labels,
    )
    assert np.corrcoef(result.pc1_scores, labels)[0, 1] > 0
    assert result.pc1_vaa_projection_correlation > 0.99
    assert result.axis_cosine > 0.99
    assert result.vaa_variance_ratio > 0.99


def test_task_axis_comparison_rejects_incompatible_shapes():
    with pytest.raises(ValueError, match="dimensions do not match"):
        compare_task_axis_to_vaa(np.ones((4, 3)), np.ones(2))
