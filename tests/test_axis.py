import numpy as np
import pytest

from vaa.axis import (
    cosine_axis_alignment,
    derive_principal_axis,
    pearson_axis_alignment,
)


def test_principal_axis_is_unit_length_and_oriented_toward_response():
    states = np.array(
        [
            [-3.0, 0.1, 1.0],
            [-1.0, -0.1, 1.0],
            [1.0, 0.1, 1.0],
            [3.0, -0.1, 1.0],
        ]
    )
    response = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    result = derive_principal_axis(states, response)
    assert np.linalg.norm(result.vector) == pytest.approx(1.0)
    assert result.vector[0] > 0
    assert result.response_correlation > 0.99
    assert result.explained_variance_ratio[0] > 0.99


def test_axis_alignment_metrics_are_explicitly_distinct():
    first = np.array([1.0, 2.0, 4.0])
    second = np.array([2.0, 4.0, 8.0])
    assert pearson_axis_alignment(first, second) == pytest.approx(1.0)
    assert cosine_axis_alignment(first, second) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="shapes differ"):
        pearson_axis_alignment(first, second[:2])


def test_axis_orientation_rejects_mismatched_response_length():
    with pytest.raises(ValueError, match="one value per activation"):
        derive_principal_axis(np.ones((3, 2)), np.ones(2))
