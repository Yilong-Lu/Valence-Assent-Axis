"""Representational-axis summaries shared by the axis-control tasks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .axis import cosine_axis_alignment, pearson_axis_alignment


@dataclass(frozen=True)
class AxisComparisonResult:
    axis_vector: np.ndarray
    pc1_scores: np.ndarray
    pc2_scores: np.ndarray
    vaa_projections: np.ndarray
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray
    pc1_vaa_projection_correlation: float
    axis_pearson: float
    axis_cosine: float
    vaa_variance_ratio: float
    sign_flipped: bool


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    value = float(np.corrcoef(first, second)[0, 1])
    if not np.isfinite(value):
        raise ValueError("Encountered an undefined correlation")
    return value


def compare_task_axis_to_vaa(
    activations: np.ndarray,
    vaa_vector: np.ndarray,
    *,
    orient_toward: np.ndarray | None = None,
) -> AxisComparisonResult:
    """Derive task PC1 and compare its geometry with a unit VAA vector."""
    states = np.asarray(activations, dtype=np.float64)
    vector = np.asarray(vaa_vector, dtype=np.float64).reshape(-1)
    if states.ndim != 2 or states.shape[0] < 3:
        raise ValueError("activations must be a 2D array with at least three rows")
    if states.shape[1] != vector.shape[0]:
        raise ValueError("Activation and VAA dimensions do not match")
    vector_norm = np.linalg.norm(vector)
    if vector_norm == 0:
        raise ValueError("VAA vector has zero norm")
    vector = vector / vector_norm

    centered = states - states.mean(axis=0, keepdims=True)
    u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    scores = u * singular_values
    sign_flipped = False
    if orient_toward is not None:
        labels = np.asarray(orient_toward, dtype=np.float64).reshape(-1)
        if labels.shape[0] != states.shape[0]:
            raise ValueError("orient_toward must have one value per activation")
        if _correlation(scores[:, 0], labels) < 0:
            vt[0] *= -1
            scores[:, 0] *= -1
            sign_flipped = True

    vaa_projections = centered @ vector
    total_variance = float(np.var(centered, axis=0, ddof=1).sum())
    if total_variance <= 0:
        raise ValueError("Task activations have zero total variance")
    explained = singular_values**2 / np.sum(singular_values**2)
    pc2_scores = scores[:, 1] if scores.shape[1] > 1 else np.zeros(states.shape[0])
    return AxisComparisonResult(
        axis_vector=vt[0].copy(),
        pc1_scores=scores[:, 0].copy(),
        pc2_scores=pc2_scores.copy(),
        vaa_projections=vaa_projections,
        singular_values=singular_values,
        explained_variance_ratio=explained,
        pc1_vaa_projection_correlation=_correlation(
            scores[:, 0], vaa_projections
        ),
        axis_pearson=pearson_axis_alignment(vt[0], vector),
        axis_cosine=cosine_axis_alignment(vt[0], vector),
        vaa_variance_ratio=float(
            np.var(vaa_projections, ddof=1) / total_variance
        ),
        sign_flipped=sign_flipped,
    )
