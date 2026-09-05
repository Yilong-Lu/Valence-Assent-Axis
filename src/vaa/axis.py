"""Principal-axis derivation and layer-wise representational alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class PrincipalAxisResult:
    vector: np.ndarray
    scores: np.ndarray
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray
    response_correlation: float | None
    sign_flipped: bool


def derive_principal_axis(
    activations: np.ndarray,
    align_response: np.ndarray | None = None,
) -> PrincipalAxisResult:
    """Derive centered PC1 and optionally orient it toward larger responses."""
    states = np.asarray(activations, dtype=np.float64)
    if states.ndim != 2 or states.shape[0] < 2:
        raise ValueError("activations must be a two-dimensional array with >=2 rows")
    centered = states - states.mean(axis=0, keepdims=True)
    u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    vector = vt[0].copy()
    scores = u[:, 0] * singular_values[0]
    response_correlation = None
    sign_flipped = False

    if align_response is not None:
        response = np.asarray(align_response, dtype=np.float64).reshape(-1)
        if response.shape[0] != states.shape[0]:
            raise ValueError("align_response must have one value per activation row")
        response_correlation = float(np.corrcoef(scores, response)[0, 1])
        if not np.isfinite(response_correlation):
            raise ValueError("Could not orient PC1 because the correlation is undefined")
        if response_correlation < 0:
            vector *= -1
            scores *= -1
            response_correlation *= -1
            sign_flipped = True

    explained = singular_values**2 / np.sum(singular_values**2)
    return PrincipalAxisResult(
        vector=vector,
        scores=scores,
        singular_values=singular_values,
        explained_variance_ratio=explained,
        response_correlation=response_correlation,
        sign_flipped=sign_flipped,
    )


def derive_layer_axes(
    activations: Mapping[int, np.ndarray],
    align_response: np.ndarray | None = None,
) -> dict[int, PrincipalAxisResult]:
    return {
        int(layer): derive_principal_axis(states, align_response)
        for layer, states in activations.items()
    }


def pearson_axis_alignment(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Return Pearson correlation between two same-width axis vectors."""
    first = np.asarray(vector_a, dtype=np.float64).reshape(-1)
    second = np.asarray(vector_b, dtype=np.float64).reshape(-1)
    if first.shape != second.shape:
        raise ValueError(f"Axis shapes differ: {first.shape} and {second.shape}")
    value = float(np.corrcoef(first, second)[0, 1])
    if not np.isfinite(value):
        raise ValueError("Axis alignment is undefined")
    return value


def cosine_axis_alignment(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    first = np.asarray(vector_a, dtype=np.float64).reshape(-1)
    second = np.asarray(vector_b, dtype=np.float64).reshape(-1)
    if first.shape != second.shape:
        raise ValueError(f"Axis shapes differ: {first.shape} and {second.shape}")
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        raise ValueError("Cannot align a zero vector")
    return float(np.dot(first, second) / denominator)
