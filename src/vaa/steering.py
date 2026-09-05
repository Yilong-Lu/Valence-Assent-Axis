"""VAA vector loading and the persistent residual-stream intervention."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .config import ModelSpec


def normalized_alpha_to_raw(
    alpha_norm: float,
    raw_alpha_range: tuple[float, float],
) -> float:
    """Map normalized intervention strength to the model-specific raw scale."""
    if not -1.0 <= alpha_norm <= 1.0:
        raise ValueError(f"alpha_norm must be within [-1, 1], got {alpha_norm}")
    alpha_min, alpha_max = raw_alpha_range
    scale = abs(alpha_min) if alpha_norm < 0 else abs(alpha_max)
    return float(alpha_norm * scale)


def load_vaa_vector(
    model_spec: ModelSpec,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Load a selected VAA vector from the non-pickle NumPy artifact."""
    array = np.load(model_spec.vector_path, allow_pickle=False)
    if array.ndim != 1 or array.shape[0] != model_spec.hidden_size:
        raise ValueError(
            f"Unexpected vector shape for {model_spec.key}: {array.shape}; "
            f"expected ({model_spec.hidden_size},)"
        )
    vector = torch.from_numpy(array)
    return vector.to(device=device, dtype=dtype)


def create_steering_hook(
    steering_vector: torch.Tensor,
    alpha_raw: float,
):
    """Add the intervention at every sequence position and generation step."""

    def steering_hook(module: Any, inputs: Any, output: Any):
        del module, inputs
        hidden_states = output[0] if isinstance(output, tuple) else output
        vector = steering_vector.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        alpha = torch.as_tensor(
            alpha_raw,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        modified = hidden_states + alpha * vector
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    return steering_hook
