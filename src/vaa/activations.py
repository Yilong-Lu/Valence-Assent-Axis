"""Assistant-start hidden-state extraction and VAA projection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from .models import get_transformer_layers


def render_chat_messages(tokenizer: Any, prompt: str | list[dict[str, str]]) -> str:
    messages = (
        [{"role": "user", "content": prompt}]
        if isinstance(prompt, str)
        else prompt
    )
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _input_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("Could not determine the model input device") from exc


def capture_assistant_start_activations(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str | list[dict[str, str]]],
    layer_numbers: Sequence[int],
    *,
    batch_size: int = 16,
    max_input_tokens: int = 1024,
) -> dict[int, np.ndarray]:
    """Capture the final prompt position after each selected transformer block."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not prompts:
        raise ValueError("prompts must not be empty")
    if not layer_numbers:
        raise ValueError("layer_numbers must not be empty")

    layers = get_transformer_layers(model)
    unique_layers = list(dict.fromkeys(int(layer) for layer in layer_numbers))
    invalid = [layer for layer in unique_layers if layer < 0 or layer >= len(layers)]
    if invalid:
        raise IndexError(f"Layer indices out of range: {invalid}")

    captured: dict[int, list[np.ndarray]] = {layer: [] for layer in unique_layers}
    handles = []

    def make_hook(layer_number: int):
        def hook(module: Any, inputs: Any, output: Any) -> None:
            del module, inputs
            hidden = output[0] if isinstance(output, tuple) else output
            state = hidden[:, -1, :].detach().float().cpu().numpy()
            captured[layer_number].append(state)

        return hook

    for layer_number in unique_layers:
        handles.append(layers[layer_number].register_forward_hook(make_hook(layer_number)))

    previous_padding_side = getattr(tokenizer, "padding_side", None)
    tokenizer.padding_side = "left"
    device = _input_device(model)
    try:
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            rendered = [render_chat_messages(tokenizer, prompt) for prompt in batch_prompts]
            inputs = tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_tokens,
            ).to(device)
            with torch.inference_mode():
                model(**inputs)
    finally:
        for handle in handles:
            handle.remove()
        if previous_padding_side is not None:
            tokenizer.padding_side = previous_padding_side

    output: dict[int, np.ndarray] = {}
    for layer_number, batches in captured.items():
        if not batches:
            raise RuntimeError(f"Layer {layer_number} hook captured no activations")
        values = np.concatenate(batches, axis=0)
        if values.shape[0] != len(prompts):
            raise RuntimeError(
                f"Layer {layer_number} captured {values.shape[0]} rows for "
                f"{len(prompts)} prompts"
            )
        output[layer_number] = values
    return output


def project_onto_vaa(
    activations: np.ndarray | torch.Tensor,
    vaa_vector: np.ndarray | torch.Tensor,
) -> dict[str, np.ndarray]:
    """Return raw, unit-vector, and cosine projections for one or more states."""
    states = torch.as_tensor(activations).detach().float()
    vector = torch.as_tensor(vaa_vector).detach().float().reshape(-1)
    if states.ndim == 1:
        states = states.unsqueeze(0)
    if states.ndim != 2 or states.shape[1] != vector.shape[0]:
        raise ValueError(
            f"Incompatible activation/vector shapes: {tuple(states.shape)} and "
            f"{tuple(vector.shape)}"
        )
    vector_norm = torch.linalg.vector_norm(vector)
    if float(vector_norm) == 0.0:
        raise ValueError("VAA vector has zero norm")
    activation_norm = torch.linalg.vector_norm(states, dim=1)
    raw = states @ vector
    unit = raw / vector_norm
    cosine = torch.where(
        activation_norm > 0,
        raw / (vector_norm * activation_norm),
        torch.zeros_like(raw),
    )
    return {
        "projection_raw": raw.cpu().numpy(),
        "projection_unit": unit.cpu().numpy(),
        "projection_cosine": cosine.cpu().numpy(),
        "activation_norm": activation_norm.cpu().numpy(),
    }
