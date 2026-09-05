"""Model and tokenizer loading from the public registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .compat import apply_model_compatibility
from .config import ModelSpec, get_model_spec
from .steering import load_vaa_vector


@dataclass(frozen=True)
class ModelBundle:
    spec: ModelSpec
    model: Any
    tokenizer: Any
    vaa_vector: torch.Tensor | None
    compatibility_applied: bool


def get_transformer_layers(model: Any) -> Any:
    try:
        return model.model.layers
    except AttributeError as exc:
        raise AttributeError(
            "Expected a causal language model with transformer blocks at "
            "model.layers"
        ) from exc


def resolve_torch_dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    aliases = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return aliases[value]
    except KeyError as exc:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown dtype '{value}'. Available dtypes: {valid}") from exc


def load_model_bundle(
    model_key: str,
    *,
    device_map: str | dict[str, Any] = "auto",
    dtype: str | torch.dtype = "bfloat16",
    local_files_only: bool = False,
    trust_remote_code: bool = True,
    load_vector: bool = True,
) -> ModelBundle:
    """Load one registered model and tokenizer, optionally with its VAA vector."""
    spec = get_model_spec(model_key)
    model_reference = spec.resolve_model_reference()
    torch_dtype = resolve_torch_dtype(dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_reference,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_reference,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    compatibility_applied = apply_model_compatibility(tokenizer, spec)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    vector = load_vaa_vector(spec) if load_vector else None
    return ModelBundle(
        spec=spec,
        model=model,
        tokenizer=tokenizer,
        vaa_vector=vector,
        compatibility_applied=compatibility_applied,
    )
