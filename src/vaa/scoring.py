"""Chat rendering and candidate-sequence scoring used by control tasks."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F

from .models import get_transformer_layers
from .steering import create_steering_hook


def render_chat_prompt(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def candidate_token_metadata(tokenizer: Any, candidate: str) -> dict[str, Any]:
    """Tokenize a standalone candidate exactly as in the reported experiments."""
    token_ids = tokenizer.encode(candidate.lstrip(), add_special_tokens=False)
    token_strings = tokenizer.convert_ids_to_tokens(token_ids)
    return {
        "token_ids": token_ids,
        "token_strings": token_strings,
        "n_tokens": len(token_ids),
        "multi_token": len(token_ids) != 1,
    }


def summarize_token_logprobs(values: list[float]) -> dict[str, float | int]:
    values = [float(value) for value in values]
    if not values:
        return {
            "sequence_logprob_sum": float("nan"),
            "sequence_logprob_mean": float("nan"),
            "sequence_n_tokens": 0,
        }
    return {
        "sequence_logprob_sum": float(np.sum(values)),
        "sequence_logprob_mean": float(np.mean(values)),
        "sequence_n_tokens": len(values),
    }


def summarize_candidate_distribution(
    candidate_probabilities: dict[str, float],
    candidates: list[str] | tuple[str, ...],
    candidate_values: list[float] | tuple[float, ...],
) -> dict[str, float | str | None]:
    """Summarize the unconditioned candidate probabilities used by the surveys."""
    if len(candidates) != len(candidate_values) or not candidates:
        raise ValueError("candidates and candidate_values must have equal nonzero length")
    missing = set(candidates) - set(candidate_probabilities)
    if missing:
        raise ValueError(f"Missing candidate probabilities: {sorted(missing)}")
    weighted = sum(
        candidate_probabilities[candidate] * value
        for candidate, value in zip(candidates, candidate_values)
    )
    mass = sum(candidate_probabilities[candidate] for candidate in candidates)
    conditional = weighted / mass if mass > 0 else None
    minimum = min(candidate_values)
    maximum = max(candidate_values)
    if maximum == minimum:
        raise ValueError("candidate_values must span a nonzero range")
    normalized = (weighted - minimum) / (maximum - minimum)
    response = max(candidates, key=candidate_probabilities.__getitem__)
    return {
        "expected_response": float(weighted),
        "expected_response_normalized": float(normalized),
        "expected_response_conditional": (
            float(conditional) if conditional is not None else None
        ),
        "candidate_probability_mass": float(mass),
        "response": response,
    }


@contextmanager
def steering_intervention(
    model: Any,
    target_layer: int,
    steering_vector: torch.Tensor | None,
    alpha_raw: float,
) -> Iterator[None]:
    if steering_vector is None:
        if float(alpha_raw) != 0.0:
            raise ValueError("A steering vector is required when alpha_raw is nonzero")
        yield
        return
    layer = get_transformer_layers(model)[target_layer]
    handle = layer.register_forward_hook(
        create_steering_hook(steering_vector, alpha_raw)
    )
    try:
        yield
    finally:
        handle.remove()


def score_candidate_sequence(
    model: Any,
    tokenizer: Any,
    rendered_prompt: str,
    candidate: str,
    *,
    target_layer: int,
    steering_vector: torch.Tensor | None,
    alpha_raw: float,
) -> dict[str, Any]:
    """Return the summed and mean log probability of a complete candidate.

    Candidate IDs are obtained by standalone tokenization after removing leading
    whitespace. This reproduces the boundary convention used for the reported
    preference and arithmetic control experiments.
    """
    prompt_batch = tokenizer(rendered_prompt, return_tensors="pt")
    prompt_ids = prompt_batch["input_ids"][0]
    metadata = candidate_token_metadata(tokenizer, candidate)
    candidate_ids = metadata["token_ids"]
    if not candidate_ids:
        return {**metadata, **summarize_token_logprobs([]), "token_logprobs": []}

    candidate_tensor = torch.tensor(candidate_ids, dtype=prompt_ids.dtype)
    input_ids = torch.cat([prompt_ids, candidate_tensor]).unsqueeze(0).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    with steering_intervention(
        model,
        target_layer,
        steering_vector,
        alpha_raw,
    ), torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logprobs = F.log_softmax(outputs.logits[0], dim=-1)
    prompt_length = int(prompt_ids.shape[0])
    token_logprobs = [
        float(logprobs[prompt_length + offset - 1, token_id].item())
        for offset, token_id in enumerate(candidate_ids)
    ]
    return {
        **metadata,
        **summarize_token_logprobs(token_logprobs),
        "token_logprobs": token_logprobs,
    }


def score_candidate_pair(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidate_a: str,
    candidate_b: str,
    *,
    target_layer: int,
    steering_vector: torch.Tensor | None,
    alpha_raw: float,
) -> dict[str, Any]:
    rendered = render_chat_prompt(tokenizer, prompt)
    score_a = score_candidate_sequence(
        model,
        tokenizer,
        rendered,
        candidate_a,
        target_layer=target_layer,
        steering_vector=steering_vector,
        alpha_raw=alpha_raw,
    )
    score_b = score_candidate_sequence(
        model,
        tokenizer,
        rendered,
        candidate_b,
        target_layer=target_layer,
        steering_vector=steering_vector,
        alpha_raw=alpha_raw,
    )
    return {
        "candidate_a": score_a,
        "candidate_b": score_b,
        "logprob_margin_a_minus_b": (
            float(score_a["sequence_logprob_sum"])
            - float(score_b["sequence_logprob_sum"])
        ),
    }


def score_first_token_candidates(
    model: Any,
    tokenizer: Any,
    rendered_prompt: str,
    candidates: list[str] | tuple[str, ...],
    *,
    target_layer: int,
    steering_vector: torch.Tensor | None,
    alpha_raw: float,
) -> dict[str, Any]:
    """Score each candidate's first standalone token after one prompt forward."""
    if not candidates:
        raise ValueError("candidates must not be empty")
    metadata = {
        candidate: candidate_token_metadata(tokenizer, candidate)
        for candidate in candidates
    }
    empty = [
        candidate
        for candidate, record in metadata.items()
        if not record["token_ids"]
    ]
    if empty:
        raise ValueError(f"Candidates have empty tokenizations: {empty}")

    prompt_inputs = tokenizer(rendered_prompt, return_tensors="pt")
    input_ids = prompt_inputs["input_ids"].to(model.device)
    attention_mask = prompt_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(model.device)
    with steering_intervention(
        model,
        target_layer,
        steering_vector,
        alpha_raw,
    ), torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[0, -1, :].float()
    logprobs = F.log_softmax(logits, dim=-1)
    probabilities = torch.softmax(logits, dim=-1)
    token_ids = {
        candidate: int(record["token_ids"][0])
        for candidate, record in metadata.items()
    }
    return {
        "candidate_token_metadata": metadata,
        "candidate_first_token_ids": token_ids,
        "candidate_logits": {
            candidate: float(logits[token_id].item())
            for candidate, token_id in token_ids.items()
        },
        "candidate_logprobs": {
            candidate: float(logprobs[token_id].item())
            for candidate, token_id in token_ids.items()
        },
        "candidate_probabilities": {
            candidate: float(probabilities[token_id].item())
            for candidate, token_id in token_ids.items()
        },
    }


def score_next_token_candidates_batch(
    model: Any,
    tokenizer: Any,
    rendered_prompts: list[str],
    candidates: list[str] | tuple[str, ...],
    *,
    target_layer: int,
    steering_vector: torch.Tensor | None,
    alpha_raw: float,
) -> list[dict[str, Any]]:
    """Score single-token response candidates for a batch of rendered prompts."""
    if not rendered_prompts:
        return []
    metadata = [candidate_token_metadata(tokenizer, candidate) for candidate in candidates]
    invalid = [
        candidate
        for candidate, record in zip(candidates, metadata)
        if record["n_tokens"] != 1
    ]
    if invalid:
        raise ValueError(f"Expected single-token candidates, found: {invalid}")
    candidate_ids = [int(record["token_ids"][0]) for record in metadata]

    previous_padding_side = getattr(tokenizer, "padding_side", None)
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(
            rendered_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(model.device)
    finally:
        if previous_padding_side is not None:
            tokenizer.padding_side = previous_padding_side
    with steering_intervention(
        model,
        target_layer,
        steering_vector,
        alpha_raw,
    ), torch.no_grad():
        outputs = model(**inputs)
    last_logits = outputs.logits[:, -1, :].float()
    last_logprobs = F.log_softmax(last_logits, dim=-1)
    last_probs = torch.softmax(last_logits, dim=-1)

    rows = []
    for row_index in range(len(rendered_prompts)):
        rows.append(
            {
                "candidate_token_ids": dict(zip(candidates, candidate_ids)),
                "candidate_logprobs": {
                    candidate: float(last_logprobs[row_index, token_id].item())
                    for candidate, token_id in zip(candidates, candidate_ids)
                },
                "candidate_probabilities": {
                    candidate: float(last_probs[row_index, token_id].item())
                    for candidate, token_id in zip(candidates, candidate_ids)
                },
            }
        )
    return rows
