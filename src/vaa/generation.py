"""Batched autoregressive generation under persistent VAA intervention."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from .activations import project_onto_vaa
from .models import get_transformer_layers
from .scoring import render_chat_prompt, steering_intervention


def set_generation_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _generated_tokens_and_stop_reason(
    token_ids: list[int],
    *,
    eos_token_ids: set[int],
    pad_token_id: int | None,
) -> tuple[list[int], str]:
    for index, token_id in enumerate(token_ids):
        if token_id in eos_token_ids:
            return token_ids[:index], "eos_token"
    if pad_token_id is not None:
        while token_ids and token_ids[-1] == pad_token_id:
            token_ids.pop()
    return token_ids, "max_new_tokens"


def generate_text_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    target_layer: int,
    steering_vector: torch.Tensor,
    alpha_raw: float,
    max_input_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
) -> list[dict[str, Any]]:
    if not prompts:
        return []
    rendered = [render_chat_prompt(tokenizer, prompt) for prompt in prompts]
    inputs = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    )
    input_device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    input_width = int(inputs["input_ids"].shape[1])
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "top_p": top_p,
        "top_k": top_k,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature

    with steering_intervention(
        model,
        target_layer,
        steering_vector,
        alpha_raw,
    ), torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    eos = tokenizer.eos_token_id
    eos_token_ids = (
        set(eos if isinstance(eos, (list, tuple, set)) else [eos])
        if eos is not None
        else set()
    )
    rows = []
    for sequence in output_ids:
        generated_ids = sequence[input_width:].detach().cpu().tolist()
        clean_ids, stop_reason = _generated_tokens_and_stop_reason(
            generated_ids,
            eos_token_ids=eos_token_ids,
            pad_token_id=tokenizer.pad_token_id,
        )
        rows.append(
            {
                "generated_text": tokenizer.decode(
                    clean_ids,
                    skip_special_tokens=True,
                ).strip(),
                "generated_n_tokens": len(clean_ids),
                "generation_stop_reason": stop_reason,
            }
        )
    return rows


def generate_text_batch_with_state_capture(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    target_layer: int,
    steering_vector: torch.Tensor,
    alpha_raw: float,
    max_input_tokens: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Greedily generate text and capture the assistant-start VAA state."""
    if not prompts:
        return []
    rendered = [render_chat_prompt(tokenizer, prompt) for prompt in prompts]
    inputs = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    )
    prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    if any(int(length) > max_input_tokens for length in prompt_lengths):
        raise ValueError(
            "At least one rendered prompt exceeds the configured input limit"
        )
    input_device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    input_width = int(inputs["input_ids"].shape[1])
    captured: dict[str, torch.Tensor] = {}

    def capture_and_steer(module: Any, hook_inputs: Any, output: Any):
        del module, hook_inputs
        hidden = output[0] if isinstance(output, tuple) else output
        vector = steering_vector.to(device=hidden.device, dtype=hidden.dtype)
        shift = torch.as_tensor(
            alpha_raw,
            device=hidden.device,
            dtype=hidden.dtype,
        ) * vector
        modified = hidden + shift
        if "pre" not in captured:
            captured["pre"] = hidden[:, -1, :].detach().float().cpu()
            captured["post"] = modified[:, -1, :].detach().float().cpu()
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    layer = get_transformer_layers(model)[target_layer]
    handle = layer.register_forward_hook(capture_and_steer)
    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = tokenizer.eos_token_id
    try:
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos,
            )
    finally:
        handle.remove()
    if "pre" not in captured or "post" not in captured:
        raise RuntimeError("Target layer did not capture assistant-start states")

    pre_projection = project_onto_vaa(captured["pre"], steering_vector)
    post_projection = project_onto_vaa(captured["post"], steering_vector)
    eos_token_ids = (
        set(eos if isinstance(eos, (list, tuple, set)) else [eos])
        if eos is not None
        else set()
    )
    output_ids = generated.sequences
    generated_ids = output_ids[:, input_width:]
    token_logprobs: list[list[float]] = [[] for _ in prompts]
    for step_index, scores in enumerate(generated.scores):
        step_ids = generated_ids[:, step_index]
        selected = torch.log_softmax(scores.float(), dim=-1).gather(
            1,
            step_ids[:, None],
        ).squeeze(1)
        for row_index, value in enumerate(selected.detach().cpu().tolist()):
            token_logprobs[row_index].append(float(value))

    rows = []
    for index, sequence in enumerate(output_ids):
        generated_ids = sequence[input_width:].detach().cpu().tolist()
        clean_ids, stop_reason = _generated_tokens_and_stop_reason(
            generated_ids,
            eos_token_ids=eos_token_ids,
            pad_token_id=tokenizer.pad_token_id,
        )
        row = {
            "prompt_n_tokens": int(prompt_lengths[index]),
            "generated_text": tokenizer.decode(
                clean_ids,
                skip_special_tokens=True,
            ).strip(),
            "generated_token_ids": clean_ids,
            "generated_token_logprobs": token_logprobs[index][: len(clean_ids)],
            "generated_sequence_logprob": float(
                sum(token_logprobs[index][: len(clean_ids)])
            ),
            "generated_n_tokens": len(clean_ids),
            "generation_stop_reason": stop_reason,
        }
        for stage, projection in (
            ("pre_addition", pre_projection),
            ("post_addition", post_projection),
        ):
            row[f"{stage}_vaa_projection_raw"] = float(
                projection["projection_raw"][index]
            )
            row[f"{stage}_vaa_projection_unit"] = float(
                projection["projection_unit"][index]
            )
            row[f"{stage}_vaa_projection_cosine"] = float(
                projection["projection_cosine"][index]
            )
            row[f"{stage}_activation_norm"] = float(
                projection["activation_norm"][index]
            )
        rows.append(row)
    return rows
