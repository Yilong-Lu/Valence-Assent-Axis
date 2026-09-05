"""Narrow model-specific compatibility adjustments used by the experiments."""

from __future__ import annotations

from typing import Any

from .config import ModelSpec


MISTRAL_V03_ORIGINAL_BLOCK = """        {%- if loop.last and system_message is defined %}
            {{- "[INST] " + system_message + "\\n\\n" + message["content"] + "[/INST] " }}
        {%- else %}
            {{- "[INST] " + message["content"] + "[/INST] " }}
        {%- endif %}"""

MISTRAL_V03_REPLACEMENT_BLOCK = """        {{- "[INST] " + message["content"] + "[/INST] " }}"""


def patch_mistral_v03_chat_template(tokenizer: Any) -> bool:
    """Apply the submitted Mistral-v0.3 template adjustment once."""
    template = getattr(tokenizer, "chat_template", None)
    if not template or MISTRAL_V03_ORIGINAL_BLOCK not in template:
        return False
    tokenizer.chat_template = template.replace(
        MISTRAL_V03_ORIGINAL_BLOCK,
        MISTRAL_V03_REPLACEMENT_BLOCK,
    )
    return True


def apply_model_compatibility(tokenizer: Any, model_spec: ModelSpec) -> bool:
    """Apply the registered compatibility adjustment, if one is required."""
    if model_spec.compatibility is None:
        return False
    if model_spec.compatibility == "mistral_v03_chat_template":
        return patch_mistral_v03_chat_template(tokenizer)
    raise ValueError(
        f"Unknown compatibility mode for {model_spec.key}: "
        f"{model_spec.compatibility}"
    )
