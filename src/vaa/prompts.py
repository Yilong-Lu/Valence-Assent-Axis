"""Versioned prompts used in the reported experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .config import REPOSITORY_ROOT


DEFAULT_PROMPT_REGISTRY_PATH = REPOSITORY_ROOT / "configs" / "prompts.json"


@dataclass(frozen=True)
class PromptSpec:
    key: str
    task_name: str
    template: str
    labels: tuple[str, ...]
    model_overrides: Mapping[str, str]
    contains_submitted_spelling: bool = False

    def template_for_model(self, model_key: str | None = None) -> str:
        if model_key is None:
            return self.template
        return self.model_overrides.get(model_key, self.template)


@dataclass(frozen=True)
class PromptRegistry:
    prompts: Mapping[str, PromptSpec]
    feedback_conditions: Mapping[str, str]


def load_prompt_registry(
    path: str | Path = DEFAULT_PROMPT_REGISTRY_PATH,
) -> PromptRegistry:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported prompt-registry schema")

    prompts: dict[str, PromptSpec] = {}
    for key, record in payload.get("prompts", {}).items():
        prompts[key] = PromptSpec(
            key=key,
            task_name=str(record["task_name"]),
            template=str(record["template"]),
            labels=tuple(str(value) for value in record.get("labels", [])),
            model_overrides={
                str(model): str(template)
                for model, template in record.get("model_overrides", {}).items()
            },
            contains_submitted_spelling=bool(
                record.get("contains_submitted_spelling", False)
            ),
        )
    if not prompts:
        raise ValueError("Prompt registry is empty")
    return PromptRegistry(
        prompts=prompts,
        feedback_conditions={
            str(condition): str(sentence)
            for condition, sentence in payload.get("feedback_conditions", {}).items()
        },
    )


def get_prompt_spec(
    prompt_key: str,
    registry: PromptRegistry | None = None,
) -> PromptSpec:
    registry = load_prompt_registry() if registry is None else registry
    try:
        return registry.prompts[prompt_key]
    except KeyError as exc:
        valid = ", ".join(sorted(registry.prompts))
        raise KeyError(
            f"Unknown prompt '{prompt_key}'. Available prompts: {valid}"
        ) from exc


def render_prompt(
    prompt_key: str,
    *,
    model_key: str | None = None,
    registry: PromptRegistry | None = None,
    **values: Any,
) -> str:
    spec = get_prompt_spec(prompt_key, registry)
    return spec.template_for_model(model_key).format(**values)


def render_feedback_prompt(
    argument: str,
    condition: str,
    *,
    registry: PromptRegistry | None = None,
) -> str:
    registry = load_prompt_registry() if registry is None else registry
    try:
        feedback = registry.feedback_conditions[condition]
    except KeyError as exc:
        valid = ", ".join(sorted(registry.feedback_conditions))
        raise KeyError(
            f"Unknown feedback condition '{condition}'. Available conditions: {valid}"
        ) from exc
    feedback_clause = f" {feedback}" if feedback else ""
    return render_prompt(
        "feedback_induced_sycophancy_v1",
        registry=registry,
        feedback_clause=feedback_clause,
        argument=argument,
    )
