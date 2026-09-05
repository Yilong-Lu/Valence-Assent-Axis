"""Configuration for open-ended reasoning and stance experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import REPOSITORY_ROOT


DEFAULT_GENERATIVE_CONFIG = (
    REPOSITORY_ROOT / "configs" / "experiments" / "generative_reasoning.yaml"
)


@dataclass(frozen=True)
class GenerationConfig:
    batch_size: int
    max_input_tokens: int
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int
    seed: int


@dataclass(frozen=True)
class GenerativeTaskSpec:
    key: str
    display_name: str
    stimulus_file: Path
    prompt_key: str | None
    conditions: Mapping[str, str]
    answer_labels: tuple[str, ...]


@dataclass(frozen=True)
class GenerativeExperimentConfig:
    normalized_alpha_grid: tuple[float, ...]
    generation: GenerationConfig
    tasks: Mapping[str, GenerativeTaskSpec]


def _repository_path(value: str, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"Generative-task path must be repository-relative: {value}")
    return root / path


def load_generative_config(
    path: str | Path = DEFAULT_GENERATIVE_CONFIG,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> GenerativeExperimentConfig:
    root = Path(repository_root)
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported generative-task configuration schema")

    alpha_grid = tuple(float(value) for value in payload["normalized_alpha_grid"])
    if not alpha_grid or tuple(sorted(set(alpha_grid))) != alpha_grid:
        raise ValueError("Normalized alpha grid must be sorted and unique")
    if alpha_grid[0] < -1 or alpha_grid[-1] > 1 or 0.0 not in alpha_grid:
        raise ValueError("Normalized alpha grid must lie in [-1, 1] and include zero")

    generation_payload = payload["generation"]
    generation = GenerationConfig(
        batch_size=int(generation_payload["batch_size"]),
        max_input_tokens=int(generation_payload["max_input_tokens"]),
        max_new_tokens=int(generation_payload["max_new_tokens"]),
        do_sample=bool(generation_payload["do_sample"]),
        temperature=float(generation_payload["temperature"]),
        top_p=float(generation_payload["top_p"]),
        top_k=int(generation_payload["top_k"]),
        seed=int(generation_payload["seed"]),
    )
    if min(
        generation.batch_size,
        generation.max_input_tokens,
        generation.max_new_tokens,
    ) <= 0:
        raise ValueError("Generation sizes must be positive")

    tasks = {}
    for key, record in payload["tasks"].items():
        conditions = {
            str(condition): str(prompt_key)
            for condition, prompt_key in record.get("conditions", {}).items()
        }
        prompt_key = record.get("prompt")
        if (prompt_key is None) == (not conditions):
            raise ValueError(
                f"Task {key} must define either one prompt or named conditions"
            )
        tasks[key] = GenerativeTaskSpec(
            key=str(key),
            display_name=str(record["display_name"]),
            stimulus_file=_repository_path(record["stimulus_file"], root),
            prompt_key=str(prompt_key) if prompt_key is not None else None,
            conditions=conditions,
            answer_labels=tuple(
                str(label).lower() for label in record.get("answer_labels", [])
            ),
        )
    expected = {"alphabetical_order", "factual_judgment", "stance_taking"}
    if set(tasks) != expected:
        raise ValueError(f"Expected generative tasks: {sorted(expected)}")
    return GenerativeExperimentConfig(alpha_grid, generation, tasks)


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows or not all(
        isinstance(row, dict) for row in rows
    ):
        raise ValueError(f"Expected a nonempty JSON record array: {path}")
    item_ids = [str(row.get("item_id", "")) for row in rows]
    if any(not item_id for item_id in item_ids) or len(set(item_ids)) != len(item_ids):
        raise ValueError(f"Stimulus item IDs must be nonempty and unique: {path}")
    return rows
