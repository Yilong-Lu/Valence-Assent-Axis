"""Configuration for decoding-temperature and prompt-spelling checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from .config import REPOSITORY_ROOT


DEFAULT_ROBUSTNESS_CONFIG = (
    REPOSITORY_ROOT / "configs" / "experiments" / "generation_robustness.yaml"
)


@dataclass(frozen=True)
class RobustnessGenerationConfig:
    batch_size: int
    max_input_tokens: int
    max_new_tokens: int
    top_p: float
    top_k: int


@dataclass(frozen=True)
class RobustnessTaskSpec:
    key: str
    display_name: str
    stimulus_file: Path
    prompts: Mapping[str, str]


@dataclass(frozen=True)
class TemperatureCondition:
    value: float
    sampled: bool
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class RobustnessProtocolSpec:
    key: str
    display_name: str
    prompt_versions: tuple[str, ...]
    normalized_alpha_grid: tuple[float, ...]
    temperatures: tuple[TemperatureCondition, ...]


@dataclass(frozen=True)
class GenerationRobustnessConfig:
    model_keys: tuple[str, ...]
    generation: RobustnessGenerationConfig
    tasks: Mapping[str, RobustnessTaskSpec]
    protocols: Mapping[str, RobustnessProtocolSpec]


def _repository_path(value: str, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"Robustness path must be repository-relative: {value}")
    return root / path


def _alpha_grid(values: list[float]) -> tuple[float, ...]:
    grid = tuple(float(value) for value in values)
    if not grid or tuple(sorted(set(grid))) != grid:
        raise ValueError("Robustness alpha grid must be sorted and unique")
    if grid[0] < -1 or grid[-1] > 1 or 0.0 not in grid:
        raise ValueError("Robustness alpha grid must lie in [-1, 1] and include zero")
    return grid


def _temperature_conditions(records: list[dict]) -> tuple[TemperatureCondition, ...]:
    conditions = tuple(
        TemperatureCondition(
            value=float(record["value"]),
            sampled=bool(record["sampled"]),
            seeds=tuple(int(seed) for seed in record["seeds"]),
        )
        for record in records
    )
    if not conditions or len({condition.value for condition in conditions}) != len(
        conditions
    ):
        raise ValueError("Robustness temperatures must be nonempty and unique")
    for condition in conditions:
        if not condition.seeds or len(set(condition.seeds)) != len(condition.seeds):
            raise ValueError("Each temperature requires unique seeds")
        if condition.sampled != (condition.value > 0):
            raise ValueError("Temperature zero must be greedy; positive values sampled")
    return conditions


def load_generation_robustness_config(
    path: str | Path = DEFAULT_ROBUSTNESS_CONFIG,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> GenerationRobustnessConfig:
    root = Path(repository_root)
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported generation-robustness configuration schema")

    generation_record = payload["generation"]
    generation = RobustnessGenerationConfig(
        batch_size=int(generation_record["batch_size"]),
        max_input_tokens=int(generation_record["max_input_tokens"]),
        max_new_tokens=int(generation_record["max_new_tokens"]),
        top_p=float(generation_record["top_p"]),
        top_k=int(generation_record["top_k"]),
    )
    if min(
        generation.batch_size,
        generation.max_input_tokens,
        generation.max_new_tokens,
    ) <= 0:
        raise ValueError("Robustness generation sizes must be positive")

    tasks = {}
    for key, record in payload["tasks"].items():
        tasks[key] = RobustnessTaskSpec(
            key=str(key),
            display_name=str(record["display_name"]),
            stimulus_file=_repository_path(record["stimulus_file"], root),
            prompts={
                "submitted": str(record["submitted_prompt"]),
                "corrected": str(record["corrected_prompt"]),
            },
        )
    expected_tasks = {"alphabetical_order", "factual_judgment"}
    if set(tasks) != expected_tasks:
        raise ValueError(f"Expected robustness tasks: {sorted(expected_tasks)}")

    protocols = {}
    for key in ("prompt_spelling", "decoding_temperature"):
        record = payload[key]
        versions = tuple(str(value) for value in record["prompt_versions"])
        if not versions or not set(versions) <= {"submitted", "corrected"}:
            raise ValueError(f"Invalid prompt versions for {key}: {versions}")
        protocols[key] = RobustnessProtocolSpec(
            key=key,
            display_name=str(record["display_name"]),
            prompt_versions=versions,
            normalized_alpha_grid=_alpha_grid(record["normalized_alpha_grid"]),
            temperatures=_temperature_conditions(record["temperatures"]),
        )

    model_keys = tuple(str(value) for value in payload["model_keys"])
    if len(model_keys) != 5 or len(set(model_keys)) != len(model_keys):
        raise ValueError("The registered robustness panel must contain five models")
    return GenerationRobustnessConfig(model_keys, generation, tasks, protocols)
