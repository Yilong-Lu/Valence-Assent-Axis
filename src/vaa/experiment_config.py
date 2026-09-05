"""Structured experiment configuration and frozen-stimulus loading."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import yaml

from .config import REPOSITORY_ROOT


DEFAULT_JUDGMENT_CONFIG = (
    REPOSITORY_ROOT / "configs" / "experiments" / "judgment_tasks.yaml"
)


@dataclass(frozen=True)
class JudgmentTaskSpec:
    key: str
    display_name: str
    prompt_key: str
    stimulus_files: tuple[Path, ...]
    candidates: tuple[str, ...]
    candidate_values: tuple[float, ...]


@dataclass(frozen=True)
class JudgmentExperimentConfig:
    normalized_alpha_grid: tuple[float, ...]
    tasks: Mapping[str, JudgmentTaskSpec]


def _relative_repository_path(value: str, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"Experiment path must be repository-relative: {value}")
    return root / path


def load_judgment_experiment_config(
    path: str | Path = DEFAULT_JUDGMENT_CONFIG,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> JudgmentExperimentConfig:
    root = Path(repository_root)
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported judgment-task configuration schema")

    tasks = {}
    for key, record in payload.get("tasks", {}).items():
        candidates = tuple(str(value) for value in record["candidates"])
        values = tuple(float(value) for value in record["candidate_values"])
        if len(candidates) != len(values) or not candidates:
            raise ValueError(f"Candidate/value mismatch for {key}")
        tasks[key] = JudgmentTaskSpec(
            key=key,
            display_name=str(record["display_name"]),
            prompt_key=str(record["prompt"]),
            stimulus_files=tuple(
                _relative_repository_path(value, root)
                for value in record["stimulus_files"]
            ),
            candidates=candidates,
            candidate_values=values,
        )
    if not tasks:
        raise ValueError("No judgment tasks are configured")
    return JudgmentExperimentConfig(
        normalized_alpha_grid=tuple(
            float(value) for value in payload["normalized_alpha_grid"]
        ),
        tasks=tasks,
    )


def load_task_stimuli(task: JudgmentTaskSpec) -> list[dict[str, str]]:
    rows = []
    for split_index, path in enumerate(task.stimulus_files):
        with path.open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError(f"Expected a JSON string array: {path}")
        split = "primary" if split_index == 0 else "held_out"
        for item_index, statement in enumerate(values):
            rows.append(
                {
                    "item_id": f"{task.key}::{split}::{item_index:03d}",
                    "split": split,
                    "statement": statement,
                }
            )
    return rows
