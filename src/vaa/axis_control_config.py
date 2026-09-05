"""Configuration and frozen stimuli for representational axis controls."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import yaml

from .config import REPOSITORY_ROOT


DEFAULT_CONFIG_PATH = (
    REPOSITORY_ROOT / "configs" / "experiments" / "axis_controls.yaml"
)


@dataclass(frozen=True)
class ValenceControlSpec:
    display_name: str
    prompt_key: str
    stimulus_file: Path


@dataclass(frozen=True)
class AnswerLabelSpec:
    key: str
    prompt_key: str
    true_label: str
    false_label: str
    protocol: str


@dataclass(frozen=True)
class SingleLetterOrderSpec:
    display_name: str
    stimulus_file: Path
    answer_label_conditions: Mapping[str, AnswerLabelSpec]


@dataclass(frozen=True)
class AxisControlConfig:
    valence: ValenceControlSpec
    single_letter_order: SingleLetterOrderSpec


def _repository_path(value: str, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"Axis-control path must be repository-relative: {value}")
    return root / path


def load_axis_control_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> AxisControlConfig:
    root = Path(repository_root)
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported axis-control configuration schema")

    valence = payload["valence"]
    single_letter = payload["single_letter_order"]
    conditions = {}
    for key, record in single_letter["answer_label_conditions"].items():
        conditions[key] = AnswerLabelSpec(
            key=key,
            prompt_key=str(record["prompt"]),
            true_label=str(record["true_label"]),
            false_label=str(record["false_label"]),
            protocol=str(record["protocol"]),
        )
    if set(conditions) != {"right_wrong", "true_false"}:
        raise ValueError("Expected right_wrong and true_false answer-label conditions")

    return AxisControlConfig(
        valence=ValenceControlSpec(
            display_name=str(valence["display_name"]),
            prompt_key=str(valence["prompt"]),
            stimulus_file=_repository_path(valence["stimulus_file"], root),
        ),
        single_letter_order=SingleLetterOrderSpec(
            display_name=str(single_letter["display_name"]),
            stimulus_file=_repository_path(single_letter["stimulus_file"], root),
            answer_label_conditions=conditions,
        ),
    )


def load_record_array(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise ValueError(f"Expected a JSON array of records: {path}")
    return [
        {str(key): str(value) for key, value in record.items()}
        for record in records
    ]
