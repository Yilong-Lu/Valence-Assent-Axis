"""Configuration and frozen stimuli for arithmetic answering and verification."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import yaml

from .config import REPOSITORY_ROOT


DEFAULT_ARITHMETIC_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "arithmetic_answering_verification.yaml"
)


@dataclass(frozen=True)
class ArithmeticExperimentConfig:
    display_name: str
    direct_prompt_key: str
    verification_prompt_key: str
    stimulus_file: Path
    normalized_alpha_grid: tuple[float, ...]
    primary_outcome: str
    candidate_scoring: str


def _repository_path(value: str, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"Arithmetic path must be repository-relative: {value}")
    return root / path


def load_arithmetic_config(
    path: str | Path = DEFAULT_ARITHMETIC_CONFIG,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> ArithmeticExperimentConfig:
    root = Path(repository_root)
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported arithmetic configuration schema")

    alpha_grid = tuple(float(value) for value in payload["normalized_alpha_grid"])
    if not alpha_grid or sorted(set(alpha_grid)) != list(alpha_grid):
        raise ValueError("Arithmetic alpha grid must contain unique sorted values")
    if alpha_grid[0] < -1 or alpha_grid[-1] > 1 or 0.0 not in alpha_grid:
        raise ValueError("Arithmetic alpha grid must lie in [-1, 1] and include zero")
    if payload["primary_outcome"] != "candidate_accuracy":
        raise ValueError("The registered arithmetic outcome must be candidate accuracy")
    if payload["candidate_scoring"] != "full_sequence_log_probability":
        raise ValueError("Unsupported arithmetic candidate scorer")

    prompts = payload["prompts"]
    return ArithmeticExperimentConfig(
        display_name=str(payload["display_name"]),
        direct_prompt_key=str(prompts["direct_numeric"]),
        verification_prompt_key=str(prompts["verification"]),
        stimulus_file=_repository_path(payload["stimulus_file"], root),
        normalized_alpha_grid=alpha_grid,
        primary_outcome=str(payload["primary_outcome"]),
        candidate_scoring=str(payload["candidate_scoring"]),
    )


def load_arithmetic_items(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    required = {"item_id", "a", "b", "correct", "incorrect", "offset"}
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a nonempty JSON array: {path}")

    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required.issubset(row):
            missing = required - set(row) if isinstance(row, dict) else required
            raise ValueError(f"Arithmetic row {index} is missing fields: {sorted(missing)}")
        item = {
            "item_id": str(row["item_id"]),
            "a": int(row["a"]),
            "b": int(row["b"]),
            "correct": int(row["correct"]),
            "incorrect": int(row["incorrect"]),
            "offset": int(row["offset"]),
        }
        if not 2 <= item["a"] <= 49 or not 2 <= item["b"] <= 49:
            raise ValueError(f"Arithmetic operands out of range in {item['item_id']}")
        if item["correct"] != item["a"] + item["b"] or item["correct"] > 99:
            raise ValueError(f"Incorrect registered sum in {item['item_id']}")
        if item["incorrect"] != item["correct"] + item["offset"]:
            raise ValueError(f"Incorrect offset in {item['item_id']}")
        if item["incorrect"] < 0 or not 1 <= abs(item["offset"]) <= 9:
            raise ValueError(f"Invalid comparison answer in {item['item_id']}")
        normalized.append(item)

    item_ids = [row["item_id"] for row in normalized]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("Arithmetic item IDs must be unique")
    unordered_pairs = [tuple(sorted((row["a"], row["b"]))) for row in normalized]
    if len(set(unordered_pairs)) != len(unordered_pairs):
        raise ValueError("Arithmetic operand pairs must be unique up to order")
    return normalized
