"""Configuration and frozen stimuli for Feedback-Induced Sycophancy."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import REPOSITORY_ROOT


DEFAULT_SYCOPHANCY_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "feedback_induced_sycophancy.yaml"
)


@dataclass(frozen=True)
class SycophancyGenerationConfig:
    batch_size: int
    max_input_tokens: int
    max_new_tokens: int
    do_sample: bool
    seed: int


@dataclass(frozen=True)
class SycophancyExperimentConfig:
    display_name: str
    stimulus_file: Path
    selection_manifest: Path
    prompt_key: str
    conditions: Mapping[str, str]
    normalized_alpha_grid: tuple[float, ...]
    generation: SycophancyGenerationConfig
    activation_endpoint: str
    primary_state_metric: str
    primary_behavioral_outcome: str


def _repository_path(value: str, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"Sycophancy path must be repository-relative: {value}")
    return root / path


def load_sycophancy_config(
    path: str | Path = DEFAULT_SYCOPHANCY_CONFIG,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> SycophancyExperimentConfig:
    root = Path(repository_root)
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported sycophancy configuration schema")

    conditions = {
        str(condition): str(feedback_key)
        for condition, feedback_key in payload["conditions"].items()
    }
    expected_conditions = {"no_feedback", "user_likes", "user_dislikes"}
    if set(conditions) != expected_conditions:
        raise ValueError(
            f"Expected feedback conditions: {sorted(expected_conditions)}"
        )

    alpha_grid = tuple(float(value) for value in payload["normalized_alpha_grid"])
    if not alpha_grid or tuple(sorted(set(alpha_grid))) != alpha_grid:
        raise ValueError("Sycophancy alpha grid must be sorted and unique")
    if alpha_grid[0] < -1 or alpha_grid[-1] > 1 or 0.0 not in alpha_grid:
        raise ValueError("Sycophancy alpha grid must lie in [-1, 1] and include zero")

    generation_payload = payload["generation"]
    generation = SycophancyGenerationConfig(
        batch_size=int(generation_payload["batch_size"]),
        max_input_tokens=int(generation_payload["max_input_tokens"]),
        max_new_tokens=int(generation_payload["max_new_tokens"]),
        do_sample=bool(generation_payload["do_sample"]),
        seed=int(generation_payload["seed"]),
    )
    if min(
        generation.batch_size,
        generation.max_input_tokens,
        generation.max_new_tokens,
    ) <= 0:
        raise ValueError("Sycophancy generation sizes must be positive")
    if generation.do_sample:
        raise ValueError("The registered sycophancy decoder must be greedy")
    if payload["activation_endpoint"] != "assistant_start_boundary":
        raise ValueError("Unsupported sycophancy activation endpoint")

    return SycophancyExperimentConfig(
        display_name=str(payload["display_name"]),
        stimulus_file=_repository_path(payload["stimulus_file"], root),
        selection_manifest=_repository_path(payload["selection_manifest"], root),
        prompt_key=str(payload["prompt"]),
        conditions=conditions,
        normalized_alpha_grid=alpha_grid,
        generation=generation,
        activation_endpoint=str(payload["activation_endpoint"]),
        primary_state_metric=str(payload["primary_state_metric"]),
        primary_behavioral_outcome=str(payload["primary_behavioral_outcome"]),
    )


def load_sycophancy_selection(path: str | Path) -> dict[str, Any]:
    """Validate the text-free selection manifest for the upstream dataset."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported sycophancy selection-manifest schema")
    rows = payload.get("items")
    required = {
        "item_id",
        "upstream_unique_index",
        "in_intervention_subset",
        "intervention_split",
    }
    if not isinstance(rows, list) or len(rows) != 296:
        raise ValueError("Expected 296 entries in the sycophancy selection manifest")
    item_ids = []
    upstream_indices = []
    split_counts = {"calibration": 0, "holdout": 0, "": 0}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required.issubset(row):
            missing = required - set(row) if isinstance(row, dict) else required
            raise ValueError(f"Sycophancy selection row {index} is missing: {sorted(missing)}")
        item_ids.append(str(row["item_id"]))
        upstream_indices.append(int(row["upstream_unique_index"]))
        split = str(row["intervention_split"])
        if split not in split_counts:
            raise ValueError(f"Unknown intervention split in {row['item_id']}: {split}")
        in_subset = bool(row["in_intervention_subset"])
        if in_subset != (split in {"calibration", "holdout"}):
            raise ValueError(f"Intervention split mismatch in {row['item_id']}")
        split_counts[split] += 1
    if len(set(item_ids)) != 296 or len(set(upstream_indices)) != 296:
        raise ValueError("Sycophancy item IDs and upstream indices must be unique")
    if split_counts != {"calibration": 50, "holdout": 50, "": 196}:
        raise ValueError(f"Unexpected sycophancy split counts: {split_counts}")
    return payload


def load_sycophancy_items(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    required = {
        "item_id",
        "argument",
        "logical_error",
        "rating",
        "in_intervention_subset",
        "intervention_split",
    }
    if not isinstance(rows, list) or len(rows) != 296:
        raise ValueError("Expected the frozen 296-item sycophancy stimulus array")
    item_ids = []
    intervention_count = 0
    split_counts = {"calibration": 0, "holdout": 0, "": 0}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required.issubset(row):
            missing = required - set(row) if isinstance(row, dict) else required
            raise ValueError(f"Sycophancy row {index} is missing: {sorted(missing)}")
        item_ids.append(str(row["item_id"]))
        in_subset = bool(row["in_intervention_subset"])
        split = str(row["intervention_split"])
        if split not in split_counts:
            raise ValueError(f"Unknown intervention split in {row['item_id']}: {split}")
        if in_subset != (split in {"calibration", "holdout"}):
            raise ValueError(f"Intervention split mismatch in {row['item_id']}")
        intervention_count += int(in_subset)
        split_counts[split] += 1
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("Sycophancy item IDs must be unique")
    if intervention_count != 100 or split_counts != {
        "calibration": 50,
        "holdout": 50,
        "": 196,
    }:
        raise ValueError(f"Unexpected sycophancy split counts: {split_counts}")
    return rows
