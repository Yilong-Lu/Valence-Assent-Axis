"""Configuration and frozen stimuli for the Subjective Preference task."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import yaml

from .config import REPOSITORY_ROOT


DEFAULT_PREFERENCE_CONFIG = (
    REPOSITORY_ROOT / "configs" / "experiments" / "subjective_preference.yaml"
)


@dataclass(frozen=True)
class PreferenceExperimentConfig:
    display_name: str
    prompt_key: str
    stimulus_file: Path
    normalized_alpha_grid: tuple[float, ...]
    exclude_same_first_token: bool
    neutral_orientation: str


def _repository_path(value: str, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"Preference path must be repository-relative: {value}")
    return root / path


def load_preference_config(
    path: str | Path = DEFAULT_PREFERENCE_CONFIG,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> PreferenceExperimentConfig:
    root = Path(repository_root)
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported Subjective Preference configuration schema")

    alpha_grid = tuple(float(value) for value in payload["normalized_alpha_grid"])
    if not alpha_grid or sorted(set(alpha_grid)) != list(alpha_grid):
        raise ValueError("Preference alpha grid must contain unique sorted values")
    if alpha_grid[0] < -1 or alpha_grid[-1] > 1 or 0.0 not in alpha_grid:
        raise ValueError("Preference alpha grid must lie in [-1, 1] and include zero")
    orientation = str(payload["neutral_orientation"])
    if orientation != "ascii_case_insensitive_alphabetical":
        raise ValueError(f"Unsupported neutral orientation: {orientation}")

    return PreferenceExperimentConfig(
        display_name=str(payload["display_name"]),
        prompt_key=str(payload["prompt"]),
        stimulus_file=_repository_path(payload["stimulus_file"], root),
        normalized_alpha_grid=alpha_grid,
        exclude_same_first_token=bool(payload["exclude_same_first_token"]),
        neutral_orientation=orientation,
    )


def load_preference_pairs(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    required = {
        "pair_id",
        "pair_class",
        "domain",
        "valence_status",
        "opposition_status",
        "word_A",
        "word_B",
    }
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a nonempty JSON array: {path}")
    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required.issubset(row):
            missing = required - set(row) if isinstance(row, dict) else required
            raise ValueError(f"Preference row {index} is missing fields: {sorted(missing)}")
        normalized.append({str(key): str(value) for key, value in row.items()})
    if len({row["pair_id"] for row in normalized}) != len(normalized):
        raise ValueError("Preference pair IDs must be unique")
    return normalized
