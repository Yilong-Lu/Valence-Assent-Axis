"""Repository-relative model and artifact configuration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Mapping

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = REPOSITORY_ROOT / "configs" / "models.yaml"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    model_id: str
    local_path_env: str
    target_layer: int
    hidden_size: int
    raw_alpha_range: tuple[float, float]
    vector_path: Path
    pca_metadata_path: Path
    layer_ranges_path: Path
    compatibility: str | None = None

    def resolve_model_reference(self) -> str:
        """Return an explicit local override or the registered model ID."""
        return os.environ.get(self.local_path_env, self.model_id)


def _repository_path(value: str, *, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"Configured repository path must be relative: {value}")
    return root / path


def load_model_registry(
    path: str | Path = DEFAULT_REGISTRY_PATH,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, ModelSpec]:
    """Load and validate the public model registry."""
    path = Path(path)
    root = Path(repository_root)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported model-registry schema")

    registry: dict[str, ModelSpec] = {}
    for key, record in payload.get("models", {}).items():
        alpha_range = tuple(float(value) for value in record["raw_alpha_range"])
        if len(alpha_range) != 2 or alpha_range[0] >= alpha_range[1]:
            raise ValueError(f"Invalid raw alpha range for {key}: {alpha_range}")

        registry[key] = ModelSpec(
            key=key,
            display_name=str(record["display_name"]),
            model_id=str(record["model_id"]),
            local_path_env=str(record["local_path_env"]),
            target_layer=int(record["target_layer"]),
            hidden_size=int(record["hidden_size"]),
            raw_alpha_range=alpha_range,
            vector_path=_repository_path(record["vector_path"], root=root),
            pca_metadata_path=_repository_path(
                record["pca_metadata_path"], root=root
            ),
            layer_ranges_path=_repository_path(
                record["layer_ranges_path"], root=root
            ),
            compatibility=record.get("compatibility"),
        )

    if not registry:
        raise ValueError("Model registry is empty")
    return registry


def get_model_spec(
    model_key: str,
    registry: Mapping[str, ModelSpec] | None = None,
) -> ModelSpec:
    """Return one registered model or raise a reader-facing error."""
    registry = load_model_registry() if registry is None else registry
    try:
        return registry[model_key]
    except KeyError as exc:
        valid = ", ".join(sorted(registry))
        raise KeyError(f"Unknown model '{model_key}'. Available models: {valid}") from exc


def validate_model_artifacts(spec: ModelSpec) -> None:
    """Check paths and selected-layer alpha metadata without loading a model."""
    for path in (spec.vector_path, spec.pca_metadata_path, spec.layer_ranges_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing artifact for {spec.key}: {path}")

    with spec.layer_ranges_path.open("r", encoding="utf-8") as handle:
        layer_ranges = json.load(handle)
    selected = layer_ranges.get(str(spec.target_layer))
    if selected is None:
        raise ValueError(
            f"Target layer {spec.target_layer} is absent from {spec.layer_ranges_path}"
        )
    selected_range = tuple(float(value) for value in selected)
    if selected_range != spec.raw_alpha_range:
        raise ValueError(
            f"Alpha range mismatch for {spec.key}: "
            f"registry={spec.raw_alpha_range}, metadata={selected_range}"
        )
