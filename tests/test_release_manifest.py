from pathlib import Path
import csv
import hashlib
import json

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "manifest" / "artifacts.yaml"


def structured_row_count(path: Path) -> int:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("items"), list):
            return len(payload["items"])
        return len(payload)
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    raise ValueError(f"Unsupported manifest file type: {path}")


def test_artifact_manifest_has_unique_repository_relative_paths():
    payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    records = payload["vaa_vectors"]["models"].values()
    release_paths = [
        record[field]
        for record in records
        for field in ("vector", "pca_metadata", "layer_ranges")
    ]

    assert len(release_paths) == len(set(release_paths))
    assert all(not Path(path).is_absolute() for path in release_paths)
    assert all((REPOSITORY_ROOT / path).exists() for path in release_paths)


def test_artifact_manifest_matches_model_registry():
    payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    artifact_models = payload["vaa_vectors"]["models"]
    registry = yaml.safe_load(
        (REPOSITORY_ROOT / payload["model_registry"]).read_text(encoding="utf-8")
    )["models"]

    assert set(artifact_models) == set(registry)
    for key, record in artifact_models.items():
        assert record["target_layer"] == registry[key]["target_layer"]
        assert record["vector"] == registry[key]["vector_path"]


def test_stimulus_manifest_checksums_and_row_counts_match():
    manifest_path = REPOSITORY_ROOT / "manifest" / "stimuli.yaml"
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    for relative_path, record in payload["files"].items():
        path = REPOSITORY_ROOT / relative_path
        assert path.is_file()
        if "sha256" in record:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        else:
            assert record["source_url"].startswith("https://")
        assert structured_row_count(path) == record["rows"]
