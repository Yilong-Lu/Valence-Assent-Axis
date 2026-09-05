import json

import numpy as np
import pytest

from vaa.config import load_model_registry, validate_model_artifacts


EXPECTED_MODELS = {
    "qwen25_3b": (26, 2048),
    "qwen25_7b": (18, 3584),
    "qwen25_14b": (28, 5120),
    "qwen25_32b": (43, 5120),
    "qwen25_72b": (52, 8192),
    "llama3_8b": (13, 4096),
    "mistral_7b": (14, 4096),
    "gemma2_9b": (22, 3584),
}


def test_registry_contains_manuscript_models():
    registry = load_model_registry()
    assert set(registry) == set(EXPECTED_MODELS)

    for key, (layer, hidden_size) in EXPECTED_MODELS.items():
        spec = registry[key]
        assert spec.target_layer == layer
        assert spec.hidden_size == hidden_size
        validate_model_artifacts(spec)


@pytest.mark.parametrize("model_key", sorted(EXPECTED_MODELS))
def test_public_vectors_are_safe_normalized_numpy_arrays(model_key):
    spec = load_model_registry()[model_key]
    vector = np.load(spec.vector_path, allow_pickle=False)

    assert vector.shape == (spec.hidden_size,)
    assert vector.dtype == np.float64
    assert np.isfinite(vector).all()
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-12)


def test_registered_alpha_ranges_match_frozen_layer_metadata():
    for spec in load_model_registry().values():
        with spec.layer_ranges_path.open("r", encoding="utf-8") as handle:
            layer_ranges = json.load(handle)
        assert tuple(layer_ranges[str(spec.target_layer)]) == spec.raw_alpha_range
