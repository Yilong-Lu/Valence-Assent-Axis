from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vaa.steering import (
    create_steering_hook,
    load_vaa_vector,
    normalized_alpha_to_raw,
)


def test_normalized_alpha_uses_separate_negative_and_positive_ranges():
    raw_range = (-41.9, 58.0)
    assert normalized_alpha_to_raw(-0.2, raw_range) == pytest.approx(-8.38)
    assert normalized_alpha_to_raw(0.2, raw_range) == pytest.approx(11.6)
    assert normalized_alpha_to_raw(0.0, raw_range) == 0.0
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        normalized_alpha_to_raw(1.1, raw_range)


def test_persistent_hook_modifies_every_sequence_position_and_preserves_tuple():
    hidden = torch.zeros((2, 3, 4), dtype=torch.float16)
    vector = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    hook = create_steering_hook(vector, 0.5)

    tensor_output = hook(None, None, hidden)
    expected = 0.5 * vector.to(torch.float16)
    assert tensor_output.dtype == torch.float16
    assert torch.equal(tensor_output, expected.expand_as(hidden))

    tuple_output = hook(None, None, (hidden, "cache"))
    assert torch.equal(tuple_output[0], expected.expand_as(hidden))
    assert tuple_output[1] == "cache"


def test_vector_loader_rejects_wrong_shape(tmp_path):
    path = tmp_path / "vector.npy"
    np.save(path, np.ones(3, dtype=np.float32), allow_pickle=False)
    spec = SimpleNamespace(key="test", vector_path=path, hidden_size=4)
    with pytest.raises(ValueError, match="Unexpected vector shape"):
        load_vaa_vector(spec)
