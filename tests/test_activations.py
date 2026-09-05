from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vaa.activations import capture_assistant_start_activations, project_onto_vaa


class AddConstant(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, hidden):
        return hidden + self.value


class FakeBatch(dict):
    def to(self, device):
        return FakeBatch({key: value.to(device) for key, value in self.items()})


class FakeTokenizer:
    padding_side = "right"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return messages[0]["content"] + "|assistant"

    def __call__(self, texts, return_tensors, padding, truncation, max_length):
        assert return_tensors == "pt"
        assert padding is True
        assert truncation is True
        assert max_length == 32
        assert self.padding_side == "left"
        sequences = []
        for text in texts:
            content_length = len(text.split("|", 1)[0])
            sequences.append([content_length, 9])
        return FakeBatch(
            {
                "input_ids": torch.tensor(sequences),
                "attention_mask": torch.ones((len(sequences), 2), dtype=torch.long),
            }
        )


class FakeActivationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(
            layers=torch.nn.ModuleList([AddConstant(1.0), AddConstant(2.0)])
        )
        self.device = torch.device("cpu")

    def forward(self, input_ids, attention_mask):
        del attention_mask
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 2)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


def test_assistant_start_capture_uses_final_non_padding_position_and_cleans_hooks():
    model = FakeActivationModel()
    tokenizer = FakeTokenizer()
    captured = capture_assistant_start_activations(
        model,
        tokenizer,
        ["a", "longer"],
        [0, 1],
        batch_size=2,
        max_input_tokens=32,
    )
    np.testing.assert_allclose(captured[0], np.full((2, 2), 10.0))
    np.testing.assert_allclose(captured[1], np.full((2, 2), 12.0))
    assert tokenizer.padding_side == "right"
    assert not model.model.layers[0]._forward_hooks
    assert not model.model.layers[1]._forward_hooks


def test_projection_matches_reported_raw_unit_and_cosine_definitions():
    projected = project_onto_vaa(
        np.array([[3.0, 4.0], [0.0, 0.0]]),
        np.array([2.0, 0.0]),
    )
    np.testing.assert_allclose(projected["projection_raw"], [6.0, 0.0])
    np.testing.assert_allclose(projected["projection_unit"], [3.0, 0.0])
    np.testing.assert_allclose(projected["projection_cosine"], [0.6, 0.0])
    np.testing.assert_allclose(projected["activation_norm"], [5.0, 0.0])


def test_capture_rejects_invalid_layer_before_forward():
    with pytest.raises(IndexError, match="out of range"):
        capture_assistant_start_activations(
            FakeActivationModel(),
            FakeTokenizer(),
            ["a"],
            [2],
        )
