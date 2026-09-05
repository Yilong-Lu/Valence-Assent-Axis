from types import SimpleNamespace

import pytest
import torch

from vaa.scoring import (
    candidate_token_metadata,
    render_chat_prompt,
    score_candidate_sequence,
    score_first_token_candidates,
    score_next_token_candidates_batch,
    summarize_token_logprobs,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert messages == [{"role": "user", "content": "prompt"}]
        assert tokenize is False
        assert add_generation_prompt is True
        return "<chat>prompt"

    def __call__(self, text, return_tensors):
        assert text == "<chat>prompt"
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[0, 1]], dtype=torch.long)}

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        mapping = {"AB": [2, 3], "A": [2], "B": [3]}
        return mapping[text]

    def convert_ids_to_tokens(self, token_ids):
        return [f"token_{token_id}" for token_id in token_ids]


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(layers=torch.nn.ModuleList([torch.nn.Identity()]))
        self.device = torch.device("cpu")

    def forward(self, input_ids, attention_mask):
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        batch, sequence_length = input_ids.shape
        hidden = torch.zeros((batch, sequence_length, 4))
        self.model.layers[0](hidden)
        logits = torch.zeros((batch, sequence_length, 5))
        logits[0, 1, 2] = 2.0
        if sequence_length > 2:
            logits[0, 2, 3] = 1.0
        return SimpleNamespace(logits=logits)


def test_candidate_metadata_strips_leading_whitespace():
    metadata = candidate_token_metadata(FakeTokenizer(), "  AB")
    assert metadata == {
        "token_ids": [2, 3],
        "token_strings": ["token_2", "token_3"],
        "n_tokens": 2,
        "multi_token": True,
    }


def test_sequence_scoring_uses_full_candidate_and_expected_logit_positions():
    tokenizer = FakeTokenizer()
    model = FakeModel()
    rendered = render_chat_prompt(tokenizer, "prompt")
    result = score_candidate_sequence(
        model,
        tokenizer,
        rendered,
        "AB",
        target_layer=0,
        steering_vector=torch.zeros(4),
        alpha_raw=0.0,
    )
    expected_first = torch.log_softmax(
        torch.tensor([0.0, 0.0, 2.0, 0.0, 0.0]), dim=0
    )[2]
    expected_second = torch.log_softmax(
        torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]), dim=0
    )[3]
    assert result["sequence_n_tokens"] == 2
    assert result["token_logprobs"] == pytest.approx(
        [expected_first.item(), expected_second.item()]
    )
    assert result["sequence_logprob_sum"] == pytest.approx(
        (expected_first + expected_second).item()
    )
    assert not model.model.layers[0]._forward_hooks


def test_unsteered_scoring_does_not_require_a_vector():
    result = score_candidate_sequence(
        FakeModel(),
        FakeTokenizer(),
        "<chat>prompt",
        "A",
        target_layer=0,
        steering_vector=None,
        alpha_raw=0.0,
    )
    assert result["sequence_n_tokens"] == 1


def test_nonzero_intervention_requires_a_vector():
    with pytest.raises(ValueError, match="steering vector is required"):
        score_candidate_sequence(
            FakeModel(),
            FakeTokenizer(),
            "<chat>prompt",
            "A",
            target_layer=0,
            steering_vector=None,
            alpha_raw=1.0,
        )


def test_first_token_scoring_uses_prompt_only_logits_for_multitoken_candidates():
    tokenizer = FakeTokenizer()
    model = FakeModel()
    result = score_first_token_candidates(
        model,
        tokenizer,
        render_chat_prompt(tokenizer, "prompt"),
        ("AB", "B"),
        target_layer=0,
        steering_vector=torch.zeros(4),
        alpha_raw=0.0,
    )
    assert result["candidate_first_token_ids"] == {"AB": 2, "B": 3}
    assert result["candidate_logits"]["AB"] == 2.0
    assert result["candidate_logits"]["B"] == 0.0
    assert not model.model.layers[0]._forward_hooks


def test_empty_logprob_summary_is_explicit():
    summary = summarize_token_logprobs([])
    assert summary["sequence_n_tokens"] == 0
    assert summary["sequence_logprob_sum"] != summary["sequence_logprob_sum"]


class FakeBatchTokenizer:
    padding_side = "right"

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return {"A": [2], "B": [3]}[text]

    def convert_ids_to_tokens(self, token_ids):
        return [f"token_{value}" for value in token_ids]

    def __call__(
        self,
        texts,
        return_tensors,
        padding,
        truncation,
        add_special_tokens,
    ):
        assert texts == ["first", "second"]
        assert self.padding_side == "left"
        assert return_tensors == "pt"
        assert padding is True
        assert truncation is True
        assert add_special_tokens is False

        class Batch(dict):
            def to(self, device):
                return self

        return Batch(
            input_ids=torch.tensor([[0, 1], [1, 0]]),
            attention_mask=torch.ones((2, 2), dtype=torch.long),
        )


class FakeBatchModel(FakeModel):
    def forward(self, input_ids, attention_mask):
        del attention_mask
        hidden = torch.zeros((2, 2, 4))
        self.model.layers[0](hidden)
        logits = torch.zeros((2, 2, 5))
        logits[0, -1, 2] = 2.0
        logits[0, -1, 3] = 1.0
        logits[1, -1, 2] = -1.0
        logits[1, -1, 3] = 3.0
        return SimpleNamespace(logits=logits)


def test_next_token_candidate_scoring_batches_prompts_and_preserves_full_softmax():
    tokenizer = FakeBatchTokenizer()
    rows = score_next_token_candidates_batch(
        FakeBatchModel(),
        tokenizer,
        ["first", "second"],
        ("A", "B"),
        target_layer=0,
        steering_vector=torch.zeros(4),
        alpha_raw=0.0,
    )
    assert len(rows) == 2
    assert rows[0]["candidate_probabilities"]["A"] > rows[0]["candidate_probabilities"]["B"]
    assert rows[1]["candidate_probabilities"]["B"] > rows[1]["candidate_probabilities"]["A"]
    assert sum(rows[0]["candidate_probabilities"].values()) < 1.0
    assert tokenizer.padding_side == "right"
