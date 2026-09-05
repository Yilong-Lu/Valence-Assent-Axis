from types import SimpleNamespace

import torch

from vaa.generation import (
    generate_text_batch,
    generate_text_batch_with_state_capture,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        del tokenize, add_generation_prompt
        return messages[0]["content"]

    def __call__(self, texts, **kwargs):
        del kwargs
        return {
            "input_ids": torch.tensor([[1, 2] for _ in texts]),
            "attention_mask": torch.ones((len(texts), 2), dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 2)
        self.layer = torch.nn.Identity()
        self.model = SimpleNamespace(layers=[self.layer])
        self.generation_config = SimpleNamespace(eos_token_id=9)
        self.hooked_state = None

    def get_input_embeddings(self):
        return self.embedding

    def generate(self, input_ids, attention_mask, **kwargs):
        del attention_mask
        hidden = torch.zeros((len(input_ids), 1, 2))
        self.hooked_state = self.layer(hidden)
        continuation = torch.tensor([[7, 8, 9] for _ in input_ids])
        sequences = torch.cat([input_ids, continuation], dim=1)
        if not kwargs.get("return_dict_in_generate"):
            return sequences
        scores = []
        for token_id in (7, 8, 9):
            logits = torch.zeros((len(input_ids), 10))
            logits[:, token_id] = 1.0
            scores.append(logits)
        return SimpleNamespace(sequences=sequences, scores=scores)


def test_generation_decodes_only_continuation_and_applies_persistent_hook():
    model = FakeModel()
    rows = generate_text_batch(
        model,
        FakeTokenizer(),
        ["prompt one", "prompt two"],
        target_layer=0,
        steering_vector=torch.tensor([1.0, 2.0]),
        alpha_raw=0.5,
        max_input_tokens=512,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.2,
        top_p=1.0,
        top_k=50,
    )
    assert rows == [
        {
            "generated_text": "7 8",
            "generated_n_tokens": 2,
            "generation_stop_reason": "eos_token",
        },
        {
            "generated_text": "7 8",
            "generated_n_tokens": 2,
            "generation_stop_reason": "eos_token",
        },
    ]
    assert torch.allclose(
        model.hooked_state,
        torch.tensor([[[0.5, 1.0]], [[0.5, 1.0]]]),
    )
    assert torch.equal(model.layer(torch.zeros((1, 1, 2))), torch.zeros((1, 1, 2)))


def test_state_capture_records_pre_and_post_addition_projections():
    rows = generate_text_batch_with_state_capture(
        FakeModel(),
        FakeTokenizer(),
        ["prompt"],
        target_layer=0,
        steering_vector=torch.tensor([1.0, 2.0]),
        alpha_raw=0.5,
        max_input_tokens=512,
        max_new_tokens=160,
    )
    assert rows[0]["prompt_n_tokens"] == 2
    assert rows[0]["generated_token_ids"] == [7, 8]
    assert len(rows[0]["generated_token_logprobs"]) == 2
    assert rows[0]["pre_addition_vaa_projection_unit"] == 0.0
    assert abs(
        rows[0]["post_addition_vaa_projection_unit"] - (2.5 / (5.0**0.5))
    ) < 1e-6
