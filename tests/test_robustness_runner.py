from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from experiments.generation_robustness.runner import (
    _selected_design,
    generate_cell,
    run_protocol,
)
from vaa.robustness_config import (
    TemperatureCondition,
    load_generation_robustness_config,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        del tokenize, add_generation_prompt
        return messages[0]["content"]

    def __call__(self, text, add_special_tokens=False):
        del text, add_special_tokens
        return {"input_ids": [1, 2]}

    def decode(self, token_ids, skip_special_tokens=True):
        del token_ids, skip_special_tokens
        return '{"think": "reason", "answer": "right"}'


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
        del attention_mask, kwargs
        hidden = torch.zeros((len(input_ids), 1, 2))
        self.hooked_state = self.layer(hidden)
        continuation = torch.tensor([[7, 9] for _ in input_ids])
        return torch.cat([input_ids, continuation], dim=1)


def _dry_args():
    return Namespace(
        model="qwen25_14b",
        output_dir=None,
        tasks=None,
        alpha_values=None,
        max_items=None,
        batch_size=None,
        max_new_tokens=None,
        device_map="auto",
        dtype="bfloat16",
        local_files_only=True,
        dry_run=True,
    )


def test_dry_run_reports_registered_row_counts(capsys):
    run_protocol("prompt_spelling", _dry_args())
    assert "'n_generation_rows': 1080" in capsys.readouterr().out
    run_protocol("decoding_temperature", _dry_args())
    assert "'n_generation_rows': 2100" in capsys.readouterr().out


def test_alpha_override_must_be_registered_subset():
    protocol = load_generation_robustness_config().protocols[
        "decoding_temperature"
    ]
    assert _selected_design(protocol, [-0.6, 0.0, 0.6]) == (-0.6, 0.0, 0.6)
    with pytest.raises(ValueError, match="registered grid"):
        _selected_design(protocol, [-1.0, 0.0, 1.0])


def test_generation_cell_applies_intervention_and_parses_answer():
    model = FakeModel()
    bundle = SimpleNamespace(
        model=model,
        tokenizer=FakeTokenizer(),
        vaa_vector=torch.tensor([1.0, 2.0]),
        spec=SimpleNamespace(
            key="model",
            target_layer=0,
            raw_alpha_range=(-1.0, 1.0),
        ),
    )
    prompt_rows = [
        {
            "task": "alphabetical_order",
            "item_id": "item-0",
            "prompt_version": "submitted",
            "prompt_key": "prompt",
            "prompt": "prompt",
            "correct_answer": "right",
            "truth_direction": 1,
        }
    ]
    rows = generate_cell(
        bundle,
        prompt_rows,
        alpha_norm=0.2,
        temperature=TemperatureCondition(0.2, True, (0,)),
        seed=0,
        batch_size=1,
        max_input_tokens=512,
        max_new_tokens=512,
        top_p=1.0,
        top_k=50,
    )
    assert torch.allclose(model.hooked_state, torch.tensor([[[0.2, 0.4]]]))
    assert rows[0]["generated_token_ids"] == [7, 9]
    assert rows[0]["strict_json_valid"] is True
    assert rows[0]["answer_canonical"] == "right"
    assert rows[0]["correct"] is True
    assert rows[0]["positive_response"] is True
