from types import SimpleNamespace

import pytest
import torch

import vaa.models as model_module
from vaa.models import get_transformer_layers, load_model_bundle, resolve_torch_dtype


def test_dtype_aliases_are_explicit():
    assert resolve_torch_dtype("bfloat16") is torch.bfloat16
    assert resolve_torch_dtype(torch.float32) is torch.float32
    with pytest.raises(ValueError, match="Available dtypes"):
        resolve_torch_dtype("auto")


def test_transformer_layer_lookup_has_reader_facing_failure():
    layers = [object(), object()]
    assert get_transformer_layers(SimpleNamespace(model=SimpleNamespace(layers=layers))) is layers
    with pytest.raises(AttributeError, match="model.layers"):
        get_transformer_layers(SimpleNamespace())


def test_model_bundle_can_load_without_an_existing_vector(monkeypatch):
    model = SimpleNamespace(eval=lambda: None)
    tokenizer = SimpleNamespace(
        pad_token_id=None,
        pad_token=None,
        eos_token="<eos>",
        padding_side="right",
    )
    spec = SimpleNamespace(resolve_model_reference=lambda: "local-model")
    monkeypatch.setattr(model_module, "get_model_spec", lambda _key: spec)
    monkeypatch.setattr(
        model_module.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: model,
    )
    monkeypatch.setattr(
        model_module.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: tokenizer,
    )
    monkeypatch.setattr(model_module, "apply_model_compatibility", lambda *_args: False)
    monkeypatch.setattr(
        model_module,
        "load_vaa_vector",
        lambda _spec: pytest.fail("The vector loader must not be called"),
    )

    bundle = load_model_bundle("test", device_map="cpu", load_vector=False)

    assert bundle.vaa_vector is None
    assert bundle.tokenizer.pad_token == "<eos>"
    assert bundle.tokenizer.padding_side == "left"
