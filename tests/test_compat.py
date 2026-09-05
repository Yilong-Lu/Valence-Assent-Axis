from types import SimpleNamespace

from vaa.compat import (
    MISTRAL_V03_ORIGINAL_BLOCK,
    MISTRAL_V03_REPLACEMENT_BLOCK,
    apply_model_compatibility,
    patch_mistral_v03_chat_template,
)


def test_mistral_patch_is_exact_and_idempotent():
    tokenizer = SimpleNamespace(
        chat_template=f"prefix\n{MISTRAL_V03_ORIGINAL_BLOCK}\nsuffix"
    )
    assert patch_mistral_v03_chat_template(tokenizer) is True
    assert MISTRAL_V03_ORIGINAL_BLOCK not in tokenizer.chat_template
    assert MISTRAL_V03_REPLACEMENT_BLOCK in tokenizer.chat_template
    assert patch_mistral_v03_chat_template(tokenizer) is False


def test_registered_compatibility_is_not_applied_to_other_models():
    tokenizer = SimpleNamespace(chat_template=MISTRAL_V03_ORIGINAL_BLOCK)
    model_spec = SimpleNamespace(key="qwen25_7b", compatibility=None)
    assert apply_model_compatibility(tokenizer, model_spec) is False
    assert tokenizer.chat_template == MISTRAL_V03_ORIGINAL_BLOCK
