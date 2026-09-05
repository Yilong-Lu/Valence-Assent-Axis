from experiments.arithmetic_answering_verification.run import (
    build_tokenization_audit,
    resolve_alpha_grid,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        mapping = {"41": [4, 1], "43": [4, 3], "True": [10], "False": [11]}
        return mapping[text]

    def convert_ids_to_tokens(self, token_ids):
        return [f"token-{token_id}" for token_id in token_ids]


def test_tokenization_audit_records_shared_numeric_prefix_without_filtering():
    prompts = [
        {
            "item_id": "arith_000",
            "prompt_id": "arith_000::direct_numeric",
            "mode": "direct_numeric",
            "statement_truth": None,
            "candidate_A": "41",
            "candidate_B": "43",
        },
        {
            "item_id": "arith_000",
            "prompt_id": "arith_000::verification_true",
            "mode": "verification_true",
            "statement_truth": True,
            "candidate_A": "True",
            "candidate_B": "False",
        },
    ]
    audit = build_tokenization_audit(prompts, FakeTokenizer())
    assert audit[0]["same_first_token"]
    assert not audit[1]["same_first_token"]


def test_alpha_override_is_sorted_and_bounded():
    assert resolve_alpha_grid((-1.0, 0.0, 1.0), [1, -1, 0]) == (-1.0, 0.0, 1.0)
