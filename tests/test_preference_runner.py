from experiments.subjective_preference.run import (
    build_tokenization_audit,
    resolve_alpha_grid,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        mapping = {"first": [1], "second": [2], "shared A": [3, 4], "shared B": [3, 5]}
        return mapping[text]

    def convert_ids_to_tokens(self, token_ids):
        return [f"token-{token_id}" for token_id in token_ids]


def test_tokenization_audit_marks_shared_first_candidate_tokens():
    pairs = [
        {"pair_id": "distinct", "word_A": "first", "word_B": "second"},
        {"pair_id": "shared", "word_A": "shared A", "word_B": "shared B"},
    ]
    audit = build_tokenization_audit(pairs, FakeTokenizer())
    assert not audit[0]["same_first_token"]
    assert audit[1]["same_first_token"]
    assert audit[1]["candidate_A"]["multi_token"]


def test_alpha_override_is_sorted_and_validated():
    assert resolve_alpha_grid((-1.0, 0.0, 1.0), [1, -1, 0]) == (-1.0, 0.0, 1.0)
