import pytest

from vaa.preference import (
    build_ordered_prompts,
    decompose_initial_state,
    decompose_order_effects,
    semantic_orientation_sign,
)


def pair(**overrides):
    values = {
        "pair_id": "pair-1",
        "pair_class": "valenced_opposite",
        "domain": "epistemic",
        "valence_status": "valenced",
        "opposition_status": "opposite",
        "word_A": "Correct",
        "word_B": "Incorrect",
    }
    values.update(overrides)
    return values


def test_preference_prompt_builder_preserves_candidate_identity_across_orders():
    rows = build_ordered_prompts(
        [pair()],
        "subjective_preference_neutral_context_v1",
    )
    assert [row["order"] for row in rows] == ["AB", "BA"]
    assert rows[0]["prompt"] == (
        "XXXXXXXXXXXXXXXXXXXXXX, Correct or Incorrect? "
        "Please answer only with Correct or Incorrect."
    )
    assert rows[1]["prompt"] == (
        "XXXXXXXXXXXXXXXXXXXXXX, Incorrect or Correct? "
        "Please answer only with Incorrect or Correct."
    )
    assert rows[0]["word_A"] == rows[1]["word_A"] == "Correct"


def test_neutral_semantic_orientation_is_alphabetically_reproducible():
    assert semantic_orientation_sign(
        pair(valence_status="neutral", word_A="apple", word_B="banana")
    ) == 1
    assert semantic_orientation_sign(
        pair(valence_status="neutral", word_A="convex", word_B="concave")
    ) == -1
    assert semantic_orientation_sign(pair()) == 1


def test_order_decomposition_separates_semantic_and_position_components():
    base = {
        "model_key": "qwen25_3b",
        "target_layer": 26,
        **pair(),
        "alpha_norm": 0.2,
        "alpha_raw": 5.0,
    }
    rows = [
        {
            **base,
            "order": "AB",
            "logprob_margin_A_minus_B": 6.0,
            "first_token_margin_A_minus_B": 4.0,
        },
        {
            **base,
            "order": "BA",
            "logprob_margin_A_minus_B": 2.0,
            "first_token_margin_A_minus_B": 0.0,
        },
    ]
    result = decompose_order_effects(rows)[0]
    assert result["semantic_component"] == 4.0
    assert result["position_component"] == 2.0
    assert result["first_token_semantic_component"] == 2.0
    assert result["first_token_position_component"] == 2.0


def test_initial_state_delta_uses_same_neutral_orientation():
    base = {
        "model_key": "qwen25_3b",
        "target_layer": 26,
        **pair(
            pair_class="neutral_opposite",
            domain="neutral_opposite",
            valence_status="neutral",
            word_A="convex",
            word_B="concave",
        ),
    }
    rows = []
    for order, value in (("AB", 3.0), ("BA", 1.0)):
        rows.append(
            {
                **base,
                "order": order,
                "projection_raw": value,
                "projection_unit": value,
                "projection_cosine": value / 10,
                "activation_norm": 10.0,
            }
        )
    result = decompose_initial_state(rows)[0]
    assert result["semantic_orientation_sign"] == -1
    assert result["projection_raw_delta_source_order"] == 2.0
    assert result["projection_raw_delta"] == -2.0


def test_order_decomposition_rejects_an_incomplete_pair():
    with pytest.raises(ValueError, match="Incomplete AB/BA"):
        decompose_order_effects(
            [
                {
                    "model_key": "qwen25_3b",
                    "target_layer": 26,
                    **pair(),
                    "order": "AB",
                    "alpha_norm": 0.0,
                    "alpha_raw": 0.0,
                    "logprob_margin_A_minus_B": 1.0,
                    "first_token_margin_A_minus_B": 1.0,
                }
            ]
        )
