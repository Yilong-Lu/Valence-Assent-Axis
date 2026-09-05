from experiments.single_letter_order.build_axis import (
    binary_auc,
    build_items as build_letter_items,
)
from experiments.valence.build_axis import build_items as build_valence_items
from vaa.axis_control_config import load_axis_control_config, load_record_array


def test_valence_builder_produces_160_balanced_word_prompts():
    spec = load_axis_control_config().valence
    records = load_record_array(spec.stimulus_file)
    items = build_valence_items(records, spec.prompt_key)
    assert len(items) == 160
    assert sum(item["valence_label"] == "positive" for item in items) == 80
    assert sum(item["valence_label"] == "negative" for item in items) == 80
    assert items[0]["word"] == "Correct"
    assert "Word: 'correct'" in items[0]["prompt"]


def test_single_letter_builder_preserves_matched_statement_content():
    spec = load_axis_control_config().single_letter_order
    records = load_record_array(spec.stimulus_file)
    right_wrong = build_letter_items(
        records, spec.answer_label_conditions["right_wrong"], "qwen25_14b"
    )
    true_false = build_letter_items(
        records, spec.answer_label_conditions["true_false"], "qwen25_14b"
    )
    assert len(right_wrong) == len(true_false) == 100
    assert sum(item["statement_true"] for item in right_wrong) == 50
    for original, control in zip(right_wrong, true_false):
        assert original["option1"] == control["option1"]
        assert original["option2"] == control["option2"]
        assert original["statement_true"] == control["statement_true"]
        assert original["prompt"].replace("right", "true").replace(
            "wrong", "false"
        ) == control["prompt"]


def test_single_letter_builder_applies_mistral_prefix_to_both_conditions():
    spec = load_axis_control_config().single_letter_order
    records = load_record_array(spec.stimulus_file)[:1]
    for condition in spec.answer_label_conditions.values():
        items = build_letter_items(records, condition, "mistral_7b")
        assert items[0]["prompt"].startswith("In the standard English alphabet order")


def test_binary_auc_is_orientation_invariant_like_one_predictor_logistic_auc():
    assert binary_auc([0.0, 1.0, 2.0, 3.0], [False, False, True, True]) == 1.0
    assert binary_auc([3.0, 2.0, 1.0, 0.0], [False, False, True, True]) == 1.0
