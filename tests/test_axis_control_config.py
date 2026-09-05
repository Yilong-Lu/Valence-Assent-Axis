from collections import Counter

from vaa.axis_control_config import load_axis_control_config, load_record_array


def test_axis_control_stimuli_match_frozen_counts_and_balance():
    config = load_axis_control_config()
    valence = load_record_array(config.valence.stimulus_file)
    letters = load_record_array(config.single_letter_order.stimulus_file)

    assert len(valence) == 80
    assert Counter(row["domain"] for row in valence) == {
        "epistemic": 20,
        "utilitarian": 20,
        "deontic": 20,
        "affective": 20,
    }
    assert len({row["pair_id"] for row in valence}) == 80

    assert len(letters) == 50
    assert len({row["pair_id"] for row in letters}) == 50
    assert all(
        row["earlier_letter"] < row["later_letter"] for row in letters
    )


def test_answer_label_conditions_are_matched_except_for_labels():
    conditions = load_axis_control_config().single_letter_order.answer_label_conditions
    right_wrong = conditions["right_wrong"]
    true_false = conditions["true_false"]
    assert (right_wrong.true_label, right_wrong.false_label) == ("right", "wrong")
    assert (true_false.true_label, true_false.false_label) == ("true", "false")
    assert right_wrong.protocol == "r1_original_right_wrong_v1"
    assert true_false.protocol == "r1_matched_answer_labels_v2"
