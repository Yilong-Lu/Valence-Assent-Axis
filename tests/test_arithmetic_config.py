from vaa.arithmetic_config import load_arithmetic_config, load_arithmetic_items


def test_arithmetic_configuration_and_frozen_items():
    config = load_arithmetic_config()
    items = load_arithmetic_items(config.stimulus_file)
    assert config.normalized_alpha_grid == tuple(
        round(-1 + 0.2 * index, 1) for index in range(11)
    )
    assert config.primary_outcome == "candidate_accuracy"
    assert config.candidate_scoring == "full_sequence_log_probability"
    assert len(items) == 150
    assert items[0] == {
        "item_id": "arith_000",
        "a": 23,
        "b": 21,
        "correct": 44,
        "incorrect": 45,
        "offset": 1,
    }
    assert items[-1]["item_id"] == "arith_149"
