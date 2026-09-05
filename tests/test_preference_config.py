from collections import Counter

from vaa.preference_config import load_preference_config, load_preference_pairs


def test_subjective_preference_config_and_stimulus_counts_are_frozen():
    config = load_preference_config()
    pairs = load_preference_pairs(config.stimulus_file)
    assert config.normalized_alpha_grid == tuple(
        round(-1 + 0.2 * index, 1) for index in range(11)
    )
    assert config.exclude_same_first_token
    assert len(pairs) == 209
    assert Counter(pair["pair_class"] for pair in pairs) == {
        "valenced_opposite": 80,
        "valenced_nonopposite": 80,
        "neutral_opposite": 24,
        "neutral_nonopposite": 25,
    }
