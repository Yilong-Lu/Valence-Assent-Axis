from vaa.sycophancy_config import (
    load_sycophancy_config,
    load_sycophancy_selection,
)


def test_sycophancy_config_matches_frozen_protocol():
    config = load_sycophancy_config()
    assert config.display_name == "Feedback-Induced Sycophancy Task"
    assert list(config.conditions) == [
        "no_feedback",
        "user_likes",
        "user_dislikes",
    ]
    assert config.generation.do_sample is False
    assert config.generation.max_input_tokens == 2048
    assert config.generation.max_new_tokens == 160
    assert config.generation.seed == 20260721
    assert config.activation_endpoint == "assistant_start_boundary"


def test_sycophancy_stimuli_have_frozen_full_and_intervention_counts():
    config = load_sycophancy_config()
    payload = load_sycophancy_selection(config.selection_manifest)
    items = payload["items"]
    assert len(items) == 296
    assert sum(item["in_intervention_subset"] for item in items) == 100
    assert sum(item["intervention_split"] == "calibration" for item in items) == 50
    assert sum(item["intervention_split"] == "holdout" for item in items) == 50
    assert payload["upstream_commit"] == "9a1694221e3639887138f61deae344335eca6752"
