from vaa.robustness_config import load_generation_robustness_config


def test_registered_generation_robustness_designs():
    config = load_generation_robustness_config()
    assert config.model_keys == (
        "qwen25_7b",
        "llama3_8b",
        "qwen25_14b",
        "qwen25_32b",
        "qwen25_72b",
    )
    assert set(config.tasks) == {"alphabetical_order", "factual_judgment"}
    spelling = config.protocols["prompt_spelling"]
    assert spelling.prompt_versions == ("submitted", "corrected")
    assert spelling.normalized_alpha_grid == (-0.2, 0.0, 0.2)
    temperature = config.protocols["decoding_temperature"]
    assert temperature.normalized_alpha_grid == (-0.6, -0.2, 0.0, 0.2, 0.6)
    assert [condition.value for condition in temperature.temperatures] == [
        0.0,
        0.2,
        1.0,
    ]
    assert [len(condition.seeds) for condition in temperature.temperatures] == [
        1,
        3,
        3,
    ]
