from vaa.generative_config import load_generative_config, load_json_records


def test_generative_config_registers_reported_tasks_and_decoding():
    config = load_generative_config()
    assert set(config.tasks) == {
        "alphabetical_order",
        "factual_judgment",
        "stance_taking",
    }
    assert config.normalized_alpha_grid == (
        -1.0,
        -0.8,
        -0.6,
        -0.4,
        -0.2,
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    )
    assert config.generation.temperature == 0.2
    assert config.generation.max_new_tokens == 512
    assert config.generation.do_sample is True


def test_frozen_generative_stimulus_counts():
    config = load_generative_config()
    assert {
        key: len(load_json_records(task.stimulus_file))
        for key, task in config.tasks.items()
    } == {
        "alphabetical_order": 30,
        "factual_judgment": 30,
        "stance_taking": 30,
    }
