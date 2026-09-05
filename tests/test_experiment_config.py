import pytest

from vaa.experiment_config import (
    load_judgment_experiment_config,
    load_task_stimuli,
)
from vaa.scoring import summarize_candidate_distribution


def test_judgment_tasks_have_frozen_175_item_sets():
    config = load_judgment_experiment_config()
    assert len(config.tasks) == 4
    assert len(config.normalized_alpha_grid) == 11
    assert config.normalized_alpha_grid[0] == -1.0
    assert config.normalized_alpha_grid[-1] == 1.0
    for task in config.tasks.values():
        rows = load_task_stimuli(task)
        assert len(rows) == 175
        assert len({row["item_id"] for row in rows}) == 175


def test_expected_response_retains_historical_unconditional_probability():
    task = load_judgment_experiment_config().tasks["sentiment_analysis_binary"]
    result = summarize_candidate_distribution(
        {"M": 0.06, "N": 0.04},
        task.candidates,
        task.candidate_values,
    )
    assert result["expected_response"] == pytest.approx(0.06)
    assert result["expected_response_conditional"] == pytest.approx(0.6)
    assert result["candidate_probability_mass"] == pytest.approx(0.1)
    assert result["response"] == "M"
