import pytest

from experiments.generative_reasoning import resolve_alpha_grid, select_conditions
from vaa.generative_config import load_generative_config


def test_alpha_override_is_sorted_and_bounded():
    assert resolve_alpha_grid((-1.0, 0.0, 1.0), [1, -1, 0]) == (-1.0, 0.0, 1.0)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        resolve_alpha_grid((-1.0, 0.0, 1.0), [-1.1, 0])


def test_condition_selection_is_explicit_and_validated():
    task = load_generative_config().tasks["alphabetical_order"]
    selected = select_conditions(task, ["think_then_answer"])
    assert list(selected) == ["think_then_answer"]
    with pytest.raises(ValueError, match="Unknown conditions"):
        select_conditions(task, ["missing"])
