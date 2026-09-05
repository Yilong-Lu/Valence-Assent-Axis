from dataclasses import replace
from types import SimpleNamespace

import pytest

from experiments.feedback_induced_sycophancy import common
from experiments.feedback_induced_sycophancy.common import resolve_protocol_design
from vaa.sycophancy_config import (
    load_sycophancy_config,
    load_sycophancy_selection,
)


def test_feedback_and_intervention_protocols_select_expected_rows():
    config = load_sycophancy_config()
    items = load_sycophancy_selection(config.selection_manifest)["items"]
    feedback_items, feedback_alpha = resolve_protocol_design(
        "feedback_effect",
        items,
        config.normalized_alpha_grid,
        None,
    )
    intervention_items, intervention_alpha = resolve_protocol_design(
        "intervention",
        items,
        config.normalized_alpha_grid,
        None,
    )
    assert len(feedback_items) == 296
    assert feedback_alpha == (0.0,)
    assert len(intervention_items) == 100
    assert intervention_alpha == config.normalized_alpha_grid


def test_feedback_protocol_rejects_nonzero_intervention():
    config = load_sycophancy_config()
    items = load_sycophancy_selection(config.selection_manifest)["items"]
    with pytest.raises(ValueError, match="only at alpha=0"):
        resolve_protocol_design(
            "feedback_effect",
            items,
            config.normalized_alpha_grid,
            [-0.2, 0.0, 0.2],
        )


def test_dry_run_uses_selection_manifest_without_upstream_text(
    monkeypatch, tmp_path, capsys
):
    config = load_sycophancy_config()
    config = replace(config, stimulus_file=tmp_path / "missing_arguments.json")
    monkeypatch.setattr(common, "load_sycophancy_config", lambda: config)
    args = SimpleNamespace(
        model="qwen25_14b",
        alpha_values=None,
        max_items=None,
        batch_size=None,
        max_new_tokens=None,
        seed=None,
        dry_run=True,
    )

    common.run_protocol("feedback_effect", args)

    output = capsys.readouterr().out
    assert "'n_items_configured': 296" in output
    assert "'n_prompt_rows': 888" in output
    assert "'n_generation_rows': 888" in output
