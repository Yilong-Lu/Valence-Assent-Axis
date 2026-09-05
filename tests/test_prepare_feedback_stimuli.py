from __future__ import annotations

import json

import pytest

from analysis.python.prepare_feedback_stimuli import build_stimuli, load_unique_arguments


def test_feedback_preparation_deduplicates_and_selects_text(tmp_path):
    upstream = tmp_path / "feedback.jsonl"
    records = [
        {"base": {"dataset": "arguments", "text": "First", "rating": 1}},
        {"base": {"dataset": "arguments", "text": "First", "rating": 1}},
        {"base": {"dataset": "other", "text": "Ignore"}},
        {"base": {"dataset": "arguments", "text": "Second", "rating": 2}},
    ]
    upstream.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    unique = load_unique_arguments(upstream)
    assert [row["text"] for row in unique] == ["First", "Second"]

    selection = [
        {
            "item_id": "argument_001",
            "upstream_unique_index": 1,
            "in_intervention_subset": False,
            "intervention_split": "",
        }
    ]
    assert build_stimuli(unique, selection)[0]["argument"] == "Second"

    selection[0]["upstream_unique_index"] = 9
    with pytest.raises(ValueError, match="index is absent"):
        build_stimuli(unique, selection)
