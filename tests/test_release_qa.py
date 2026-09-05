import json
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import yaml

from analysis.python.release_audit import audit_repository


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_public_checkout_has_no_release_audit_findings():
    assert audit_repository(REPOSITORY_ROOT) == []


def test_release_metadata_agree():
    citation = yaml.safe_load(
        (REPOSITORY_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    )
    project = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    assert citation["version"] == project["version"]
    assert citation["license"] == "MIT"
    assert project["license"] == {"file": "LICENSE"}
    assert (REPOSITORY_ROOT / "THIRD_PARTY_NOTICES.md").is_file()


def test_feedback_argument_text_is_prepared_from_upstream():
    tracked_text = (
        REPOSITORY_ROOT
        / "data/stimuli/feedback_induced_sycophancy/arguments.json"
    )
    selection = (
        REPOSITORY_ROOT
        / "data/stimuli/feedback_induced_sycophancy/argument_selection.json"
    )
    payload = yaml.safe_load(selection.read_text(encoding="utf-8"))
    assert not tracked_text.exists()
    assert payload["upstream_repository"] == "https://github.com/meg-tong/sycophancy-eval"
    assert payload["upstream_commit"] == "9a1694221e3639887138f61deae344335eca6752"
    assert payload["upstream_file_url"] == (
        "https://raw.githubusercontent.com/meg-tong/sycophancy-eval/"
        "9a1694221e3639887138f61deae344335eca6752/datasets/feedback.jsonl"
    )
    assert len(payload["items"]) == 296


def test_alphabetical_examples_use_displayed_order_as_ground_truth():
    pattern = re.compile(r"'([^']+)' comes before '([^']+)'")
    for path in sorted((REPOSITORY_ROOT / "examples/alpha_order").glob("*.json")):
        records = json.loads(path.read_text(encoding="utf-8"))
        for record in records:
            prompt = record["prompt_data"][0]["content"]
            match = pattern.search(prompt)
            assert match is not None
            option1, option2 = match.groups()
            expected = "right" if option1.casefold() < option2.casefold() else "wrong"
            assert record["correct_answer"] == expected
