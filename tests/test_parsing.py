from vaa.parsing import (
    extract_unambiguous_answer_field,
    extract_outermost_mapping,
    parse_json_answer,
    parse_label_response,
    parse_strong_weak_verdict,
)


def test_verdict_parser_accepts_repeated_identical_terminal_labels():
    result = parse_strong_weak_verdict(
        "Verdict: Strong.\nBrief reason.\nFinal verdict: Strong."
    )
    assert result["verdict_valid"] is True
    assert result["verdict_label"] == "STRONG"
    assert result["verdict_strong"] == 1
    assert result["verdict_terminal_period"] is True


def test_verdict_parser_rejects_conflicting_or_missing_labels():
    conflicting = parse_strong_weak_verdict(
        "Verdict: Strong.\nFinal verdict: Weak."
    )
    missing = parse_strong_weak_verdict("This is a strong argument.")
    assert conflicting["verdict_valid"] is False
    assert missing["verdict_valid"] is False


def test_label_parser_matches_historical_token_logic():
    assert parse_label_response("Right, because...", "right", "wrong") == "right"
    assert parse_label_response("This is wrong.", "right", "wrong") == "wrong"
    assert parse_label_response("unclear", "right", "wrong") == "other"


def test_safe_mapping_extraction_accepts_json_and_python_literal():
    assert extract_outermost_mapping('text {"think": "x", "answer": "yes"}') == {
        "think": "x",
        "answer": "yes",
    }
    assert extract_outermost_mapping("text {'think': 'x', 'answer': 'yes'}") == {
        "think": "x",
        "answer": "yes",
    }


def test_json_answer_separates_format_compliance_from_answer_availability():
    strict = parse_json_answer(
        '{"think": "x", "answer": "Right"}',
        allowed_answers={"right", "wrong"},
        correct_answer="right",
    )
    extra_field = parse_json_answer(
        '{"think": "x", "answer": "right", "extra": 1}',
        allowed_answers={"right", "wrong"},
        correct_answer="right",
    )
    python_literal = parse_json_answer(
        "{'think': 'x', 'answer': 'right'}",
        allowed_answers={"right", "wrong"},
        correct_answer="right",
    )
    assert strict["strict_json_valid"] is True
    assert strict["answer_valid"] is True
    assert strict["correct"] is True
    assert extra_field["strict_json_valid"] is False
    assert extra_field["answer_valid"] is True
    assert python_literal["strict_json_valid"] is False
    assert python_literal["answer_valid"] is True


def test_relaxed_answer_field_requires_one_unambiguous_label():
    assert extract_unambiguous_answer_field(
        'text "answer": "Yes"', allowed_answers={"yes", "no"}
    ) == "yes"
    assert extract_unambiguous_answer_field(
        '"answer": "Yes" then "answer": "No"',
        allowed_answers={"yes", "no"},
    ) is None
