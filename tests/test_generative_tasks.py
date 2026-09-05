from vaa.generative_config import load_generative_config, load_json_records
from vaa.generative_tasks import (
    build_alphabetical_prompts,
    build_factual_prompts,
    build_stance_prompts,
    parse_task_response,
)


def test_alphabetical_builder_balances_order_truth_and_conditions():
    config = load_generative_config()
    task = config.tasks["alphabetical_order"]
    items = load_json_records(task.stimulus_file)
    rows = build_alphabetical_prompts(items, task.conditions, model_key="qwen25_14b")
    assert len(rows) == 120
    assert sum(row["statement_truth"] for row in rows) == 60
    assert {row["condition"] for row in rows} == {
        "think_then_answer",
        "answer_then_think",
    }
    first = rows[0]
    assert first["prompt"].startswith(
        "'apple' comes before 'banana' in alphabetical order."
    )
    assert "Please only anwer in JSON format" in first["prompt"]
    assert first["correct_answer"] == "right"
    assert first["truth_direction"] == 1
    assert rows[1]["correct_answer"] == "wrong"
    assert rows[1]["truth_direction"] == -1
    corrected_pair = [
        row
        for row in rows
        if row["item_id"] == "alphabetical_016"
        and row["condition"] == "think_then_answer"
        and row["order"] == "a_first"
    ][0]
    assert corrected_pair["option1"] == "hard"
    assert corrected_pair["option2"] == "head"
    assert corrected_pair["statement_truth"] is True
    assert all(
        row["statement_truth"]
        == (row["option1"].casefold() < row["option2"].casefold())
        for row in rows
    )


def test_factual_and_stance_builders_match_frozen_first_prompts():
    config = load_generative_config()
    factual = config.tasks["factual_judgment"]
    factual_items = load_json_records(factual.stimulus_file)
    factual_rows = build_factual_prompts(
        factual_items,
        factual.prompt_key,
        model_key="qwen25_14b",
    )
    assert len(factual_rows) == 30
    assert factual_rows[0]["correct_answer"] == "yes"
    assert factual_rows[0]["truth_direction"] == 1
    assert factual_rows[0]["prompt"].startswith(
        "Answer the question below.\nQuestion: Are there developed countries"
    )

    stance = config.tasks["stance_taking"]
    stance_items = load_json_records(stance.stimulus_file)
    stance_rows = build_stance_prompts(
        stance_items,
        stance.prompt_key,
        model_key="qwen25_14b",
    )
    assert len(stance_rows) == 30
    assert stance_rows[0]["prompt"].startswith(
        "Critically evaluate the following statement: 'Students should"
    )


def test_objective_parser_separates_strict_format_from_usable_answer():
    malformed_reasoning = (
        '{"think": "line one\nline two", "answer": "right"}'
    )
    parsed = parse_task_response(
        "alphabetical_order",
        malformed_reasoning,
        "right",
    )
    assert parsed["strict_json_valid"] is False
    assert parsed["answer_valid"] is True
    assert parsed["answer_parse_mode"] == "quoted_answer_field"
    assert parsed["correct"] is True


def test_stance_parser_retains_reason_and_answer_without_scoring_stance():
    parsed = parse_task_response(
        "stance_taking",
        '{"think": "Evidence is mixed.", "answer": "I disagree."}',
        None,
    )
    assert parsed == {
        "json_object_found": True,
        "strict_json_valid": True,
        "think": "Evidence is mixed.",
        "answer": "I disagree.",
    }
