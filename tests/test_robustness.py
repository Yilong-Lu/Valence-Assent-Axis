from vaa.robustness import build_robustness_items, build_robustness_prompts
from vaa.robustness_config import load_generation_robustness_config


def test_robustness_item_selection_is_balanced_and_complete():
    config = load_generation_robustness_config()
    alphabetical = build_robustness_items(config.tasks["alphabetical_order"])
    factual = build_robustness_items(config.tasks["factual_judgment"])
    assert len(alphabetical) == len(factual) == 30
    assert sum(row["correct_answer"] == "right" for row in alphabetical) == 15
    assert alphabetical[0]["option1"] == "apple"
    assert alphabetical[0]["option2"] == "banana"
    assert alphabetical[1]["option1"] == "dog"
    assert alphabetical[1]["option2"] == "cat"


def test_prompt_versions_differ_only_in_registered_spelling():
    config = load_generation_robustness_config()
    task = config.tasks["alphabetical_order"]
    items = build_robustness_items(task)[:1]
    submitted = build_robustness_prompts(
        task,
        items,
        "submitted",
        model_key="qwen25_14b",
    )[0]["prompt"]
    corrected = build_robustness_prompts(
        task,
        items,
        "corrected",
        model_key="qwen25_14b",
    )[0]["prompt"]
    assert "only anwer in JSON" in submitted
    assert "only answer in JSON" in corrected
    assert submitted.replace("anwer", "answer") == corrected
