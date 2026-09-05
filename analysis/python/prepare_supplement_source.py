"""Extract compact source tables for the Supplementary figures.

The input is an archived experiment-results directory. Only variables used by
the manuscript plots are retained; free-text model responses and judge rationales
are deliberately omitted from these plotting tables.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = (
    "qwen25_3b",
    "qwen25_7b",
    "qwen25_14b",
    "qwen25_32b",
    "qwen25_72b",
    "llama3_8b",
    "mistral_7b",
    "gemma2_9b",
)
ALPHABETICAL_STATEMENT = re.compile(
    r"'(?P<option1>[^']+)'\s+comes before\s+'(?P<option2>[^']+)'",
    flags=re.IGNORECASE,
)


def response_type(frame: pd.DataFrame) -> pd.Series:
    pairs = list(zip(frame["factual_correctness_score"], frame["logical_consistency_score"]))
    labels = {
        (1, 1): "Sound Reasoning",
        (-1, 1): "Coherent Hallucination",
        (1, -1): "Contradictory Reasoning",
        (-1, -1): "Incoherent Hallucination",
    }
    return pd.Series((labels.get(pair, "Ambiguous/Mixed") for pair in pairs), index=frame.index)


def detailed_response_type(frame: pd.DataFrame) -> pd.Series:
    pairs = list(zip(frame["factual_correctness_score"], frame["logical_consistency_score"]))
    labels = {
        (1, 1): "Sound Reasoning",
        (-1, 1): "Coherent Hallucination",
        (1, -1): "Contradictory Reasoning",
        (-1, -1): "Incoherent Hallucination",
        (0, 1): "Cherry-picking",
        (1, 0): "Ambiguous Logic",
    }
    return pd.Series((labels.get(pair, "Mixed") for pair in pairs), index=frame.index)


def load_json(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required archived result is missing: {path}")
    return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))


def add_normalized_alpha(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    levels = sorted(result["alpha"].round(4).unique())
    if len(levels) != 11:
        raise ValueError(f"Expected 11 alpha levels, found {len(levels)}")
    mapping = {value: round(index / 5 - 1, 4) for index, value in enumerate(levels)}
    result["alpha"] = result["alpha"].round(4)
    result["alpha_norm"] = result["alpha"].map(mapping)
    return result


def correct_alphabetical_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute labels from the displayed word order in each question."""
    result = frame.copy()
    matches = result["question"].astype(str).str.extract(ALPHABETICAL_STATEMENT)
    if matches.isna().any(axis=None):
        raise ValueError("Could not parse one or more Alphabetical Order questions")
    is_true = matches["option1"].str.casefold() < matches["option2"].str.casefold()
    corrected_answer = np.where(is_true, "right", "wrong")
    changed = ~result["true_answer"].str.casefold().eq(corrected_answer)
    result.loc[changed, "correct"] = 1 - result.loc[changed, "correct"].astype(int)
    result["true_answer"] = corrected_answer
    return result


def load_evaluated(results_root: Path, model: str, experiment: str) -> pd.DataFrame:
    path = (
        results_root
        / "experiments"
        / model
        / "evaluation"
        / f"eval_{experiment}_{model}_deepseek_r1.json"
    )
    frame = add_normalized_alpha(load_json(path))
    frame["model_name"] = model
    frame["experiment"] = experiment
    frame["response_type"] = response_type(frame)
    frame["response_type_detailed"] = detailed_response_type(frame)
    if "alphabetical" in experiment:
        frame = correct_alphabetical_labels(frame)
        direction = np.where(frame["true_answer"].eq("right"), 1, -1)
    elif experiment == "TruthfulQA":
        direction = np.where(frame["true_answer"].str.split(",").str[0].eq("Yes"), 1, -1)
    else:
        direction = 1
    frame["alignment_pressure"] = frame["alpha_norm"] * direction
    return frame


def prepare_alphabetical(results_root: Path) -> pd.DataFrame:
    frames = []
    for model in MODELS:
        for experiment in ("alphabetical_think_answer", "alphabetical_answer_think"):
            frame = load_evaluated(results_root, model, experiment)
            frames.append(
                frame[
                    [
                        "model_name",
                        "experiment",
                        "i",
                        "alpha",
                        "alpha_norm",
                        "alignment_pressure",
                        "correct",
                        "response_type",
                    ]
                ]
            )
    return pd.concat(frames, ignore_index=True)


def prepare_factual(results_root: Path) -> pd.DataFrame:
    frames = []
    for model in MODELS:
        frame = load_evaluated(results_root, model, "TruthfulQA")
        frames.append(
            frame[
                [
                    "model_name",
                    "i",
                    "alpha",
                    "alpha_norm",
                    "alignment_pressure",
                    "correct",
                    "response_type_detailed",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True)


def prepare_stance(results_root: Path) -> pd.DataFrame:
    frames = []
    for model in MODELS:
        evaluated = load_evaluated(results_root, model, "attitude_critical")
        evaluation_dir = results_root / "experiments" / model / "evaluation"
        answer = add_normalized_alpha(
            load_json(evaluation_dir / f"stance_stance_{model}_qwen3_32b.json")
        )
        reasoning = add_normalized_alpha(
            load_json(evaluation_dir / f"stance_think_stance_{model}_qwen3_32b.json")
        )
        keys = ["question", "alpha"]
        answer = answer[keys + ["stance_score"]].rename(
            columns={"stance_score": "answer_stance"}
        )
        reasoning = reasoning[keys + ["stance_score"]].rename(
            columns={"stance_score": "reasoning_stance"}
        )
        compact = evaluated.merge(answer, on=keys, how="left", validate="many_to_one")
        compact = compact.merge(reasoning, on=keys, how="left", validate="many_to_one")
        compact["alignment_pressure"] = compact["alpha_norm"]
        missing = int(compact[["answer_stance", "reasoning_stance"]].isna().sum().sum())
        if missing:
            print(f"Retaining {missing} missing judge score(s) for {model}")
        frames.append(
            compact[
                [
                    "model_name",
                    "i",
                    "alpha",
                    "alpha_norm",
                    "alignment_pressure",
                    "answer_stance",
                    "reasoning_stance",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/source_data"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "supplement_alphabetical_order.csv": prepare_alphabetical(args.results_root),
        "supplement_factual_judgment.csv": prepare_factual(args.results_root),
        "supplement_stance_taking.csv": prepare_stance(args.results_root),
    }
    for name, frame in outputs.items():
        destination = args.output_dir / name
        frame.to_csv(destination, index=False)
        print(f"Wrote {len(frame):,} rows to {destination}")


if __name__ == "__main__":
    main()
