#!/usr/bin/env python3
"""Build a uniform answer-label summary from Single-Letter Order outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


FIELDS = [
    "model_name",
    "answer_pair",
    "prompt_protocol",
    "target_layer",
    "n_prompts",
    "pca_n_components",
    "pc1_d_projection_r",
    "pc1_d_projection_p",
    "abs_pc1_d_projection_r",
    "pc2_d_projection_r",
    "pc2_d_projection_p",
    "direct_pc1_axis_pearson",
    "abs_direct_pc1_axis_pearson",
    "direct_pc1_axis_cosine",
    "abs_direct_pc1_axis_cosine",
    "d_stance_variance_ratio",
    "response_parse_rate",
    "response_accuracy_raw_parser",
    "response_counts",
    "true_answer_counts",
    "source_json",
    "prompt_template",
]

MODEL_ORDER = {
    model_name: index
    for index, model_name in enumerate(
        (
            "qwen25_3b",
            "qwen25_7b",
            "llama3_8b",
            "mistral_7b",
            "gemma2_9b",
            "qwen25_14b",
            "qwen25_32b",
            "qwen25_72b",
        )
    )
}
ANSWER_PAIR_ORDER = {"right_wrong": 0, "true_false": 1}
MATCHED_TRUE_FALSE_PROTOCOL = "r1_matched_answer_labels_v2"


def count_json(values: list[Any]) -> str:
    counts = Counter(str(value).lower() for value in values if value is not None)
    return json.dumps(dict(sorted(counts.items())), sort_keys=True)


def summarize_file(
    path: Path, root: Path, allow_legacy_true_false: bool = False
) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    answer_pair = data["answer_pair"]
    prompt_protocol = data.get("prompt_protocol", "legacy_unspecified")
    if (
        answer_pair == "true_false"
        and prompt_protocol != MATCHED_TRUE_FALSE_PROTOCOL
        and not allow_legacy_true_false
    ):
        raise ValueError(
            f"{path} is not a matched true/false output: "
            f"prompt_protocol={prompt_protocol!r}. Rerun with the current "
            "run_experiments_word_assent.py or pass --allow-legacy-true-false "
            "for provenance-only summaries."
        )
    pc1_r = float(data["r1"])
    direct_pearson = float(data["representaion_similarity"])
    direct_cosine = float(data["representaion_similarity_cosine"])
    return {
        "model_name": path.parent.name,
        "answer_pair": answer_pair,
        "prompt_protocol": prompt_protocol,
        "target_layer": int(data["layer"]),
        "n_prompts": len(data["prompt"]),
        "pca_n_components": int(data["pca_n_components"]),
        "pc1_d_projection_r": pc1_r,
        "pc1_d_projection_p": float(data["p1"]),
        "abs_pc1_d_projection_r": abs(pc1_r),
        "pc2_d_projection_r": float(data["r2"]),
        "pc2_d_projection_p": float(data["p2"]),
        "direct_pc1_axis_pearson": direct_pearson,
        "abs_direct_pc1_axis_pearson": abs(direct_pearson),
        "direct_pc1_axis_cosine": direct_cosine,
        "abs_direct_pc1_axis_cosine": abs(direct_cosine),
        "d_stance_variance_ratio": float(data["stance_vector_variance_ratio"]),
        "response_parse_rate": float(data["response_parse_rate"]),
        "response_accuracy_raw_parser": float(data["response_accuracy"]),
        "response_counts": count_json(data["parsed_response"]),
        "true_answer_counts": count_json(data["true_answer"]),
        "source_json": str(path.relative_to(root)),
        "prompt_template": data["prompt_template"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-legacy-true-false",
        action="store_true",
        help="Allow superseded true/false outputs for provenance-only summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.result_root.resolve()
    paths = sorted(root.glob("*/pca_projection_layer*_word_assent*.json"))
    if not paths:
        raise FileNotFoundError(f"No Single-Letter Order JSON files under {root}")
    rows = [
        summarize_file(path, root, args.allow_legacy_true_false) for path in paths
    ]
    rows.sort(
        key=lambda row: (
            MODEL_ORDER.get(row["model_name"], len(MODEL_ORDER)),
            row["model_name"],
            ANSWER_PAIR_ORDER.get(row["answer_pair"], len(ANSWER_PAIR_ORDER)),
            row["answer_pair"],
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} answer-label rows to {args.output.resolve()}")


if __name__ == "__main__":
    main()
