"""Summarize arithmetic-statement representation outputs across models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODEL_ORDER = {
    model: index
    for index, model in enumerate(
        [
            "qwen25_3b",
            "qwen25_7b",
            "qwen25_14b",
            "qwen25_32b",
            "qwen25_72b",
            "llama3_8b",
            "mistral_7b",
            "gemma2_9b",
        ]
    )
}


def summarize_file(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    model_name = path.parent.name
    required = {
        "layer",
        "r1",
        "p1",
        "representaion_similarity",
        "representaion_similarity_cosine",
        "stance_vector_variance_ratio",
        "explained_variance_ratio",
        "statements",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise KeyError(f"{path} is missing fields: {', '.join(missing)}")
    return {
        "model_name": model_name,
        "target_layer": int(data["layer"]),
        "n_items": len(data["statements"]),
        "projection_correlation": abs(float(data["r1"])),
        "projection_correlation_p": float(data["p1"]),
        "axis_pearson_correlation": abs(float(data["representaion_similarity"])),
        "axis_cosine_similarity": abs(float(data["representaion_similarity_cosine"])),
        "vaa_variance_explained": float(data["stance_vector_variance_ratio"]),
        "task_pc1_variance_explained": float(data["explained_variance_ratio"][0]),
    }


def collect_rows(input_root: Path) -> list[dict]:
    paths = sorted(input_root.glob("*/arithmetic_statement.json"))
    if not paths:
        raise FileNotFoundError(
            f"No <model>/arithmetic_statement.json files found under {input_root}"
        )
    rows = [summarize_file(path) for path in paths]
    unknown = sorted(set(row["model_name"] for row in rows).difference(MODEL_ORDER))
    if unknown:
        raise ValueError(f"Unregistered model directories: {', '.join(unknown)}")
    rows.sort(key=lambda row: MODEL_ORDER[row["model_name"]])
    return rows


def write_rows(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = collect_rows(args.input_root.resolve())
    write_rows(rows, args.output.resolve())
    print(f"Wrote {len(rows)} model summaries to {args.output}")


if __name__ == "__main__":
    main()
