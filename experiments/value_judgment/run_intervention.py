"""Run Value Judgment and Sentiment Analysis intervention curves."""

from __future__ import annotations

import argparse
from pathlib import Path

from vaa.config import REPOSITORY_ROOT
from vaa.experiment_config import (
    JudgmentTaskSpec,
    load_judgment_experiment_config,
    load_task_stimuli,
)
from vaa.io import write_json_atomic, write_jsonl_atomic
from vaa.models import load_model_bundle
from vaa.prompts import render_prompt
from vaa.scoring import (
    render_chat_prompt,
    score_next_token_candidates_batch,
    summarize_candidate_distribution,
)
from vaa.steering import normalized_alpha_to_raw


def batches(values, batch_size):
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def task_summary(task: JudgmentTaskSpec, max_items: int | None = None) -> dict:
    stimuli = load_task_stimuli(task)
    if max_items is not None:
        stimuli = stimuli[:max_items]
    return {
        "task": task.key,
        "prompt_key": task.prompt_key,
        "n_items": len(stimuli),
        "candidates": list(task.candidates),
    }


def run_task(bundle, task, alpha_norms, batch_size, max_items):
    stimuli = load_task_stimuli(task)
    if max_items is not None:
        stimuli = stimuli[:max_items]
    prompts = [
        render_prompt(task.prompt_key, statement=row["statement"])
        for row in stimuli
    ]
    rendered = [render_chat_prompt(bundle.tokenizer, prompt) for prompt in prompts]
    rows = []
    for alpha_norm in alpha_norms:
        alpha_raw = normalized_alpha_to_raw(
            alpha_norm,
            bundle.spec.raw_alpha_range,
        )
        scores = []
        for prompt_batch in batches(rendered, batch_size):
            scores.extend(
                score_next_token_candidates_batch(
                    bundle.model,
                    bundle.tokenizer,
                    prompt_batch,
                    task.candidates,
                    target_layer=bundle.spec.target_layer,
                    steering_vector=bundle.vaa_vector,
                    alpha_raw=alpha_raw,
                )
            )
        for stimulus, prompt, score in zip(stimuli, prompts, scores):
            rows.append(
                {
                    "model_key": bundle.spec.key,
                    "task": task.key,
                    "item_id": stimulus["item_id"],
                    "split": stimulus["split"],
                    "statement": stimulus["statement"],
                    "prompt_key": task.prompt_key,
                    "prompt": prompt,
                    "target_layer": bundle.spec.target_layer,
                    "alpha_norm": alpha_norm,
                    "alpha_raw": alpha_raw,
                    **score,
                    **summarize_candidate_distribution(
                        score["candidate_probabilities"],
                        task.candidates,
                        task.candidate_values,
                    ),
                }
            )
    return rows


def run(args: argparse.Namespace) -> None:
    config = load_judgment_experiment_config()
    task_keys = args.tasks or list(config.tasks)
    unknown = set(task_keys) - set(config.tasks)
    if unknown:
        raise ValueError(f"Unknown task keys: {sorted(unknown)}")
    alpha_norms = (
        tuple(args.alpha_norms)
        if args.alpha_norms is not None
        else config.normalized_alpha_grid
    )
    summaries = [task_summary(config.tasks[key], args.max_items) for key in task_keys]
    if args.dry_run:
        print(
            {
                "model_key": args.model,
                "alpha_norms": list(alpha_norms),
                "tasks": summaries,
                "expected_rows": sum(row["n_items"] for row in summaries)
                * len(alpha_norms),
            }
        )
        return

    bundle = load_model_bundle(
        args.model,
        device_map=args.device_map,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    output_dir = args.output_dir or (
        REPOSITORY_ROOT / "results" / "generated" / "judgment_tasks" / args.model
    )
    all_rows = []
    for key in task_keys:
        task_rows = run_task(
            bundle,
            config.tasks[key],
            alpha_norms,
            args.batch_size,
            args.max_items,
        )
        write_jsonl_atomic(output_dir / f"{key}.jsonl", task_rows)
        all_rows.extend(task_rows)
        print(f"{key}: wrote {len(task_rows)} rows")
    write_json_atomic(
        output_dir / "metadata.json",
        {
            "model_key": args.model,
            "target_layer": bundle.spec.target_layer,
            "raw_alpha_range": list(bundle.spec.raw_alpha_range),
            "normalized_alpha_grid": list(alpha_norms),
            "tasks": summaries,
            "n_rows": len(all_rows),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen25_14b")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--alpha-norms", nargs="+", type=float)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
