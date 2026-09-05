"""Shared execution layer for registered generation robustness checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from vaa.config import REPOSITORY_ROOT
from vaa.generation import set_generation_seed
from vaa.generative_tasks import parse_task_response
from vaa.io import read_jsonl, write_json_atomic, write_jsonl_atomic
from vaa.models import load_model_bundle
from vaa.robustness import build_robustness_items, build_robustness_prompts
from vaa.robustness_config import (
    RobustnessProtocolSpec,
    TemperatureCondition,
    load_generation_robustness_config,
)
from vaa.scoring import render_chat_prompt, steering_intervention
from vaa.steering import normalized_alpha_to_raw


PROTOCOLS = {"prompt_spelling", "decoding_temperature"}


def _identity_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _label(value: float) -> str:
    return f"{value:+.1f}".replace("+", "p").replace("-", "m").replace(".", "p")


def _temperature_label(value: float) -> str:
    return f"{value:.1f}".replace(".", "p")


def _cell_path(
    root: Path,
    task: str,
    prompt_version: str,
    temperature: float,
    alpha_norm: float,
    seed: int,
) -> Path:
    return root / (
        f"{task}__{prompt_version}__temp_{_temperature_label(temperature)}"
        f"__alpha_{_label(alpha_norm)}__seed_{seed}.jsonl"
    )


def _normalize_eos_ids(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(token_id) for token_id in value}


def _encode_prompts(
    tokenizer: Any,
    prompt_rows: list[dict[str, Any]],
    max_input_tokens: int,
) -> list[dict[str, Any]]:
    examples = []
    for row in prompt_rows:
        chat_text = render_chat_prompt(tokenizer, row["prompt"])
        prompt_ids = list(
            tokenizer(chat_text, add_special_tokens=False)["input_ids"]
        )
        if not prompt_ids or len(prompt_ids) > max_input_tokens:
            raise ValueError(
                f"Invalid prompt length for {row['item_id']}: "
                f"{len(prompt_ids)} (limit {max_input_tokens})"
            )
        examples.append({**row, "prompt_ids": prompt_ids})
    return examples


def _pad_sequences(
    sequences: list[list[int]],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), width),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row_index, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[row_index, -length:] = torch.tensor(
            sequence,
            dtype=torch.long,
            device=device,
        )
        attention_mask[row_index, -length:] = 1
    return input_ids, attention_mask


def _batches(values: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def generate_cell(
    bundle: Any,
    prompt_rows: list[dict[str, Any]],
    *,
    alpha_norm: float,
    temperature: TemperatureCondition,
    seed: int,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
) -> list[dict[str, Any]]:
    set_generation_seed(seed)
    examples = _encode_prompts(bundle.tokenizer, prompt_rows, max_input_tokens)
    alpha_raw = round(
        normalized_alpha_to_raw(alpha_norm, bundle.spec.raw_alpha_range),
        6,
    )
    input_device = bundle.model.get_input_embeddings().weight.device
    pad_token_id = int(bundle.tokenizer.pad_token_id)
    eos = getattr(bundle.model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = bundle.tokenizer.eos_token_id
    eos_ids = _normalize_eos_ids(eos)
    rows = []
    for batch in _batches(examples, batch_size):
        input_ids, attention_mask = _pad_sequences(
            [row["prompt_ids"] for row in batch],
            pad_token_id,
            input_device,
        )
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "do_sample": temperature.sampled,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos,
        }
        if temperature.sampled:
            generation_kwargs.update(
                {"temperature": temperature.value, "top_p": top_p, "top_k": top_k}
            )
        with steering_intervention(
            bundle.model,
            bundle.spec.target_layer,
            bundle.vaa_vector,
            alpha_raw,
        ), torch.inference_mode():
            output_ids = bundle.model.generate(**generation_kwargs)
        prompt_width = int(input_ids.shape[1])
        for row_index, example in enumerate(batch):
            token_ids = output_ids[row_index, prompt_width:].detach().cpu().tolist()
            stop_reason = "max_new_tokens"
            for token_index, token_id in enumerate(token_ids):
                if token_id in eos_ids:
                    token_ids = token_ids[: token_index + 1]
                    stop_reason = "eos_token"
                    break
            generated_text = bundle.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
            ).strip()
            parsed = parse_task_response(
                str(example["task"]),
                generated_text,
                str(example["correct_answer"]),
            )
            answer = parsed.get("answer_canonical")
            rows.append(
                {
                    **{key: value for key, value in example.items() if key != "prompt_ids"},
                    "model_key": bundle.spec.key,
                    "target_layer": bundle.spec.target_layer,
                    "prompt_n_tokens": len(example["prompt_ids"]),
                    "alpha_norm": float(alpha_norm),
                    "alpha_raw": alpha_raw,
                    "alignment_pressure_norm": float(
                        alpha_norm * int(example["truth_direction"])
                    ),
                    "temperature": temperature.value,
                    "do_sample": temperature.sampled,
                    "top_p": top_p if temperature.sampled else None,
                    "top_k": top_k if temperature.sampled else None,
                    "seed": seed,
                    "generated_text": generated_text,
                    "generated_token_ids": token_ids,
                    "generated_n_tokens": len(token_ids),
                    "generation_stop_reason": stop_reason,
                    **parsed,
                    "positive_response": (
                        bool(answer in {"right", "yes"}) if answer is not None else None
                    ),
                }
            )
    return rows


def _validate_cell(
    rows: list[dict[str, Any]],
    expected_item_ids: set[str],
    *,
    prompt_version: str,
    temperature: float,
    alpha_norm: float,
    seed: int,
) -> None:
    if {str(row["item_id"]) for row in rows} != expected_item_ids:
        raise ValueError("Existing robustness cell has a different item set")
    for row in rows:
        observed = (
            row["prompt_version"],
            float(row["temperature"]),
            float(row["alpha_norm"]),
            int(row["seed"]),
        )
        expected = (prompt_version, temperature, alpha_norm, seed)
        if observed != expected:
            raise ValueError(f"Existing robustness cell mismatch: {observed}")


def _selected_design(
    protocol: RobustnessProtocolSpec,
    alpha_override: list[float] | None,
) -> tuple[float, ...]:
    if alpha_override is None:
        return protocol.normalized_alpha_grid
    values = tuple(sorted(set(float(value) for value in alpha_override)))
    if not values or any(value not in protocol.normalized_alpha_grid for value in values):
        raise ValueError("alpha-values must be a subset of the registered grid")
    return values


def run_protocol(protocol_key: str, args: argparse.Namespace) -> None:
    if protocol_key not in PROTOCOLS:
        raise ValueError(f"Unknown robustness protocol: {protocol_key}")
    config = load_generation_robustness_config()
    protocol = config.protocols[protocol_key]
    if args.model not in config.model_keys:
        raise ValueError(
            f"{protocol.display_name} is registered for: {', '.join(config.model_keys)}"
        )
    task_keys = list(config.tasks) if args.tasks is None else list(args.tasks)
    unknown = set(task_keys) - set(config.tasks)
    if unknown:
        raise ValueError(f"Unknown robustness tasks: {sorted(unknown)}")
    if args.max_items is not None and args.max_items <= 0:
        raise ValueError("max-items must be positive")
    alpha_grid = _selected_design(protocol, args.alpha_values)
    generation = config.generation
    batch_size = args.batch_size or generation.batch_size
    max_new_tokens = args.max_new_tokens or generation.max_new_tokens

    prompt_sets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for task_key in task_keys:
        task = config.tasks[task_key]
        items = build_robustness_items(task)
        if args.max_items is not None:
            items = items[: args.max_items]
        for version in protocol.prompt_versions:
            prompt_sets[(task_key, version)] = build_robustness_prompts(
                task,
                items,
                version,
                model_key=args.model,
            )
    expected_rows = sum(
        len(prompt_sets[(task, version)])
        * len(alpha_grid)
        * sum(len(condition.seeds) for condition in protocol.temperatures)
        for task in task_keys
        for version in protocol.prompt_versions
    )
    design = {
        "schema_version": 1,
        "protocol": protocol_key,
        "model_key": args.model,
        "tasks": task_keys,
        "prompt_versions": list(protocol.prompt_versions),
        "normalized_alpha_grid": list(alpha_grid),
        "temperatures": [
            {
                "value": condition.value,
                "sampled": condition.sampled,
                "seeds": list(condition.seeds),
            }
            for condition in protocol.temperatures
        ],
        "max_items_per_task": args.max_items,
        "batch_size": batch_size,
        "max_input_tokens": generation.max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "top_p": generation.top_p,
        "top_k": generation.top_k,
    }
    design["identity_sha256"] = _identity_hash(design)
    if args.dry_run:
        print({**design, "n_generation_rows": expected_rows})
        return

    bundle = load_model_bundle(
        args.model,
        device_map=args.device_map,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    output_dir = args.output_dir or (
        REPOSITORY_ROOT / "results" / "generated" / protocol_key / args.model
    )
    cells_dir = output_dir / "cells"
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("identity_sha256") != design["identity_sha256"]:
            raise ValueError("Existing output directory has a different run identity")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(metadata_path, {**design, "status": "running"})

    all_rows = []
    for task_key in task_keys:
        for version in protocol.prompt_versions:
            prompt_rows = build_robustness_prompts(
                config.tasks[task_key],
                build_robustness_items(config.tasks[task_key])[
                    : args.max_items if args.max_items is not None else None
                ],
                version,
                model_key=bundle.spec.key,
            )
            item_ids = {str(row["item_id"]) for row in prompt_rows}
            for condition in protocol.temperatures:
                for alpha_norm in alpha_grid:
                    for seed in condition.seeds:
                        path = _cell_path(
                            cells_dir,
                            task_key,
                            version,
                            condition.value,
                            alpha_norm,
                            seed,
                        )
                        if path.exists():
                            rows = read_jsonl(path)
                        else:
                            rows = generate_cell(
                                bundle,
                                prompt_rows,
                                alpha_norm=alpha_norm,
                                temperature=condition,
                                seed=seed,
                                batch_size=batch_size,
                                max_input_tokens=generation.max_input_tokens,
                                max_new_tokens=max_new_tokens,
                                top_p=generation.top_p,
                                top_k=generation.top_k,
                            )
                            write_jsonl_atomic(path, rows)
                        _validate_cell(
                            rows,
                            item_ids,
                            prompt_version=version,
                            temperature=condition.value,
                            alpha_norm=alpha_norm,
                            seed=seed,
                        )
                        all_rows.extend(rows)
    if len(all_rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, assembled {len(all_rows)}")
    keys = {
        (
            row["task"],
            row["item_id"],
            row["prompt_version"],
            float(row["temperature"]),
            float(row["alpha_norm"]),
            int(row["seed"]),
        )
        for row in all_rows
    }
    if len(keys) != len(all_rows):
        raise RuntimeError("Duplicate rows in assembled robustness output")
    write_jsonl_atomic(output_dir / "generations.jsonl", all_rows)
    write_json_atomic(
        metadata_path,
        {
            **design,
            "status": "complete",
            "n_generation_rows": len(all_rows),
            "target_layer": bundle.spec.target_layer,
            "raw_alpha_range": list(bundle.spec.raw_alpha_range),
            "answer_parser": "strict_json_and_unambiguous_answer_field",
        },
    )
    print(f"Wrote {len(all_rows)} rows to {output_dir}")


def parse_args(protocol_key: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Run the {protocol_key.replace('_', ' ')} analysis."
    )
    parser.add_argument("--model", default="qwen25_14b")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--alpha-values", nargs="+", type=float)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()
