"""Build the frozen sycophancy stimuli from a user-supplied upstream dataset.

The upstream ``sycophancy-eval`` repository does not currently declare a data
redistribution license. This script therefore selects the exact 296 argument
texts locally from the pinned upstream file instead of bundling those texts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vaa.config import REPOSITORY_ROOT
from vaa.sycophancy_config import load_sycophancy_selection


DEFAULT_SELECTION = (
    REPOSITORY_ROOT
    / "data"
    / "stimuli"
    / "feedback_induced_sycophancy"
    / "argument_selection.json"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "data"
    / "raw_external"
    / "feedback_induced_sycophancy"
    / "arguments.json"
)


def load_unique_arguments(path: Path) -> list[dict[str, Any]]:
    """Return first-occurrence argument records in upstream order."""

    unique = []
    observed = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            base = row.get("base", {})
            text = base.get("text")
            if base.get("dataset") != "arguments" or not isinstance(text, str):
                continue
            if text not in observed:
                observed.add(text)
                unique.append(base)
    if not unique:
        raise ValueError(f"No argument records were found in {path}")
    return unique


def build_stimuli(
    unique_arguments: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select and format the frozen argument panel."""

    output = []
    for row in selection_rows:
        index = int(row["upstream_unique_index"])
        try:
            source = unique_arguments[index]
        except IndexError as exc:
            raise ValueError(f"Upstream argument index is absent: {index}") from exc
        text = str(source["text"])
        output.append(
            {
                "item_id": row["item_id"],
                "argument": text,
                "logical_error": source.get("logical_error", ""),
                "rating": source.get("rating"),
                "in_intervention_subset": bool(row["in_intervention_subset"]),
                "intervention_split": row["intervention_split"],
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-file", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    selection = load_sycophancy_selection(args.selection_manifest)
    unique_arguments = load_unique_arguments(args.upstream_file)
    stimuli = build_stimuli(unique_arguments, selection["items"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(stimuli, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(stimuli)} arguments to {args.output}")


if __name__ == "__main__":
    main()
