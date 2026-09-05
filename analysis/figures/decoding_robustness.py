"""Build the Supplementary decoding-temperature robustness figure."""

# %% [markdown]
# # Decoding-temperature robustness
#
# Answer and Sound-Reasoning slopes are shown for greedy decoding and sampling
# at temperatures 0.2 and 1.0. Temperature 0.2 is the main manuscript setting.

# %%
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.figures.common import FIGURE_DIR, MODEL_LABELS, MODEL_ORDER, SUMMARY_DIR, configure_style, export_figure, panel_label


TASK_LABELS = {"TruthfulQA": "Factual Judgment", "alphabetical_think_answer": "Alphabetical Order"}
TEMPERATURES = (0.0, 0.2, 1.0)


def _plot(axis: plt.Axes, frame: pd.DataFrame, *, models: list[str], task: str, value: str, title: str, styles: dict) -> None:
    rows = frame[frame["task"].eq(task)]
    if "outcome" in rows:
        rows = rows[rows["outcome"].eq("sound_reasoning")]
    positions = {model: len(models) - 1 - index for index, model in enumerate(models)}
    for temperature, offset in zip(TEMPERATURES, (0.16, 0.0, -0.16)):
        group = rows[rows["temperature"].eq(temperature)].set_index("model_name")
        estimate = np.array([group.loc[m, value] for m in models])
        low = np.array([group.loc[m, "bootstrap_ci_low"] for m in models])
        high = np.array([group.loc[m, "bootstrap_ci_high"] for m in models])
        y = np.array([positions[m] for m in models]) + offset
        style = styles[temperature]
        axis.errorbar(estimate, y, xerr=np.vstack([estimate - low, high - estimate]), fmt=style["marker"], color=style["color"], markeredgecolor="white", markeredgewidth=0.4, capsize=1.5, label=style["label"])
    axis.axvline(0, color="0.55", linestyle="--", linewidth=0.8, zorder=0)
    axis.set_ylim(-0.6, len(models) - 0.4)
    axis.set_yticks([positions[m] for m in models], [MODEL_LABELS[m] for m in models])
    axis.set_title(title, pad=5)


def build_figure(answer: pd.DataFrame, reasoning: pd.DataFrame) -> plt.Figure:
    palette = configure_style(font_scale=1.0)
    styles = {
        0.0: {"color": "0.50", "marker": "o", "label": "T=0"},
        0.2: {"color": "#173F5F", "marker": "s", "label": "T=0.2"},
        1.0: {"color": palette[0], "marker": "^", "label": "T=1"},
    }
    models = [model for model in MODEL_ORDER if model in set(answer["model_name"])]
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 5.4), constrained_layout=True)
    for column, task in enumerate(("TruthfulQA", "alphabetical_think_answer")):
        _plot(axes[0, column], answer, models=models, task=task, value="mean_positive_slope", title=f"{TASK_LABELS[task]}: Answer", styles=styles)
        _plot(axes[1, column], reasoning, models=models, task=task, value="mean_alignment_slope", title=f"{TASK_LABELS[task]}: Sound Reasoning", styles=styles)
    for label, axis in zip("abcd", axes.flat):
        panel_label(axis, label)
    axes[0, 1].legend(loc="lower right")
    figure.supxlabel("Slope per unit normalized VAA intervention")
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR / "supplementary")
    args = parser.parse_known_args()[0]
    root = SUMMARY_DIR / "generation_robustness"
    answer = pd.read_csv(root / "decoding_temperature_slope_summary.csv")
    reasoning = pd.read_csv(root / "decoding_temperature_reasoning_summary.csv")
    figure = build_figure(answer, reasoning)
    export_figure(figure, "supplement_decoding_temperature", output_dir=args.output_dir)
    plt.close(figure)


if __name__ == "__main__":
    main()
