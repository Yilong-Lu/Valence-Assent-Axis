"""Build revised Figure 6, the cross-model generality summary."""

# %% [markdown]
# # Figure 6: cross-model generality
#
# Panel a retains the original heatmap implementation. The other
# panels summarize the corresponding model-specific estimates and intervals.

# %%
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from analysis.figures.common import (
    FIGURE_DIR,
    MODEL_LABELS,
    MODEL_ORDER,
    SOURCE_DATA_DIR,
    export_figure,
    panel_label,
)


MODEL_Y = np.arange(len(MODEL_ORDER), dtype=float)
COLORS = {
    "dark_blue": "#003366",
    "blue": sns.color_palette("colorblind")[0],
    "orange": "#BF6953",
}


def configure_figure_style() -> None:
    """Apply the typography used in the manuscript Figure 6 layout."""

    sns.set_theme(
        style="ticks",
        context="paper",
        font_scale=1.2,
        rc={
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 1.2,
            "axes.edgecolor": "black",
        },
    )
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "lines.markersize": 6,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _forest_axis(axis: plt.Axes, *, show_labels: bool) -> None:
    axis.set_ylim(len(MODEL_ORDER) - 0.45, -0.55)
    axis.set_yticks(MODEL_Y, [MODEL_LABELS[m] for m in MODEL_ORDER] if show_labels else [""] * len(MODEL_ORDER))
    axis.tick_params(axis="y", length=3)


def _draw_estimates(axis: plt.Axes, frame: pd.DataFrame, *, color, marker="o", offset=0.0, label=None, face=None) -> None:
    rows = frame.set_index("model_name").loc[list(MODEL_ORDER)]
    estimate = rows["estimate"].to_numpy(float)
    low = rows["ci_low"].to_numpy(float)
    high = rows["ci_high"].to_numpy(float)
    axis.errorbar(estimate, MODEL_Y + offset, xerr=np.vstack([estimate - low, high - estimate]), fmt=marker, color="black", markerfacecolor=color if face is None else face, markeredgecolor="black", markeredgewidth=0.5, linewidth=1.8, capsize=0, label=label, zorder=3)


def build_figure(source: pd.DataFrame) -> plt.Figure:
    configure_figure_style()
    figure = plt.figure(figsize=(12, 7.2), constrained_layout=False)
    grid = GridSpec(2, 20, figure=figure, height_ratios=[1.0, 1.02])
    axes = {
        "a": figure.add_subplot(grid[0, 0:11]),
        "b": figure.add_subplot(grid[0, 14:20]),
        "c": figure.add_subplot(grid[1, 0:5]),
        "d": figure.add_subplot(grid[1, 5:10]),
        "e_state": figure.add_subplot(grid[1, 10:15]),
        "e_verdict": figure.add_subplot(grid[1, 15:20]),
    }

    layer = source[source["panel"].eq("a")].copy()
    depth_grid = np.linspace(0, 1, 80)
    model_order = tuple(reversed(MODEL_ORDER))
    matrix = []
    targets = []
    for model in model_order:
        rows = layer[layer["model_name"].eq(model)].sort_values("relative_layer")
        matrix.append(np.interp(depth_grid, rows["relative_layer"], rows["estimate"]))
        targets.append(rows.loc[rows["is_target_layer"].astype(bool), "relative_layer"].iloc[0])
    axis = axes["a"]
    image = axis.imshow(np.vstack(matrix), aspect="auto", cmap="coolwarm", origin="lower", extent=[0, 1, -0.5, len(MODEL_ORDER) - 0.5], vmin=0.1, vmax=0.9, interpolation="nearest")
    for row, target in enumerate(targets):
        axis.add_patch(
            plt.Rectangle(
                (target * 79 / 80 - 1 / 160, row - 0.5),
                1 / 80,
                1,
                facecolor="indianred",
                edgecolor="black",
                linewidth=0.5,
                alpha=1,
                zorder=3,
            )
        )
    axis.set_yticks(range(len(model_order)), [MODEL_LABELS[m] for m in model_order])
    axis.set_xlabel("Relative Layer Depth", fontsize=10)
    axis.tick_params(axis="both", which="major", labelsize=9)
    axis.spines["top"].set_visible(True)
    axis.spines["right"].set_visible(True)
    axis.legend(
        handles=[
            Patch(
                facecolor="indianred",
                edgecolor="black",
                linewidth=1,
                label="Chosen Intervention Layer",
                alpha=0.8,
            )
        ],
        loc="upper left",
        bbox_to_anchor=(-0.03, 1.10),
        frameon=False,
        fontsize=9,
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.02, pad=0.03)
    colorbar.ax.tick_params(labelsize=9)
    colorbar.set_label(
        "Representation Similarity", rotation=270, labelpad=10, fontsize=9
    )

    axis = axes["b"]
    cross_domain = source[source["panel"].eq("b")]
    for task, color, offset in [
        ("Value Judgment Task", COLORS["dark_blue"], -0.11),
        ("Sentiment Analysis Task", COLORS["orange"], 0.11),
    ]:
        _draw_estimates(axis, cross_domain[cross_domain["task"].eq(task)], color=color, offset=offset, label=task.removesuffix(" Task"))
    _forest_axis(axis, show_labels=True)
    axis.set(xlabel=r"Causal Control Effect Size ($b$)", xlim=(0, 1.02), title="Value Judgment and Sentiment Analysis")
    axis.legend(loc="lower left", fontsize=8.5)

    axis = axes["c"]
    preference = source[source["panel"].eq("c")]
    for label, marker, offset, face in [
        ("Non-opposed", "s", -0.11, "white"),
        ("Opposed/contrast", "o", 0.11, COLORS["blue"]),
    ]:
        _draw_estimates(
            axis,
            preference[preference["stratum_label"].eq(label)],
            color=COLORS["blue"],
            marker=marker,
            offset=offset,
            label=label,
            face=face,
        )
    _forest_axis(axis, show_labels=True)
    axis.set(xlabel=r"Valence-Specific Steering Effect ($b$)", xlim=(0, 13), title="Subjective Preference")
    legend = axis.get_legend()
    if legend is not None:
        legend.remove()
    axis.plot(
        0.97,
        0.94,
        marker="s",
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=0.8,
        transform=axis.transAxes,
        clip_on=False,
    )
    axis.text(
        0.93,
        0.94,
        "Non-opposed",
        transform=axis.transAxes,
        ha="right",
        va="center",
        fontsize=7.5,
    )
    axis.plot(
        0.97,
        0.865,
        marker="o",
        color=COLORS["blue"],
        markeredgecolor="black",
        markeredgewidth=0.8,
        transform=axis.transAxes,
        clip_on=False,
    )
    axis.text(
        0.93,
        0.865,
        "Opposed/contrast",
        transform=axis.transAxes,
        ha="right",
        va="center",
        fontsize=7.5,
    )

    axis = axes["d"]
    _draw_estimates(axis, source[source["panel"].eq("d")], color=COLORS["blue"])
    axis.axvline(0, color="0.55", linestyle="--", linewidth=1, zorder=0)
    _forest_axis(axis, show_labels=False)
    axis.set(xlabel=r"Coherent-Hallucination Effect ($b$)", xlim=(-3.1, 0.12), title="Factual Judgment")

    feedback = source[source["panel"].eq("e")]
    state = feedback[feedback["endpoint"].str.contains("Projection", na=False)]
    verdict = feedback[feedback["endpoint"].str.contains("Verdict", na=False)]
    axis = axes["e_state"]
    _draw_estimates(axis, state, color=COLORS["blue"])
    _forest_axis(axis, show_labels=False)
    axis.set(xlabel="Pre-Response VAA Projection Difference", xlim=(0, 2.85))
    axis = axes["e_verdict"]
    _draw_estimates(axis, verdict, color=COLORS["blue"], marker="s")
    _forest_axis(axis, show_labels=False)
    axis.set(xlabel="Strong-Verdict Rate Difference", xlim=(0, 98))

    panel_label(axes["a"], "a", x=-0.08)
    panel_label(axes["b"], "b", x=-0.10)
    panel_label(axes["c"], "c", x=-0.19)
    panel_label(axes["d"], "d", x=0)
    panel_label(axes["e_state"], "e", x=0)
    figure.text(
        0.785,
        0.465,
        "Preference-Induced Sycophancy (user-like − user-dislike)",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.10, top=0.94, hspace=0.40, wspace=1.40)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR / "generated")
    args = parser.parse_known_args()[0]
    figure = build_figure(pd.read_csv(SOURCE_DATA_DIR / "figure6.csv"))
    export_figure(figure, "figure6", output_dir=args.output_dir)
    plt.close(figure)


if __name__ == "__main__":
    main()
