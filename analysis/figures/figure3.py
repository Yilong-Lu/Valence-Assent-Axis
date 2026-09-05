"""Build revised Figure 3 from its registered Source Data table."""

# %% [markdown]
# # Figure 3: VAA definition and specificity
#
# The figure combines the Subjective Preference controls, the original
# Valence and Objective Truth axis comparisons, and the arithmetic framing
# control for the primary Qwen2.5-14B model.

# %%
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from analysis.figures.common import (
    FIGURE_DIR,
    SOURCE_DATA_DIR,
    configure_style,
    export_figure,
    panel_label,
)


PAIR_ORDER = (
    "neutral_nonopposite",
    "neutral_opposite",
    "valenced_nonopposite",
    "valenced_opposite",
)
PAIR_LABELS = (
    "Neutral\nnon-opposed",
    "Neutral\nopposed /\ncontrast",
    "Valenced\nnon-opposed",
    "Valenced\nopposed",
)


def build_figure(source: pd.DataFrame) -> plt.Figure:
    palette = configure_style()
    figure, axes = plt.subplot_mosaic(
        [["a", "a", "a", "b", "b", "b"], ["c", "c", "d", "d", "e", "e"]],
        figsize=(10, 6.5),
        dpi=150,
    )

    trajectories = source[source["panel"].eq("a")].copy()
    trajectory_style = {
        "valenced_opposite": (palette[2], "-"),
        "valenced_nonopposite": (palette[2], (0, (2, 2))),
        "neutral_opposite": ("0.82", "-"),
        "neutral_nonopposite": ("0.82", (0, (2, 2))),
    }
    axis = axes["a"]
    for series, (color, linestyle) in trajectory_style.items():
        rows = trajectories[trajectories["series"].eq(series)].sort_values("x")
        axis.fill_between(rows["x"], rows["ci_low"], rows["ci_high"], color=color, alpha=0.16)
        axis.plot(rows["x"], rows["y"], color=color, linestyle=linestyle, marker="o", markeredgecolor="black", linewidth=2)
    axis.axhline(0, color="0.55", linestyle="--", linewidth=1, zorder=0)
    axis.set(xlabel=r"Intervention Coefficient $\alpha$", ylabel=r"$\Delta$ Logit (Semantic Pair)", xlim=(-1.05, 1.05), ylim=(-25, 10))
    axis.set_xticks(np.linspace(-1, 1, 6))
    axis.legend(
        handles=[
            Line2D([0], [0], color=palette[2], marker="o", markeredgecolor="black", linewidth=2, label="Valenced"),
            Line2D([0], [0], color="0.82", marker="o", markeredgecolor="black", linewidth=2, label="Neutral"),
            Line2D([0], [0], color="0.25", linestyle=(0, (2, 2)), linewidth=2, label="Non-opposed"),
            Line2D([0], [0], color="0.25", linewidth=2, label="Opposed/contrast"),
        ],
        fontsize=9,
    )

    observed = source[source["panel"].eq("b_observed")].copy()
    null = source[source["panel"].eq("b_permutation_null")].copy()
    axis = axes["b"]
    offsets = {"semantic": -0.12, "position": 0.12}
    markers = {"semantic": "o", "position": "s"}
    faces = {"semantic": palette[2], "position": "white"}
    for component in ("semantic", "position"):
        rows = observed[observed["series"].eq(component)].set_index("category").loc[list(PAIR_ORDER)]
        x = np.arange(len(PAIR_ORDER), dtype=float) + offsets[component]
        axis.errorbar(
            x,
            rows["y"],
            yerr=np.vstack([rows["y"] - rows["ci_low"], rows["ci_high"] - rows["y"]]),
            fmt=markers[component],
            color="black",
            markerfacecolor=faces[component],
            markeredgecolor="black",
            linewidth=1,
            label=component.capitalize(),
            zorder=3,
        )
        null_rows = null[null["series"].eq(component)].set_index("category").loc[list(PAIR_ORDER)]
        for x_value, (_, row) in zip(x, null_rows.iterrows()):
            axis.add_patch(
                Rectangle(
                    (x_value - 0.10, row["ci_low"]),
                    0.20,
                    row["ci_high"] - row["ci_low"],
                    facecolor="0.7",
                    edgecolor="none",
                    alpha=0.25,
                    zorder=0,
                )
            )
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles + [Patch(facecolor="0.7", alpha=0.25)], labels + ["95% Permutation Null"], fontsize=9)
    axis.set_xticks(range(4), PAIR_LABELS, fontsize=9)
    axis.set(xlabel="", ylabel=r"Mean Absolute Slope, $|\beta_{\alpha}|$", ylim=(0, 12))

    scatter_specs = {
        "c": ("Projection on Valence Axis", "Valence"),
        "d": ("Projection on Objective Truth Axis", "Response"),
    }
    for panel, (xlabel, legend_title) in scatter_specs.items():
        rows = source[source["panel"].eq(panel)].copy()
        axis = axes[panel]
        sns.scatterplot(data=rows, x="x", y="y", hue="series", palette=[palette[2], palette[3]], alpha=0.65, edgecolor="black", ax=axis)
        correlation = rows["x"].corr(rows["y"])
        axis.text(0.05, 0.95, f"$r = {correlation:.2f}$", transform=axis.transAxes, va="top", bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.5})
        axis.set(xlabel=xlabel, ylabel="Projection on Judgment Axis")
        axis.legend(title=legend_title if panel == "c" else "", fontsize=9, title_fontsize=9)

    arithmetic = source[source["panel"].eq("e")].copy()
    labels = {
        "direct_numeric": "Direct Numeric Answer",
        "verification_true": "True Verification",
        "verification_false": "False Verification",
    }
    colors = {"direct_numeric": "black", "verification_true": palette[3], "verification_false": "seagreen"}
    axis = axes["e"]
    for series in labels:
        rows = arithmetic[arithmetic["series"].eq(series)].sort_values("x")
        axis.fill_between(rows["x"], rows["ci_low"], rows["ci_high"], color=colors[series], alpha=0.12)
        axis.plot(rows["x"], rows["y"], marker="o", markeredgecolor="black", linewidth=2, color=colors[series], label=labels[series])
    axis.set(xlabel=r"Intervention Coefficient $\alpha$", ylabel="Candidate Accuracy", xlim=(-1.12, 1.12), ylim=(-0.04, 1.04))
    axis.set_xticks(np.linspace(-1, 1, 5))
    axis.set_yticks([0, 0.5, 1.0])
    axis.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 0.58))

    for label, axis in axes.items():
        panel_label(axis, label, x=-0.12 if label in {"a", "b"} else -0.22, y=1.12 if label in {"a", "b"} else 1.07)
    figure.tight_layout(pad=1.3)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=str, default=str(FIGURE_DIR / "generated"))
    args = parser.parse_known_args()[0]
    source = pd.read_csv(SOURCE_DATA_DIR / "figure3.csv")
    figure = build_figure(source)
    export_figure(figure, "figure3", output_dir=Path(args.output_dir))
    plt.close(figure)


if __name__ == "__main__":
    main()
