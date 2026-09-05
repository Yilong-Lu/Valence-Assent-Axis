"""Build the quantitative panels of Figure 4."""

# %% [markdown]
# # Figure 4: objective reasoning under VAA intervention
#
# Panels a/b and d/e are generated here. The two response examples are arranged
# with these panels in the complete composite `figures/main/figure4.pdf`.

# %%
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.figures.common import FIGURE_DIR, SOURCE_DATA_DIR, configure_style, export_figure, panel_label


CATEGORY_ORDER = (
    "Sound Reasoning",
    "Contradictory Reasoning",
    "Coherent Hallucination",
    "Incoherent Hallucination",
    "Ambiguous/Mixed",
)
CATEGORY_COLORS = {
    "Sound Reasoning": "#69BFA8",
    "Contradictory Reasoning": "#5A9CC3",
    "Coherent Hallucination": "#E49B62",
    "Incoherent Hallucination": "#F44336",
    "Ambiguous/Mixed": "#BDBDBD",
}


def _composition(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame.groupby(["alignment_pressure", "response_type"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=CATEGORY_ORDER, fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0).mul(100)


def build_figure(think: pd.DataFrame, answer: pd.DataFrame, factual: pd.DataFrame) -> plt.Figure:
    configure_style()
    figure, axes = plt.subplot_mosaic(
        [["a", "b"], ["d", "e"]],
        figsize=(9, 6.5),
        gridspec_kw={"width_ratios": [0.9, 1.35]},
    )
    axis = axes["a"]
    sns.lineplot(data=think, x="alignment_pressure", y="correct", color="#003366", linewidth=2, label="Think-then-answer", ax=axis)
    sns.lineplot(data=answer, x="alignment_pressure", y="correct", color="#003366", linewidth=2, linestyle="--", label="Answer-then-think", ax=axis)
    axis.legend(fontsize=9)
    axis = axes["d"]
    sns.lineplot(data=factual, x="alignment_pressure", y="correct", color="#003366", linewidth=2, ax=axis)
    for label in ("a", "d"):
        axes[label].set(xlabel=r"Alignment Pressure ($\alpha_{\mathrm{aligned}}$)", ylabel="Correctness", xlim=(-1, 1), ylim=(-0.01, 1.01))

    for label, frame in (("b", think), ("e", factual)):
        axis = axes[label]
        composition = _composition(frame)
        composition.plot(kind="bar", stacked=True, width=0.8, color=[CATEGORY_COLORS[c] for c in composition.columns], edgecolor="black", linewidth=0.5, alpha=0.6, rot=0, ax=axis)
        axis.set(xlabel=r"Alignment Pressure ($\alpha_{\mathrm{aligned}}$)", ylabel="Percentage (%)", ylim=(0, 100))
        if label == "b":
            axis.legend(title="Reasoning Pattern", loc="lower right", fontsize=8, title_fontsize=8)
        else:
            axis.get_legend().remove()
    for label, axis in axes.items():
        panel_label(axis, label)
    figure.tight_layout(pad=1.2)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR / "generated")
    args = parser.parse_known_args()[0]
    think = pd.read_csv(SOURCE_DATA_DIR / "figure4_alphabetical_think_then_answer.csv")
    answer = pd.read_csv(SOURCE_DATA_DIR / "figure4_alphabetical_answer_then_think.csv")
    factual = pd.read_csv(SOURCE_DATA_DIR / "figure4_factual_judgment.csv")
    figure = build_figure(think, answer, factual)
    export_figure(figure, "figure4_quantitative_panels", output_dir=args.output_dir)
    plt.close(figure)


if __name__ == "__main__":
    main()
