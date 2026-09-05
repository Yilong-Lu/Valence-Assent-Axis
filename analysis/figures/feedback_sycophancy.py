"""Build the Supplementary cross-model preference-intervention curves."""

# %% [markdown]
# # Preference-Induced Sycophancy across models
#
# Strictly parsed Strong-verdict probabilities and 95% Wilson intervals are
# shown under no stated preference, user-like preference, and user-dislike preference.

# %%
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.figures.common import FIGURE_DIR, MODEL_LABELS, MODEL_ORDER, PROCESSED_DATA_DIR, configure_style, export_figure


CONDITION_ORDER = ("baseline", "user_like", "user_dislike")
CONDITION_LABELS = {"baseline": "No Preference", "user_like": "User Likes", "user_dislike": "User Dislikes"}
CONDITION_MARKERS = {"baseline": "o", "user_like": "^", "user_dislike": "s"}


def build_figure(frame: pd.DataFrame) -> plt.Figure:
    palette = configure_style(font_scale=1.0)
    colors = {"baseline": "0.45", "user_like": palette[2], "user_dislike": palette[3]}
    figure, axes = plt.subplots(2, 4, figsize=(7.2, 5.0), sharex=True, sharey=True, constrained_layout=True)
    handles = labels = None
    for model, axis in zip(MODEL_ORDER, axes.flat):
        for condition in CONDITION_ORDER:
            rows = frame[(frame["model_name"].eq(model)) & (frame["condition"].eq(condition))].sort_values("alpha_norm")
            axis.fill_between(rows["alpha_norm"], rows["strict_ci_low"], rows["strict_ci_high"], color=colors[condition], alpha=0.10, linewidth=0)
            axis.plot(rows["alpha_norm"], rows["strong_probability_strict"], color=colors[condition], marker=CONDITION_MARKERS[condition], markeredgecolor="white", markeredgewidth=0.3, label=CONDITION_LABELS[condition])
        axis.set_title(MODEL_LABELS[model])
        axis.set(xlim=(-1, 1), ylim=(-0.03, 1.03))
        if handles is None:
            handles, labels = axis.get_legend_handles_labels()
    figure.supxlabel("Normalized VAA steering strength")
    figure.supylabel(r"$P$(Strong verdict)")
    figure.legend(handles, labels, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.035), columnspacing=1.2)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR / "supplementary")
    args = parser.parse_known_args()[0]
    frame = pd.read_csv(PROCESSED_DATA_DIR / "feedback_induced_sycophancy" / "steering_curve_summary.csv")
    expected = len(MODEL_ORDER) * len(CONDITION_ORDER) * 11
    if len(frame) != expected:
        raise ValueError(f"Expected {expected} curve rows, found {len(frame)}")
    figure = build_figure(frame)
    export_figure(figure, "supplement_feedback_sycophancy", output_dir=args.output_dir)
    plt.close(figure)


if __name__ == "__main__":
    main()
