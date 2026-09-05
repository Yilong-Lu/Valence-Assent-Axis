"""Build the quantitative panels used in revised Figure 5."""

# %% [markdown]
# # Figure 5: stance-taking and preference-induced sycophancy
#
# This notebook builds panels b/c and e/f/g. The qualitative response example
# and compact task schematic are arranged with these vector panels in draw.io;
# the complete composite is tracked as `figures/main/figure5.pdf`.

# %%
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.figures.common import (
    FIGURE_DIR,
    PROCESSED_DATA_DIR,
    SOURCE_DATA_DIR,
    configure_style,
    export_figure,
)


CONDITION_LABELS = {
    "baseline": "No Preference",
    "user_like": "User Likes",
    "user_dislike": "User Dislikes",
}
REASONING_ORDER = (
    "Sound Reasoning",
    "Ambiguous Logic",
    "Contradictory Reasoning",
    "Cherry-picking",
    "Coherent Hallucination",
    "Incoherent Hallucination",
    "Mixed",
)
REASONING_COLORS = {
    "Sound Reasoning": "#69BFA8",
    "Ambiguous Logic": "#1F9EB7",
    "Contradictory Reasoning": "#5A9CC3",
    "Cherry-picking": "#E5B45F",
    "Coherent Hallucination": "#E49B62",
    "Incoherent Hallucination": "#F44336",
    "Mixed": "#BDBDBD",
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stance = pd.read_csv(SOURCE_DATA_DIR / "figure5_stance_qwen25_14b.csv")
    feedback = pd.read_csv(PROCESSED_DATA_DIR / "feedback_induced_sycophancy" / "feedback_effect.csv")
    intervention = pd.read_csv(PROCESSED_DATA_DIR / "feedback_induced_sycophancy" / "intervention.csv")
    feedback = feedback[feedback["model_name"].eq("qwen25_14b")].copy()
    intervention = intervention[intervention["model_name"].eq("qwen25_14b")].copy()
    return stance, feedback, intervention


def build_stance_panels(stance: pd.DataFrame) -> plt.Figure:
    configure_style()
    figure, axes = plt.subplot_mosaic([["b", "c"]], figsize=(11, 3.25), gridspec_kw={"width_ratios": [1, 1.6]})

    axis = axes["b"]
    sns.lineplot(data=stance, x="alignment_pressure", y="answer_stance", color="#003366", linewidth=2, label="Answer Stance", ax=axis)
    sns.lineplot(data=stance, x="alignment_pressure", y="reasoning_stance", color="#BF6953", linewidth=2, linestyle="--", label="Reasoning Stance", ax=axis)
    axis.set(xlabel=r"Intervention Coefficient ($\alpha$)", ylabel="Stance on Statements", xlim=(-1, 1))
    axis.set_yticks(range(1, 8), ["Strongly\nDisagree", "Disagree", "Slightly\nDisagree", "Neutral", "Slightly\nAgree", "Agree", "Strongly\nAgree"], fontsize=8)
    axis.legend(fontsize=8, loc="upper left")

    composition = pd.crosstab(stance["alignment_pressure"], stance["response_type_more"], normalize="index").mul(100)
    composition = composition.reindex(columns=REASONING_ORDER, fill_value=0).sort_index()
    axis = axes["c"]
    composition.plot(kind="bar", stacked=True, width=0.8, color=[REASONING_COLORS[c] for c in composition.columns], edgecolor="black", linewidth=0.5, alpha=0.6, rot=0, ax=axis)
    axis.set(xlabel=r"Intervention Coefficient ($\alpha$)", ylabel="Percentage (%)", ylim=(0, 100))
    axis.legend(title="Reasoning Pattern", loc=(1, 0.30), fontsize=9, title_fontsize=9, alignment="left")
    figure.tight_layout(pad=1.2)
    return figure


def build_feedback_panels(feedback: pd.DataFrame, intervention: pd.DataFrame) -> plt.Figure:
    palette = configure_style()
    order = ["User Likes", "User Dislikes"]
    colors = {"User Likes": palette[2], "User Dislikes": "#F44336"}
    feedback["condition_label"] = feedback["condition"].map(CONDITION_LABELS)
    intervention["condition_label"] = intervention["condition"].map(CONDITION_LABELS)

    state = feedback[["item_id", "condition_label", "pre_addition_vaa_projection_unit_z_baseline"]].rename(columns={"pre_addition_vaa_projection_unit_z_baseline": "state"})
    baseline_state = state[state["condition_label"].eq("No Preference")][["item_id", "state"]].rename(columns={"state": "baseline"})
    state_shift = state[state["condition_label"].isin(order)].merge(baseline_state, on="item_id", validate="many_to_one")
    state_shift["shift"] = state_shift["state"] - state_shift["baseline"]

    verdict = feedback[["item_id", "condition_label", "verdict_strong"]].dropna()
    baseline_verdict = verdict[verdict["condition_label"].eq("No Preference")][["item_id", "verdict_strong"]].rename(columns={"verdict_strong": "baseline"})
    verdict_shift = verdict[verdict["condition_label"].isin(order)].merge(baseline_verdict, on="item_id", validate="many_to_one")
    verdict_shift["shift"] = 100 * (verdict_shift["verdict_strong"] - verdict_shift["baseline"])

    figure, axes = plt.subplot_mosaic([["e", "f", "g"]], figsize=(11, 3.25))
    axis = axes["e"]
    sns.violinplot(data=state_shift, x="condition_label", y="shift", order=order, hue="condition_label", palette=colors, legend=False, inner=None, cut=0, linewidth=1, common_norm=True, ax=axis)
    for body, color in zip(axis.collections, colors.values()):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.20)
    sns.lineplot(data=state_shift, x="condition_label", y="shift", units="item_id", estimator=None, color="0.25", alpha=0.06, linewidth=0.5, legend=False, ax=axis)
    sns.pointplot(data=state_shift, x="condition_label", y="shift", order=order, hue="condition_label", palette=colors, legend=False, errorbar=("ci", 95), markers="o", markersize=4, ax=axis)
    axis.axhline(0, color="0.4", linestyle="--", linewidth=1)
    axis.set(xlabel="", ylabel="Pre-Response VAA Projection Shift")

    axis = axes["f"]
    means = verdict_shift.groupby("condition_label")["shift"].mean().reindex(order)
    axis.bar(range(2), means, width=0.42, color=[colors[c] for c in order], alpha=0.18, edgecolor=[colors[c] for c in order], linewidth=1)
    sns.pointplot(data=verdict_shift, x="condition_label", y="shift", order=order, hue="condition_label", palette=colors, errorbar=("ci", 95), markers="o", markersize=5, legend=False, ax=axis)
    axis.axhline(0, color="0.4", linestyle="--", linewidth=1)
    axis.set(xlabel="", ylabel="Strong-Verdict Rate Shift")

    axis = axes["g"]
    intervention["verdict_strong_percent"] = 100 * intervention["verdict_strong"]
    sns.lineplot(data=intervention, x="alpha_norm", y="verdict_strong_percent", hue="condition_label", style="condition_label", hue_order=list(CONDITION_LABELS.values()), style_order=list(CONDITION_LABELS.values()), palette=["gray", colors["User Likes"], colors["User Dislikes"]], linewidth=2, errorbar=("ci", 95), ax=axis)
    axis.set(xlabel=r"Intervention Coefficient $\alpha$", ylabel="Strong-Verdict Rate (%)")
    axis.legend(title="", fontsize=9, loc="upper left")
    figure.tight_layout(pad=1.2)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR / "generated")
    args = parser.parse_known_args()[0]
    stance, feedback, intervention = load_data()
    stance_figure = build_stance_panels(stance)
    feedback_figure = build_feedback_panels(feedback, intervention)
    export_figure(stance_figure, "figure5_stance_panels", output_dir=args.output_dir)
    export_figure(feedback_figure, "figure5_feedback_panels", output_dir=args.output_dir)
    plt.close(stance_figure)
    plt.close(feedback_figure)


if __name__ == "__main__":
    main()
