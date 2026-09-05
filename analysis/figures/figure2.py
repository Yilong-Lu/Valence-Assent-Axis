"""Build Figure 2, the Qwen2.5-14B VAA discovery and intervention figure."""

# %% [markdown]
# # Figure 2: discovering and intervening on the Judgment Axis
#
# The selected layer is highlighted in the layer profile. The remaining panels
# show the selected-layer PCA geometry and the intervention responses in the
# Value Judgment and Sentiment Analysis tasks.

# %%
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.figures.common import FIGURE_DIR, SOURCE_DATA_DIR, configure_style, export_figure, panel_label


TARGET_LAYER = 28


def load_data() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    layer_rows = json.loads((SOURCE_DATA_DIR / "figure2_layer_profiles_qwen25_14b.json").read_text())
    layer = pd.DataFrame(
        {
            "layer": [row["layer"] for row in layer_rows],
            "alignment": [abs(row["binary_continuous_alignment"]) for row in layer_rows],
            "binary_corr": [abs(row["binary_pc1_label"]) for row in layer_rows],
            "continuous_corr": [abs(row["continuous_pc1_label"]) for row in layer_rows],
        }
    )
    pca = json.loads((SOURCE_DATA_DIR / "figure2_pca_qwen25_14b.json").read_text())
    intervention = pd.read_csv(SOURCE_DATA_DIR / "figure2_intervention_qwen25_14b.csv")
    return layer, pca, intervention


def build_figure(layer: pd.DataFrame, pca: dict, intervention: pd.DataFrame) -> plt.Figure:
    palette = configure_style()
    figure, axes = plt.subplot_mosaic(
        [["a", "a", "d"], ["b", "c", "e"]],
        figsize=(10, 6.5),
        dpi=150,
    )

    axis = axes["a"]
    sns.lineplot(data=layer, x="layer", y="alignment", marker="o", markeredgecolor="black", markersize=7, linewidth=2.5, color="#333333", label="Representational Similarity", ax=axis)
    sns.lineplot(data=layer, x="layer", y="binary_corr", color="#6699CC", linestyle="--", linewidth=2, label="Decision Correlation (Binary)", ax=axis)
    sns.lineplot(data=layer, x="layer", y="continuous_corr", color="#003366", linewidth=2, label="Decision Correlation (Continuous)", ax=axis)
    selected = layer.loc[layer["layer"].eq(TARGET_LAYER), "alignment"].iloc[0]
    axis.scatter(TARGET_LAYER, selected, color="indianred", s=200, edgecolor="black", marker="*", zorder=5)
    axis.set(xlabel="Model Layer", ylabel="Similarity / Correlation", xlim=(-1, layer["layer"].max() + 1), ylim=(0, 1))
    axis.set_xticks(range(0, int(layer["layer"].max()) + 2, 4))
    axis.legend(fontsize=9)

    axis = axes["b"]
    sns.scatterplot(x=-np.asarray(pca["PC1"]), y=pca["PC2"], hue=pca["y_logit"], palette="vlag", alpha=0.6, edgecolor="black", linewidth=0.5, ax=axis)
    axis.set(xlabel="Projection on PC1 (Layer 28)", ylabel="Projection on PC2")
    axis.legend(title="Logit of Support", fontsize=6, title_fontsize=6)

    axis = axes["c"]
    variance = pca["explained_variance_ratio"][:20]
    sns.barplot(x=list(range(1, 21)), y=variance, color=palette[7], alpha=0.8, ax=axis)
    axis.plot(np.arange(20), variance, marker="o", markeredgecolor="black", linewidth=1.5, color="#333333")
    axis.set(xlabel="Principal Component Index", ylabel="Explained Variance Ratio")
    axis.set_xticks(np.asarray([1, 5, 10, 15, 20]) - 1, [1, 5, 10, 15, 20])
    inset = axis.inset_axes([0.4, 0.4, 0.55, 0.55])
    correlations = np.abs(pca["r"][:20])
    sns.barplot(x=list(range(1, 21)), y=correlations, color=palette[7], alpha=0.8, ax=inset)
    inset.plot(np.arange(20), correlations, marker="o", markeredgecolor="black", markersize=3, linewidth=1, color="#333333")
    inset.set(xlabel="", ylabel="", ylim=(0, 1.01))
    inset.set_title(r"Decision Correlation ($|r|$)", fontsize=7, pad=2)
    inset.set_xticks([0, 9, 19], [1, 10, 20], fontsize=6)
    inset.set_yticks([0, 0.5, 1], [0, 0.5, 1], fontsize=6)

    axis = axes["d"]
    for group, label, linestyle in [("binarySentiment", "Binary", "--"), ("continuousSentiment", "Continuous", "-")]:
        sns.lineplot(data=intervention[intervention["group"].eq(group)], x="alpha_norm", y="expected_probability", color="#BF6953", linestyle=linestyle, linewidth=2.5, label=label, ax=axis)
    axis.set(xlabel="", ylabel="Probability of Positive", ylim=(0, 1))
    axis.tick_params(axis="x", labelbottom=False)

    axis = axes["e"]
    sns.lineplot(data=intervention[intervention["group"].eq("continuous09")], x="alpha_norm", y="expected_probability", color="#003366", linewidth=2.5, ax=axis)
    axis.set(xlabel=r"Intervention Coefficient $\alpha$", ylabel="Continuous Support Score", ylim=(0, 1))

    for label, axis in axes.items():
        panel_label(axis, label, x=-0.10 if label == "a" else -0.13)
    figure.tight_layout(pad=1.2)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR / "generated")
    args = parser.parse_known_args()[0]
    figure = build_figure(*load_data())
    export_figure(figure, "figure2", output_dir=args.output_dir)
    plt.close(figure)


if __name__ == "__main__":
    main()
