"""Shared paths, labels, styling, and export helpers for manuscript figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATA_DIR = REPOSITORY_ROOT / "data" / "source_data"
PROCESSED_DATA_DIR = REPOSITORY_ROOT / "data" / "processed"
SUMMARY_DIR = REPOSITORY_ROOT / "results" / "summaries"
FIGURE_DIR = REPOSITORY_ROOT / "figures"

MODEL_ORDER = (
    "qwen25_3b",
    "mistral_7b",
    "qwen25_7b",
    "llama3_8b",
    "gemma2_9b",
    "qwen25_14b",
    "qwen25_32b",
    "qwen25_72b",
)
MODEL_LABELS = {
    "qwen25_3b": "Qwen2.5 3B",
    "mistral_7b": "Mistral-7B v0.3",
    "qwen25_7b": "Qwen2.5 7B",
    "llama3_8b": "Llama-3.1 8B",
    "gemma2_9b": "Gemma-2 9B",
    "qwen25_14b": "Qwen2.5 14B",
    "qwen25_32b": "Qwen2.5 32B",
    "qwen25_72b": "Qwen2.5 72B",
}


def configure_style(font_scale: float = 1.2) -> list[tuple[float, float, float]]:
    """Apply the Seaborn style used by the manuscript figures."""

    sns.set_theme(
        style="ticks",
        context="paper",
        font_scale=font_scale,
        rc={"axes.spines.right": False, "axes.spines.top": False},
    )
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 1.2,
            "axes.edgecolor": "black",
            "legend.frameon": False,
        }
    )
    return sns.color_palette("colorblind")


def panel_label(
    axis: plt.Axes,
    label: str,
    *,
    x: float = -0.13,
    y: float = 1.10,
) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
    )


def export_figure(
    figure: plt.Figure,
    stem: str,
    *,
    output_dir: Path,
    dpi: int = 400,
) -> dict[str, Path]:
    """Write editable vector formats and a review PNG."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pdf": output_dir / f"{stem}.pdf",
        "svg": output_dir / f"{stem}.svg",
        "png": output_dir / f"{stem}.png",
    }
    figure.savefig(paths["pdf"], bbox_inches="tight")
    figure.savefig(paths["svg"], bbox_inches="tight")
    figure.savefig(paths["png"], bbox_inches="tight", dpi=dpi)
    return paths
