from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pandas as pd

from analysis.figures.common import MODEL_ORDER, SOURCE_DATA_DIR


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_main_figure_source_data_are_complete() -> None:
    figure3 = pd.read_csv(SOURCE_DATA_DIR / "figure3.csv")
    assert set(figure3["panel"]) == {"a", "b_observed", "b_permutation_null", "c", "d", "e"}

    figure6 = pd.read_csv(SOURCE_DATA_DIR / "figure6.csv")
    assert set(figure6["panel"]) == {"a", "b", "c", "d", "e"}
    assert set(figure6["model_name"].dropna()) == set(MODEL_ORDER)


def test_figure2_selected_layer_sources_match() -> None:
    profiles = json.loads((SOURCE_DATA_DIR / "figure2_layer_profiles_qwen25_14b.json").read_text())
    assert {row["layer"] for row in profiles} == set(range(48))
    intervention = pd.read_csv(SOURCE_DATA_DIR / "figure2_intervention_qwen25_14b.csv")
    assert set(intervention["group"]) == {"continuous09", "binarySentiment", "continuousSentiment"}
    assert intervention["alpha_norm"].between(-1, 1).all()
    continuous = intervention[intervention["group"].eq("continuous09")]
    assert continuous["statement_id"].nunique() == 175


def test_figure_notebooks_are_clean_and_portable() -> None:
    notebooks = sorted((REPOSITORY_ROOT / "notebooks" / "main_figures").glob("*.ipynb"))
    notebooks += sorted(
        (REPOSITORY_ROOT / "notebooks" / "supplementary_figures").glob("*.ipynb")
    )
    assert len(notebooks) == 7
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        assert not any(cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code")
        serialized = nbformat.writes(notebook)
        assert "/" + "home/" not in serialized


def test_supplement_source_tables_are_complete() -> None:
    expected_models = set(MODEL_ORDER)
    for name in (
        "supplement_cross_domain_intervention.csv",
        "supplement_alphabetical_order.csv",
        "supplement_factual_judgment.csv",
        "supplement_stance_taking.csv",
    ):
        frame = pd.read_csv(SOURCE_DATA_DIR / name)
        assert set(frame["model_name"]) == expected_models
        assert frame["alpha_norm"].between(-1, 1).all()
        if name == "supplement_cross_domain_intervention.csv":
            continuous = frame[frame["Group"].eq("continuous09")]
            assert (
                continuous.groupby("model_name")["statement_id"].nunique() == 175
            ).all()


def test_manuscript_figure_assets_are_present() -> None:
    main = {f"figure{number}.pdf" for number in range(2, 7)}
    assert main.issubset({path.name for path in (REPOSITORY_ROOT / "figures" / "main").glob("*.pdf")})
    submitted = {
        "submitted_layer_selection.pdf",
        "submitted_qwen14_pca.pdf",
        "submitted_cross_domain_control.pdf",
        "submitted_tsne_exploratory.pdf",
        "submitted_alphabetical_order.pdf",
        "submitted_factual_judgment.pdf",
        "submitted_stance_taking.pdf",
    }
    assert submitted.issubset(
        {path.name for path in (REPOSITORY_ROOT / "figures" / "supplementary").glob("*.pdf")}
    )
