import json
from pathlib import Path

from analysis.python.prepare_axis_benchmark_source import collect_rows


def test_axis_benchmark_summary_uses_absolute_reported_metrics(tmp_path: Path) -> None:
    model_dir = tmp_path / "qwen25_14b"
    model_dir.mkdir()
    payload = {
        "layer": 28,
        "r1": -0.988,
        "p1": 0.0,
        "representaion_similarity": -0.573,
        "representaion_similarity_cosine": -0.574,
        "stance_vector_variance_ratio": 0.188,
        "explained_variance_ratio": [0.557, 0.1],
        "statements": ["1+2=3", "2+2=5"],
    }
    (model_dir / "arithmetic_statement.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    row = collect_rows(tmp_path)[0]
    assert row["projection_correlation"] == 0.988
    assert row["axis_pearson_correlation"] == 0.573
    assert row["n_items"] == 2
