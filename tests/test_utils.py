"""Unit tests for utility helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sciwizard.utils.io import safe_read_csv, unique_filename
from sciwizard.utils.metrics import format_metric, primary_metric, score_colour
from sciwizard.utils.validation import (
    clamp,
    require_column,
    require_dataframe,
    require_numeric_columns,
    truncate_str,
)

# ---------------------------------------------------------------------------
# validation.py
# ---------------------------------------------------------------------------


def test_require_dataframe_ok():
    df = pd.DataFrame({"a": [1, 2]})
    assert require_dataframe(df) is df


def test_require_dataframe_none():
    with pytest.raises(ValueError, match="None"):
        require_dataframe(None)


def test_require_dataframe_empty():
    with pytest.raises(ValueError, match="empty"):
        require_dataframe(pd.DataFrame())


def test_require_column_ok():
    df = pd.DataFrame({"x": [1]})
    require_column(df, "x")  # no exception


def test_require_column_missing():
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="not found"):
        require_column(df, "y")


def test_require_numeric_columns_ok():
    df = pd.DataFrame({"a": [1.0, 2.0]})
    require_numeric_columns(df, ["a"])  # no exception


def test_require_numeric_columns_fails():
    df = pd.DataFrame({"a": ["foo", "bar"]})
    with pytest.raises(ValueError, match="numeric"):
        require_numeric_columns(df, ["a"])


def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(20, 0, 10) == 10


def test_truncate_str_short():
    assert truncate_str("hello", 10) == "hello"


def test_truncate_str_long():
    result = truncate_str("hello world", 8)
    assert len(result) <= 8
    assert result.endswith("…")


# ---------------------------------------------------------------------------
# metrics.py
# ---------------------------------------------------------------------------


def test_primary_metric_classification():
    metrics = {"Accuracy": 0.95, "F1 Score": 0.94}
    name, val = primary_metric("classification", metrics)
    assert name == "Accuracy"
    assert val == 0.95


def test_primary_metric_regression():
    metrics = {"R²": 0.88, "MAE": 3.2}
    name, val = primary_metric("regression", metrics)
    assert name == "R²"


def test_primary_metric_empty():
    name, val = primary_metric("classification", {})
    assert name == "—"
    assert val == 0.0


def test_format_metric():
    assert format_metric("Accuracy", 0.9412) == "Accuracy: 0.9412"


def test_score_colour_green():
    assert score_colour(0.95) == "#a6e3a1"


def test_score_colour_yellow():
    assert score_colour(0.75) == "#f9e2af"


def test_score_colour_red():
    assert score_colour(0.5) == "#f38ba8"


def test_score_colour_negative():
    assert score_colour(-0.1) == "#f38ba8"


# ---------------------------------------------------------------------------
# io.py
# ---------------------------------------------------------------------------


def test_safe_read_csv_ok(tmp_path: Path):
    p = tmp_path / "test.csv"
    p.write_text("a,b\n1,2\n3,4\n")
    df = safe_read_csv(p)
    assert df.shape == (2, 2)


def test_safe_read_csv_not_found():
    with pytest.raises(ValueError, match="not found"):
        safe_read_csv("/nonexistent/path.csv")


def test_unique_filename_no_conflict(tmp_path: Path):
    p = unique_filename(tmp_path, "model", ".joblib")
    assert p == tmp_path / "model.joblib"


def test_unique_filename_conflict(tmp_path: Path):
    (tmp_path / "model.joblib").touch()
    p = unique_filename(tmp_path, "model", ".joblib")
    assert p == tmp_path / "model_1.joblib"


def test_unique_filename_multiple_conflicts(tmp_path: Path):
    (tmp_path / "model.joblib").touch()
    (tmp_path / "model_1.joblib").touch()
    p = unique_filename(tmp_path, "model", ".joblib")
    assert p == tmp_path / "model_2.joblib"
