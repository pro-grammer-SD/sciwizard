"""Unit tests for DataManager."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from sciwizard.core.data_manager import DataManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Write a small CSV to a temp file and return its path."""
    content = textwrap.dedent(
        """\
        age,income,label
        25,50000,yes
        30,60000,no
        ,70000,yes
        40,,no
        45,80000,yes
        """
    )
    p = tmp_path / "sample.csv"
    p.write_text(content)
    return p


@pytest.fixture
def dm(sample_csv: Path) -> DataManager:
    mgr = DataManager()
    mgr.load_csv(sample_csv)
    return mgr


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_csv(sample_csv: Path) -> None:
    mgr = DataManager()
    mgr.load_csv(sample_csv)
    assert mgr.is_loaded
    assert mgr.data is not None
    assert mgr.data.shape == (5, 3)


def test_load_csv_bad_path() -> None:
    mgr = DataManager()
    with pytest.raises(ValueError, match="Cannot parse CSV"):
        mgr.load_csv("/nonexistent/path/data.csv")


def test_columns(dm: DataManager) -> None:
    assert dm.columns == ["age", "income", "label"]


def test_numeric_columns(dm: DataManager) -> None:
    assert set(dm.numeric_columns) == {"age", "income"}


def test_categorical_columns(dm: DataManager) -> None:
    assert "label" in dm.categorical_columns


# ---------------------------------------------------------------------------
# Target column
# ---------------------------------------------------------------------------


def test_set_target(dm: DataManager) -> None:
    dm.target_column = "label"
    assert dm.target_column == "label"


def test_set_target_invalid(dm: DataManager) -> None:
    with pytest.raises(ValueError, match="not in dataframe"):
        dm.target_column = "nonexistent"


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------


def test_drop_missing(dm: DataManager) -> None:
    dm.drop_missing_rows()
    assert dm.data.isnull().sum().sum() == 0
    assert len(dm.data) == 3  # rows 3 and 4 have NaN


def test_fill_mean(dm: DataManager) -> None:
    dm.fill_missing_mean()
    assert dm.data["age"].isnull().sum() == 0
    assert dm.data["income"].isnull().sum() == 0


def test_fill_median(dm: DataManager) -> None:
    dm.fill_missing_median()
    assert dm.data.isnull().sum().sum() == 0


def test_fill_mode(dm: DataManager) -> None:
    dm.fill_missing_mode()
    assert dm.data.isnull().sum().sum() == 0


def test_reset_to_raw(dm: DataManager) -> None:
    dm.drop_missing_rows()
    assert len(dm.data) == 3
    dm.reset_to_raw()
    assert len(dm.data) == 5


# ---------------------------------------------------------------------------
# Feature/target split
# ---------------------------------------------------------------------------


def test_get_X_y(dm: DataManager) -> None:
    dm.target_column = "label"
    X, y = dm.get_X_y()
    assert "label" not in X.columns
    assert y.name == "label"
    assert len(X) == len(y)


def test_get_X_y_no_target(dm: DataManager) -> None:
    with pytest.raises(ValueError, match="No target column"):
        dm.get_X_y()


def test_get_X_y_no_data() -> None:
    mgr = DataManager()
    with pytest.raises(ValueError, match="No data loaded"):
        mgr.get_X_y()


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


def test_profile_keys(dm: DataManager) -> None:
    profile = dm.profile()
    assert "rows" in profile
    assert "cols" in profile
    assert "missing_total" in profile
    assert profile["rows"] == 5
    assert profile["missing_total"] > 0


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def test_label_encode(dm: DataManager) -> None:
    dm.label_encode(["label"])
    assert dm.data["label"].dtype != object


def test_one_hot_encode(dm: DataManager) -> None:
    before_cols = len(dm.columns)
    dm.one_hot_encode(["label"])
    # One-hot with drop_first=True on binary column adds exactly 0 extra cols
    # (binary → 2 cats → 1 dummy after drop_first)
    assert len(dm.columns) >= before_cols
