"""Data loading, profiling, and preprocessing orchestration."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataManager:
    """Owns the currently loaded dataset and exposes clean access patterns.

    All mutation methods return a new DataManager so the caller can decide
    whether to commit the change.
    """

    def __init__(self) -> None:
        self._raw: pd.DataFrame | None = None
        self._processed: pd.DataFrame | None = None
        self._target_column: str | None = None
        self._file_path: Path | None = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_csv(self, path: str | Path) -> None:
        """Load a CSV file into memory.

        Args:
            path: Filesystem path to the CSV.

        Raises:
            ValueError: If the file cannot be parsed as CSV.
        """
        path = Path(path)
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise ValueError(f"Cannot parse CSV at {path}: {exc}") from exc

        self._raw = df
        self._processed = df.copy()
        self._file_path = path
        self._target_column = None
        logger.info("Loaded %d rows × %d cols from %s", *df.shape, path.name)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def raw(self) -> pd.DataFrame | None:
        """Original unmodified dataframe."""
        return self._raw

    @property
    def data(self) -> pd.DataFrame | None:
        """Working (possibly preprocessed) dataframe."""
        return self._processed

    @property
    def target_column(self) -> str | None:
        return self._target_column

    @target_column.setter
    def target_column(self, col: str) -> None:
        if self._processed is not None and col not in self._processed.columns:
            raise ValueError(f"Column '{col}' not in dataframe")
        self._target_column = col

    @property
    def file_name(self) -> str:
        return self._file_path.name if self._file_path else "No file loaded"

    @property
    def is_loaded(self) -> bool:
        return self._processed is not None

    @property
    def columns(self) -> list[str]:
        return list(self._processed.columns) if self._processed is not None else []

    @property
    def numeric_columns(self) -> list[str]:
        if self._processed is None:
            return []
        return list(self._processed.select_dtypes(include="number").columns)

    @property
    def categorical_columns(self) -> list[str]:
        if self._processed is None:
            return []
        return list(self._processed.select_dtypes(include=["object", "category"]).columns)

    # ------------------------------------------------------------------
    # Feature / target split
    # ------------------------------------------------------------------

    def get_X_y(self) -> tuple[pd.DataFrame, pd.Series]:
        """Return feature matrix and target vector.

        Returns:
            Tuple of (X, y).

        Raises:
            ValueError: If no target column is set or data is not loaded.
        """
        if self._processed is None:
            raise ValueError("No data loaded")
        if self._target_column is None:
            raise ValueError("No target column selected")

        y = self._processed[self._target_column]
        X = self._processed.drop(columns=[self._target_column])
        return X, y

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    def profile(self) -> dict:
        """Generate a lightweight data profile summary.

        Returns:
            Dictionary with keys: rows, cols, missing, dtypes, describe.
        """
        if self._processed is None:
            return {}

        df = self._processed
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)

        return {
            "rows": len(df),
            "cols": len(df.columns),
            "missing_total": int(missing.sum()),
            "missing_by_col": {
                col: {"count": int(missing[col]), "pct": float(missing_pct[col])}
                for col in df.columns
                if missing[col] > 0
            },
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "describe": df.describe(include="all").to_dict(),
        }

    # ------------------------------------------------------------------
    # Missing value handling
    # ------------------------------------------------------------------

    def drop_missing_rows(self) -> None:
        """Drop all rows containing at least one missing value."""
        if self._processed is not None:
            before = len(self._processed)
            self._processed = self._processed.dropna()
            logger.info("Dropped %d rows with missing values", before - len(self._processed))

    def fill_missing_mean(self) -> None:
        """Fill numeric NaN cells with column mean."""
        if self._processed is not None:
            num_cols = self.numeric_columns
            self._processed[num_cols] = self._processed[num_cols].fillna(
                self._processed[num_cols].mean()
            )
            logger.info("Filled numeric NaNs with column means")

    def fill_missing_median(self) -> None:
        """Fill numeric NaN cells with column median."""
        if self._processed is not None:
            num_cols = self.numeric_columns
            self._processed[num_cols] = self._processed[num_cols].fillna(
                self._processed[num_cols].median()
            )

    def fill_missing_mode(self) -> None:
        """Fill all NaN cells with column mode."""
        if self._processed is not None:
            for col in self._processed.columns:
                mode_val = self._processed[col].mode()
                if not mode_val.empty:
                    self._processed[col] = self._processed[col].fillna(mode_val[0])

    def reset_to_raw(self) -> None:
        """Restore the processed dataframe to the raw loaded state."""
        if self._raw is not None:
            self._processed = self._raw.copy()
            logger.info("Data reset to original")

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def label_encode(self, columns: list[str]) -> None:
        """Label-encode the specified columns in-place.

        Args:
            columns: Column names to encode.
        """
        from sklearn.preprocessing import LabelEncoder

        if self._processed is None:
            return
        le = LabelEncoder()
        for col in columns:
            if col in self._processed.columns:
                self._processed[col] = le.fit_transform(
                    self._processed[col].astype(str)
                )
        logger.info("Label-encoded %s", columns)

    def one_hot_encode(self, columns: list[str]) -> None:
        """One-hot encode the specified columns in-place.

        Args:
            columns: Column names to encode.
        """
        if self._processed is None:
            return
        self._processed = pd.get_dummies(self._processed, columns=columns, drop_first=True)
        logger.info("One-hot encoded %s", columns)

    # ------------------------------------------------------------------
    # Batch prediction support
    # ------------------------------------------------------------------

    def load_prediction_csv(self, path: str | Path) -> pd.DataFrame:
        """Load an unlabelled CSV for batch prediction.

        Args:
            path: Path to the CSV file.

        Returns:
            The loaded dataframe.
        """
        df = pd.read_csv(path)
        logger.info("Prediction batch loaded: %d rows", len(df))
        return df
