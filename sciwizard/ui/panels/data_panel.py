"""Data panel — CSV loading, table preview, profiling, and cleaning."""

from __future__ import annotations

import contextlib
import logging

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from sciwizard.core.data_manager import DataManager
from sciwizard.ui.widgets.common import Divider, MutedLabel, PrimaryButton, SectionHeader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pandas → Qt table model
# ---------------------------------------------------------------------------


class PandasModel(QAbstractTableModel):
    """Expose a pandas DataFrame as a read-only Qt table model.

    Args:
        df: The dataframe to display.
        max_rows: Maximum rows shown (performance guard).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        max_rows: int = 500,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._df = df.head(max_rows)

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._df)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self._df.columns)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            val = self._df.iat[index.row(), index.column()]
            return str(val)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._df.columns[section])
            return str(section + 1)
        return None


# ---------------------------------------------------------------------------
# Data Panel
# ---------------------------------------------------------------------------


class DataPanel(QWidget):
    """Full-featured data management panel.

    Signals:
        data_loaded: Emitted with the DataManager after successful CSV load.
    """

    data_loaded = Signal(object)

    def __init__(self, data_manager: DataManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dm = data_manager
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)

        # Header row
        header_row = QHBoxLayout()
        header_row.addWidget(SectionHeader("Dataset"))
        header_row.addStretch()

        load_btn = PrimaryButton("📂  Load CSV")
        load_btn.setToolTip("Load a comma-separated dataset from disk")
        load_btn.clicked.connect(self._load_csv)
        header_row.addWidget(load_btn)

        root.addLayout(header_row)
        self._file_label = MutedLabel("No file loaded")
        root.addWidget(self._file_label)
        root.addWidget(Divider())

        # Target column selector
        col_row = QHBoxLayout()
        col_row.addWidget(QLabel("Target column:"))
        self._target_combo = QComboBox()
        self._target_combo.setMinimumWidth(180)
        self._target_combo.setEnabled(False)
        self._target_combo.setToolTip("Select the column you want to predict")
        self._target_combo.currentTextChanged.connect(self._on_target_changed)
        col_row.addWidget(self._target_combo)
        col_row.addStretch()
        root.addLayout(col_row)

        # Missing value controls
        mv_group = QGroupBox("Missing Value Handling")
        mv_layout = QHBoxLayout(mv_group)
        mv_layout.setSpacing(8)

        for label, slot in [
            ("Drop rows", self._drop_missing),
            ("Fill mean", self._fill_mean),
            ("Fill median", self._fill_median),
            ("Fill mode", self._fill_mode),
            ("Reset", self._reset_data),
        ]:
            btn = QPushButton(label)
            btn.setEnabled(False)
            btn.clicked.connect(slot)
            mv_layout.addWidget(btn)
            setattr(self, f"_mv_btn_{label.replace(' ', '_').lower()}", btn)

        mv_layout.addStretch()
        root.addWidget(mv_group)

        # Main splitter: table on left, profile on right
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Table view
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.addWidget(QLabel("Preview (up to 500 rows)"))
        self._table_view = QTableView()
        self._table_view.setAlternatingRowColors(True)
        self._table_view.horizontalHeader().setStretchLastSection(False)
        table_layout.addWidget(self._table_view)
        splitter.addWidget(table_container)

        # Profile panel
        profile_container = QWidget()
        profile_layout = QVBoxLayout(profile_container)
        profile_layout.setContentsMargins(0, 0, 0, 0)
        profile_layout.addWidget(QLabel("Data Profile"))
        self._profile_text = QTextEdit()
        self._profile_text.setReadOnly(True)
        self._profile_text.setFontFamily("Courier New")
        self._profile_text.setMinimumWidth(280)
        profile_layout.addWidget(self._profile_text)
        splitter.addWidget(profile_container)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, stretch=1)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _load_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        try:
            self._dm.load_csv(path)
        except ValueError as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return

        self._refresh_ui()
        self.data_loaded.emit(self._dm)

    def _refresh_ui(self) -> None:
        df = self._dm.data
        if df is None:
            return

        self._file_label.setText(
            f"📄  {self._dm.file_name}  —  {len(df):,} rows × {len(df.columns)} columns"
        )

        # Populate table
        self._table_view.setModel(PandasModel(df))
        self._table_view.resizeColumnsToContents()

        # Populate target combo
        self._target_combo.setEnabled(True)
        self._target_combo.blockSignals(True)
        self._target_combo.clear()
        self._target_combo.addItems(list(df.columns))
        self._target_combo.blockSignals(False)
        self._target_combo.setCurrentIndex(len(df.columns) - 1)

        # Enable missing value buttons
        for attr in dir(self):
            if attr.startswith("_mv_btn_"):
                getattr(self, attr).setEnabled(True)

        # Profile
        self._update_profile()

    def _update_profile(self) -> None:
        profile = self._dm.profile()
        if not profile:
            return
        lines = [
            f"Rows:      {profile['rows']:,}",
            f"Columns:   {profile['cols']}",
            f"Missing:   {profile['missing_total']:,} cells",
            "",
            "--- Data Types ---",
        ]
        for col, dtype in profile["dtypes"].items():
            lines.append(f"  {col:<25} {dtype}")

        if profile["missing_by_col"]:
            lines.append("")
            lines.append("--- Missing Values ---")
            for col, info in profile["missing_by_col"].items():
                lines.append(f"  {col:<25} {info['count']:>5} ({info['pct']:.1f}%)")

        self._profile_text.setPlainText("\n".join(lines))

    def _on_target_changed(self, col: str) -> None:
        if col and self._dm.is_loaded:
            with contextlib.suppress(ValueError):
                self._dm.target_column = col

    def _drop_missing(self) -> None:
        self._dm.drop_missing_rows()
        self._refresh_ui()

    def _fill_mean(self) -> None:
        self._dm.fill_missing_mean()
        self._refresh_ui()

    def _fill_median(self) -> None:
        self._dm.fill_missing_median()
        self._refresh_ui()

    def _fill_mode(self) -> None:
        self._dm.fill_missing_mode()
        self._refresh_ui()

    def _reset_data(self) -> None:
        self._dm.reset_to_raw()
        self._refresh_ui()
