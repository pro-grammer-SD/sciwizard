"""Preprocessing panel — scaling, encoding, and feature engineering controls."""

from __future__ import annotations

import logging

from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from sciwizard.core.data_manager import DataManager
from sciwizard.ui.widgets.common import Divider, MutedLabel, SectionHeader

logger = logging.getLogger(__name__)


class PreprocessingPanel(QWidget):
    """Interactive preprocessing configuration panel."""

    def __init__(self, data_manager: DataManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dm = data_manager
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)

        root.addWidget(SectionHeader("Preprocessing"))
        root.addWidget(Divider())

        row = QHBoxLayout()
        row.setSpacing(16)
        row.setAlignment(__import__("PySide6.QtCore", fromlist=["Qt"]).Qt.AlignmentFlag.AlignTop)

        # ---- Encoding group ----
        enc_group = QGroupBox("Categorical Encoding")
        enc_layout = QVBoxLayout(enc_group)
        enc_layout.addWidget(MutedLabel("Select columns to encode:"))

        self._enc_list = QListWidget()
        self._enc_list.setMaximumHeight(180)
        enc_layout.addWidget(self._enc_list)

        enc_mode_layout = QHBoxLayout()
        enc_mode_layout.addWidget(QLabel("Method:"))
        self._enc_combo = QComboBox()
        self._enc_combo.addItems(["Label Encode", "One-Hot Encode"])
        enc_mode_layout.addWidget(self._enc_combo)
        enc_layout.addLayout(enc_mode_layout)

        apply_enc_btn = QPushButton("Apply Encoding")
        apply_enc_btn.clicked.connect(self._apply_encoding)
        enc_layout.addWidget(apply_enc_btn)
        row.addWidget(enc_group)

        # ---- Scaling group ----
        scale_group = QGroupBox("Feature Scaling Info")
        scale_layout = QVBoxLayout(scale_group)
        scale_layout.addWidget(
            MutedLabel(
                "StandardScaler is applied automatically during training "
                "when 'Scale features' is checked in the Training panel.\n\n"
                "Available scalers:\n• StandardScaler (z-score)\n• MinMaxScaler (0–1)\n• RobustScaler (IQR-based)"
            )
        )
        scale_layout.addStretch()
        row.addWidget(scale_group)

        # ---- Feature engineering ----
        feat_group = QGroupBox("Feature Engineering")
        feat_layout = QVBoxLayout(feat_group)

        self._drop_list = QListWidget()
        self._drop_list.setMaximumHeight(180)
        feat_layout.addWidget(MutedLabel("Columns to drop:"))
        feat_layout.addWidget(self._drop_list)

        drop_btn = QPushButton("Drop selected columns")
        drop_btn.clicked.connect(self._drop_columns)
        feat_layout.addWidget(drop_btn)
        row.addWidget(feat_group)

        root.addLayout(row)

        # ---- Log ----
        root.addWidget(QLabel("Operations log:"))
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(120)
        self._log.setFontFamily("Courier New")
        root.addWidget(self._log)
        root.addStretch()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def refresh_columns(self) -> None:
        """Rebuild column lists from the current data manager state."""
        cat_cols = self._dm.categorical_columns
        all_cols = self._dm.columns

        self._enc_list.clear()
        for col in cat_cols:
            item = QListWidgetItem(col)
            item.setCheckState(__import__("PySide6.QtCore", fromlist=["Qt"]).Qt.CheckState.Unchecked)
            self._enc_list.addItem(item)

        self._drop_list.clear()
        for col in all_cols:
            item = QListWidgetItem(col)
            item.setCheckState(__import__("PySide6.QtCore", fromlist=["Qt"]).Qt.CheckState.Unchecked)
            self._drop_list.addItem(item)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _get_checked(self, list_widget: QListWidget) -> list[str]:
        from PySide6.QtCore import Qt
        checked = []
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                checked.append(item.text())
        return checked

    def _apply_encoding(self) -> None:
        cols = self._get_checked(self._enc_list)
        if not cols:
            QMessageBox.information(self, "Nothing selected", "Check at least one column.")
            return
        method = self._enc_combo.currentText()
        try:
            if method == "Label Encode":
                self._dm.label_encode(cols)
            else:
                self._dm.one_hot_encode(cols)
            self._log.append(f"✅ {method} applied to: {', '.join(cols)}")
            self.refresh_columns()
        except Exception as exc:
            self._log.append(f"❌ Error: {exc}")

    def _drop_columns(self) -> None:
        cols = self._get_checked(self._drop_list)
        if not cols:
            QMessageBox.information(self, "Nothing selected", "Check columns to drop.")
            return
        df = self._dm.data
        if df is None:
            return
        existing = [c for c in cols if c in df.columns]
        self._dm._processed = df.drop(columns=existing)
        self._log.append(f"✅ Dropped columns: {', '.join(existing)}")
        self.refresh_columns()
