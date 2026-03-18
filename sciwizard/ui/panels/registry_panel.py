"""Model Registry panel — list, load, and delete saved models."""

from __future__ import annotations

import logging

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from sciwizard.core.model_registry import ModelRegistry
from sciwizard.ui.widgets.common import Divider, PrimaryButton, SectionHeader

logger = logging.getLogger(__name__)


class RegistryTableModel(QAbstractTableModel):
    """Qt model for the model registry list."""

    _COLS = ["ID", "Alias", "Algorithm", "Task", "Key Metric", "CV Mean", "Saved At"]

    def __init__(self, models: list[dict], parent=None) -> None:
        super().__init__(parent)
        self._models = models

    def rowCount(self, p=QModelIndex()):
        return len(self._models)

    def columnCount(self, p=QModelIndex()):
        return len(self._COLS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._COLS[section]
        return None

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        m = self._models[index.row()]
        col = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0:
                return m.get("model_id", "")
            if col == 1:
                return m.get("alias", "")
            if col == 2:
                return m.get("model_name", "")
            if col == 3:
                return m.get("task_type", "")
            if col == 4:
                metrics = m.get("metrics", {})
                if metrics:
                    k, v = next(iter(metrics.items()))
                    return f"{k}: {v}"
                return "—"
            if col == 5:
                cv = m.get("cv_mean")
                return f"{cv:.4f}" if cv is not None else "—"
            if col == 6:
                return m.get("saved_at", "")[:19].replace("T", " ")
        return None

    def get_model_id(self, row: int) -> str:
        return self._models[row].get("model_id", "")


class RegistryPanel(QWidget):
    """Model registry browser panel.

    Signals:
        model_loaded: Emitted with (pipeline, metadata) when user clicks Load.
    """

    model_loaded = Signal(object, dict)

    def __init__(
        self, registry: ModelRegistry, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._registry = registry
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)

        header_row = QHBoxLayout()
        header_row.addWidget(SectionHeader("Model Registry"))
        header_row.addStretch()
        refresh_btn = QPushButton("🔄  Refresh")
        refresh_btn.clicked.connect(self.refresh)
        header_row.addWidget(refresh_btn)
        root.addLayout(header_row)
        root.addWidget(Divider())

        self._table = QTableView()
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        root.addWidget(self._table, stretch=1)

        btn_row = QHBoxLayout()
        self._load_btn = PrimaryButton("📦  Load Selected")
        self._load_btn.clicked.connect(self._load_selected)
        btn_row.addWidget(self._load_btn)

        self._delete_btn = QPushButton("🗑  Delete Selected")
        self._delete_btn.setStyleSheet(
            "QPushButton { color: #f38ba8; border-color: #f38ba8; }"
            "QPushButton:hover { background-color: #3b1e27; }"
        )
        self._delete_btn.clicked.connect(self._delete_selected)
        btn_row.addWidget(self._delete_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

        self._status = QLabel("")
        self._status.setObjectName("muted")
        root.addWidget(self._status)

        self.refresh()

    def refresh(self) -> None:
        """Reload models from disk."""
        models = self._registry.list_models()
        self._table.setModel(RegistryTableModel(models))
        self._status.setText(f"{len(models)} model(s) in registry")

    def _load_selected(self) -> None:
        idx = self._table.currentIndex()
        if not idx.isValid():
            QMessageBox.information(self, "Select a model", "Click a row first.")
            return
        model = self._table.model()
        model_id = model.get_model_id(idx.row())
        try:
            pipeline, meta = self._registry.load(model_id)
            self.model_loaded.emit(pipeline, meta)
            self._status.setText(f"Loaded: {meta.get('alias')} ({model_id})")
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))

    def _delete_selected(self) -> None:
        idx = self._table.currentIndex()
        if not idx.isValid():
            return
        model = self._table.model()
        model_id = model.get_model_id(idx.row())
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete model {model_id}? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._registry.delete(model_id)
            self.refresh()
