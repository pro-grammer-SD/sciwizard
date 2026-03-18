"""Experiment history panel — browse and clear past runs."""

from __future__ import annotations

import logging

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
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

from sciwizard.core.experiment_tracker import ExperimentTracker
from sciwizard.ui.widgets.common import Divider, SectionHeader

logger = logging.getLogger(__name__)


class ExperimentModel(QAbstractTableModel):
    _COLS = [
        "Run ID", "Timestamp", "Dataset", "Model", "Task",
        "Key Metric", "CV Mean", "Duration (s)", "Notes",
    ]

    def __init__(self, entries: list[dict], parent=None) -> None:
        super().__init__(parent)
        self._entries = entries

    def rowCount(self, p=QModelIndex()):
        return len(self._entries)

    def columnCount(self, p=QModelIndex()):
        return len(self._COLS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._COLS[section]
        return None

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        e = self._entries[index.row()]
        col = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0:
                return e.get("run_id", "")
            if col == 1:
                return e.get("timestamp", "")[:19].replace("T", " ")
            if col == 2:
                return e.get("dataset", "")
            if col == 3:
                return e.get("model_name", "")
            if col == 4:
                return e.get("task_type", "")
            if col == 5:
                metrics = e.get("metrics", {})
                if metrics:
                    k, v = next(iter(metrics.items()))
                    return f"{k}: {v}"
                return "—"
            if col == 6:
                cv = e.get("cv_mean")
                return f"{cv:.4f}" if cv is not None else "—"
            if col == 7:
                return str(e.get("train_duration_s", ""))
            if col == 8:
                return e.get("notes", "")
        return None


class ExperimentsPanel(QWidget):
    """Browse experiment history."""

    def __init__(
        self,
        tracker: ExperimentTracker,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._tracker = tracker
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)

        hdr = QHBoxLayout()
        hdr.addWidget(SectionHeader("Experiment History"))
        hdr.addStretch()

        refresh_btn = QPushButton("🔄  Refresh")
        refresh_btn.clicked.connect(self.refresh)
        hdr.addWidget(refresh_btn)

        clear_btn = QPushButton("🗑  Clear Log")
        clear_btn.setStyleSheet("QPushButton { color: #f38ba8; }")
        clear_btn.clicked.connect(self._clear)
        hdr.addWidget(clear_btn)
        root.addLayout(hdr)
        root.addWidget(Divider())

        self._table = QTableView()
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self._table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self._table, stretch=1)

        self._count_label = QLabel("")
        self._count_label.setObjectName("muted")
        root.addWidget(self._count_label)

        self.refresh()

    def refresh(self) -> None:
        """Reload history from disk."""
        entries = self._tracker.load_history()
        self._table.setModel(ExperimentModel(entries))
        self._table.resizeColumnsToContents()
        self._count_label.setText(f"{len(entries)} experiment run(s) logged")

    def _clear(self) -> None:
        reply = QMessageBox.question(
            self,
            "Clear Experiment Log",
            "This will permanently delete all experiment history. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._tracker.clear()
            self.refresh()
