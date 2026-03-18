"""AutoML panel — sweep all models, show leaderboard, pick best."""

from __future__ import annotations

import logging

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, QThreadPool, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from sciwizard.core.data_manager import DataManager
from sciwizard.core.model_trainer import AutoMLEntry, ModelTrainer
from sciwizard.ui.widgets.common import Divider, PrimaryButton, SectionHeader
from sciwizard.ui.workers import Worker

logger = logging.getLogger(__name__)


class LeaderboardModel(QAbstractTableModel):
    """Qt model for the AutoML leaderboard table."""

    _HEADERS = ["Rank", "Model", "Score", "Metric", "CV Mean", "CV Std", "Duration (s)"]

    def __init__(self, entries: list[AutoMLEntry], parent=None) -> None:
        super().__init__(parent)
        self._entries = entries

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._entries)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self._HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._HEADERS[section]
        return None

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        entry = self._entries[index.row()]
        col = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            return [
                str(index.row() + 1),
                entry.model_name,
                f"{entry.score:.4f}",
                entry.metric,
                f"{entry.cv_mean:.4f}",
                f"{entry.cv_std:.4f}",
                f"{entry.duration_s:.3f}",
            ][col]
        if role == Qt.ItemDataRole.BackgroundRole and index.row() == 0:
            from PySide6.QtGui import QColor
            return QColor("#2a2a4a")
        return None


class AutoMLPanel(QWidget):
    """AutoML leaderboard panel.

    Signals:
        best_model_selected: Emitted with model name string of top pick.
    """

    best_model_selected = Signal(str)

    def __init__(self, data_manager: DataManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dm = data_manager
        self._pool = QThreadPool.globalInstance()
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)

        root.addWidget(SectionHeader("AutoML — Model Sweep"))
        root.addWidget(Divider())

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Task:"))
        self._task_combo = QComboBox()
        self._task_combo.addItems(["Classification", "Regression"])
        ctrl.addWidget(self._task_combo)

        self._run_btn = PrimaryButton("⚡  Run AutoML")
        self._run_btn.clicked.connect(self._run)
        ctrl.addWidget(self._run_btn)
        ctrl.addStretch()
        root.addLayout(ctrl)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        self._status_label = MutedLabel("Run AutoML to compare all models automatically.")
        root.addWidget(self._status_label)

        self._table = QTableView()
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self._table, stretch=1)

        self._best_label = QLabel("")
        self._best_label.setStyleSheet("font-size: 14px; font-weight: 600; color: #a6e3a1;")
        root.addWidget(self._best_label)

    def _run(self) -> None:
        if not self._dm.is_loaded:
            QMessageBox.warning(self, "No Data", "Please load a dataset first.")
            return
        if not self._dm.target_column:
            QMessageBox.warning(self, "No Target", "Select a target column on the Data tab.")
            return

        try:
            X, y = self._dm.get_X_y()
        except ValueError as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return

        task = self._task_combo.currentText().lower()
        trainer = ModelTrainer(task_type=task)

        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status_label.setText("Running…")

        def _sweep():
            results = []
            total_box = [0]

            def _cb(current, total):
                total_box[0] = total
                pct = int(current / total * 100)
                # Thread-safe update via signal would be ideal, but we emit indirectly
                results.append(pct)

            entries = trainer.automl(X, y, progress_callback=_cb)
            return entries

        worker = Worker(_sweep)
        worker.signals.finished.connect(self._on_done)
        worker.signals.error.connect(self._on_error)
        self._pool.start(worker)

    def _on_done(self, entries: list[AutoMLEntry]) -> None:
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)

        model = LeaderboardModel(entries)
        self._table.setModel(model)
        self._table.resizeColumnsToContents()

        if entries:
            best = entries[0]
            self._best_label.setText(
                f"🏆  Best: {best.model_name}  —  {best.metric}: {best.score:.4f}"
            )
            self._status_label.setText(f"Evaluated {len(entries)} models.")
            self.best_model_selected.emit(best.model_name)

    def _on_error(self, exc: Exception, tb: str) -> None:
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)
        logger.error("AutoML error:\n%s", tb)
        QMessageBox.critical(self, "AutoML Error", str(exc))


# Local import to avoid circular at top
from sciwizard.ui.widgets.common import MutedLabel  # noqa: E402
