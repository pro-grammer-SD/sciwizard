"""Prediction panel — single-row form-based and batch CSV prediction."""

from __future__ import annotations

import logging

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, QThreadPool
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from sciwizard.core.data_manager import DataManager
from sciwizard.core.model_trainer import TrainingResult
from sciwizard.ui.widgets.common import Divider, PrimaryButton, SectionHeader
from sciwizard.ui.workers import Worker

logger = logging.getLogger(__name__)


class _PandasReadonlyModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._df = df

    def rowCount(self, p=QModelIndex()):
        return len(self._df)

    def columnCount(self, p=QModelIndex()):
        return len(self._df.columns)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            return str(self._df.iat[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._df.columns[section])
            return str(section + 1)
        return None


class PredictionPanel(QWidget):
    """Single-row and batch prediction panel."""

    def __init__(
        self,
        data_manager: DataManager,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._dm = data_manager
        self._result: TrainingResult | None = None
        self._field_inputs: dict[str, QLineEdit] = {}
        self._pool = QThreadPool.globalInstance()
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)

        root.addWidget(SectionHeader("Prediction"))
        root.addWidget(Divider())

        self._no_model_label = QLabel(
            "⚠️  No trained model loaded. Train a model first."
        )
        self._no_model_label.setStyleSheet("color: #f9e2af; font-size: 13px;")
        root.addWidget(self._no_model_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ---- Single prediction ----
        single_widget = QGroupBox("Single Prediction")
        single_layout = QVBoxLayout(single_widget)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._form_container = QWidget()
        self._form_layout = QFormLayout(self._form_container)
        self._form_layout.setSpacing(8)
        scroll.setWidget(self._form_container)
        single_layout.addWidget(scroll)

        self._predict_btn = PrimaryButton("Predict")
        self._predict_btn.setEnabled(False)
        self._predict_btn.clicked.connect(self._predict_single)
        single_layout.addWidget(self._predict_btn)

        self._single_result_label = QLabel("")
        self._single_result_label.setStyleSheet("font-size: 18px; font-weight: 700; color: #63d4c4;")
        single_layout.addWidget(self._single_result_label)
        splitter.addWidget(single_widget)

        # ---- Batch prediction ----
        batch_widget = QGroupBox("Batch Prediction (CSV)")
        batch_layout = QVBoxLayout(batch_widget)

        load_row = QHBoxLayout()
        self._batch_load_btn = QPushButton("📂  Load CSV")
        self._batch_load_btn.setEnabled(False)
        self._batch_load_btn.clicked.connect(self._load_batch)
        load_row.addWidget(self._batch_load_btn)

        self._batch_save_btn = QPushButton("💾  Save Results")
        self._batch_save_btn.setEnabled(False)
        self._batch_save_btn.clicked.connect(self._save_batch)
        load_row.addWidget(self._batch_save_btn)
        load_row.addStretch()
        batch_layout.addLayout(load_row)

        self._batch_table = QTableView()
        self._batch_table.setAlternatingRowColors(True)
        batch_layout.addWidget(self._batch_table)

        self._batch_df: pd.DataFrame | None = None
        splitter.addWidget(batch_widget)

        root.addWidget(splitter, stretch=1)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def load_result(self, result: TrainingResult) -> None:
        """Populate prediction form from a training result.

        Args:
            result: The result whose feature_names define the form fields.
        """
        self._result = result
        self._no_model_label.setVisible(False)
        self._predict_btn.setEnabled(True)
        self._batch_load_btn.setEnabled(True)

        # Rebuild form
        for i in reversed(range(self._form_layout.count())):
            item = self._form_layout.itemAt(i)
            if item and item.widget():
                item.widget().deleteLater()
        self._field_inputs.clear()

        for feature in result.feature_names:
            edit = QLineEdit()
            edit.setPlaceholderText("0")
            self._form_layout.addRow(feature + ":", edit)
            self._field_inputs[feature] = edit

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _predict_single(self) -> None:
        if not self._result:
            return
        try:
            row = {
                feat: float(edit.text() or 0)
                for feat, edit in self._field_inputs.items()
            }
        except ValueError as exc:
            QMessageBox.warning(self, "Input Error", f"Invalid value: {exc}")
            return

        df_row = pd.DataFrame([row])
        pipeline = self._result.pipeline

        try:
            pred = pipeline.predict(df_row)[0]
            if self._result.label_encoder:
                pred = self._result.label_encoder.inverse_transform([int(pred)])[0]
            self._single_result_label.setText(f"Prediction:  {pred}")
        except Exception as exc:
            logger.exception("Single prediction failed")
            QMessageBox.critical(self, "Prediction Error", str(exc))

    def _load_batch(self) -> None:
        if not self._result:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load CSV", "", "CSV Files (*.csv)")
        if not path:
            return

        def _run():
            df = pd.read_csv(path)
            # Keep only known feature columns
            known = [c for c in self._result.feature_names if c in df.columns]
            X = df[known].fillna(0)
            preds = self._result.pipeline.predict(X)
            if self._result.label_encoder:
                preds = self._result.label_encoder.inverse_transform(preds.astype(int))
            df["prediction"] = preds
            return df

        worker = Worker(_run)
        worker.signals.finished.connect(self._on_batch_done)
        worker.signals.error.connect(
            lambda exc, tb: QMessageBox.critical(self, "Batch Error", str(exc))
        )
        self._pool.start(worker)

    def _on_batch_done(self, df: pd.DataFrame) -> None:
        self._batch_df = df
        self._batch_table.setModel(_PandasReadonlyModel(df))
        self._batch_table.resizeColumnsToContents()
        self._batch_save_btn.setEnabled(True)

    def _save_batch(self) -> None:
        if self._batch_df is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "predictions.csv", "CSV Files (*.csv)"
        )
        if path:
            self._batch_df.to_csv(path, index=False)
            QMessageBox.information(self, "Saved", f"Results saved to:\n{path}")
