"""Training panel — model selection, hyperparameters, threading, metrics."""

from __future__ import annotations

import logging

from PySide6.QtCore import QThreadPool, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QMessageBox,
    QProgressBar,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from sciwizard.core.data_manager import DataManager
from sciwizard.core.experiment_tracker import ExperimentTracker
from sciwizard.core.model_registry import ModelRegistry
from sciwizard.core.model_trainer import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    ModelTrainer,
    TrainingResult,
)
from sciwizard.ui.widgets.common import Divider, MetricCard, PrimaryButton, SectionHeader
from sciwizard.ui.workers import Worker

logger = logging.getLogger(__name__)


class TrainingPanel(QWidget):
    """Model training control panel.

    Signals:
        training_finished: Emitted with a TrainingResult on success.
    """

    training_finished = Signal(object)

    def __init__(
        self,
        data_manager: DataManager,
        registry: ModelRegistry,
        tracker: ExperimentTracker,
        plugin_models: dict | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._dm = data_manager
        self._registry = registry
        self._tracker = tracker
        self._plugin_models: dict = plugin_models or {}
        self._last_result: TrainingResult | None = None
        self._pool = QThreadPool.globalInstance()
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)

        # ---- Left: controls ----
        left = QWidget()
        left.setMaximumWidth(340)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        left_layout.addWidget(SectionHeader("Train Model"))
        left_layout.addWidget(Divider())

        # Task type
        task_group = QGroupBox("Task")
        task_form = QFormLayout(task_group)
        self._task_combo = QComboBox()
        self._task_combo.addItems(["Classification", "Regression"])
        self._task_combo.currentTextChanged.connect(self._on_task_changed)
        task_form.addRow("Type:", self._task_combo)

        self._scale_cb = QCheckBox("Scale features (StandardScaler)")
        self._scale_cb.setChecked(True)
        task_form.addRow(self._scale_cb)
        left_layout.addWidget(task_group)

        # Model selection
        model_group = QGroupBox("Model")
        model_form = QFormLayout(model_group)
        self._model_combo = QComboBox()
        self._refresh_model_list()
        model_form.addRow("Algorithm:", self._model_combo)
        left_layout.addWidget(model_group)

        # Split
        split_group = QGroupBox("Train / Test Split")
        split_form = QFormLayout(split_group)
        self._test_size_spin = QDoubleSpinBox()
        self._test_size_spin.setRange(0.05, 0.5)
        self._test_size_spin.setSingleStep(0.05)
        self._test_size_spin.setValue(0.2)
        self._test_size_spin.setSuffix("  (fraction)")
        split_form.addRow("Test size:", self._test_size_spin)

        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 9999)
        self._seed_spin.setValue(42)
        split_form.addRow("Random seed:", self._seed_spin)
        left_layout.addWidget(split_group)

        # Save options
        save_group = QGroupBox("Post-Training")
        save_form = QFormLayout(save_group)
        self._autosave_cb = QCheckBox("Save to registry automatically")
        self._autosave_cb.setChecked(True)
        self._autotrack_cb = QCheckBox("Log to experiment tracker")
        self._autotrack_cb.setChecked(True)
        save_form.addRow(self._autosave_cb)
        save_form.addRow(self._autotrack_cb)
        left_layout.addWidget(save_group)

        self._train_btn = PrimaryButton("🚀  Train")
        self._train_btn.clicked.connect(self._start_training)
        left_layout.addWidget(self._train_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        left_layout.addWidget(self._progress)

        left_layout.addStretch()
        root.addWidget(left)

        # ---- Right: results ----
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        right_layout.addWidget(SectionHeader("Results"))
        right_layout.addWidget(Divider())

        self._metrics_container = QWidget()
        self._metrics_layout = QHBoxLayout(self._metrics_container)
        self._metrics_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self._metrics_container)

        self._log_output = QTextEdit()
        self._log_output.setReadOnly(True)
        self._log_output.setFontFamily("Courier New")
        self._log_output.setPlaceholderText("Training log appears here…")
        right_layout.addWidget(self._log_output, stretch=1)

        root.addWidget(right, stretch=1)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_task_changed(self, task: str) -> None:
        self._refresh_model_list()

    def _refresh_model_list(self) -> None:
        task = self._task_combo.currentText().lower()
        catalogue = CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS
        all_names = list(catalogue.keys()) + list(self._plugin_models.keys())
        self._model_combo.clear()
        self._model_combo.addItems(all_names)

    def _start_training(self) -> None:
        if not self._dm.is_loaded:
            QMessageBox.warning(self, "No Data", "Please load a dataset first.")
            return
        if not self._dm.target_column:
            QMessageBox.warning(self, "No Target", "Please select a target column on the Data tab.")
            return

        try:
            X, y = self._dm.get_X_y()
        except ValueError as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return

        task_type = self._task_combo.currentText().lower()
        model_name = self._model_combo.currentText()

        # Inject plugin model into catalogue before training
        from sciwizard.core.model_trainer import CLASSIFICATION_MODELS, REGRESSION_MODELS
        if model_name in self._plugin_models:
            import copy
            target = CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS
            target[model_name] = copy.deepcopy(self._plugin_models[model_name])

        trainer = ModelTrainer(
            task_type=task_type,
            test_size=self._test_size_spin.value(),
            random_state=self._seed_spin.value(),
            scale_features=self._scale_cb.isChecked(),
        )

        self._train_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._log("Starting training…")

        worker = Worker(trainer.train, model_name, X, y)
        worker.signals.finished.connect(self._on_training_done)
        worker.signals.error.connect(self._on_training_error)
        self._pool.start(worker)

    def _on_training_done(self, result: TrainingResult) -> None:
        self._progress.setVisible(False)
        self._train_btn.setEnabled(True)
        self._last_result = result

        # Show metrics
        for i in reversed(range(self._metrics_layout.count())):
            w = self._metrics_layout.itemAt(i).widget()
            if w:
                w.deleteLater()

        row = MetricCard.row(result.metrics)
        self._metrics_layout.addWidget(row)

        # Log
        cv_mean = result.cv_scores.mean() if result.cv_scores.size else 0.0
        cv_std = result.cv_scores.std() if result.cv_scores.size else 0.0
        self._log(f"\n✅  {result.model_name} trained in {result.train_duration_s:.3f}s")
        self._log(f"   Metrics: {result.metrics}")
        self._log(f"   CV ({len(result.cv_scores)}-fold): {cv_mean:.4f} ± {cv_std:.4f}")

        # Persist
        model_id = None
        if self._autosave_cb.isChecked():
            model_id = self._registry.save(result)
            self._log(f"   Saved to registry as {model_id}")

        if self._autotrack_cb.isChecked():
            self._tracker.log(
                result,
                dataset_name=self._dm.file_name,
                model_id=model_id,
            )
            self._log("   Experiment logged")

        self.training_finished.emit(result)

    def _on_training_error(self, exc: Exception, tb: str) -> None:
        self._progress.setVisible(False)
        self._train_btn.setEnabled(True)
        self._log(f"\n❌  Training failed: {exc}")
        logger.error("Training error:\n%s", tb)
        QMessageBox.critical(self, "Training Error", str(exc))

    def _log(self, msg: str) -> None:
        self._log_output.append(msg)
