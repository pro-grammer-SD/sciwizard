"""Hyperparameter tuning panel — GridSearchCV with a live results table."""

from __future__ import annotations

import logging

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, QThreadPool
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from sciwizard.core.data_manager import DataManager
from sciwizard.core.model_trainer import CLASSIFICATION_MODELS, REGRESSION_MODELS
from sciwizard.ui.widgets.common import Divider, MutedLabel, PrimaryButton, SectionHeader
from sciwizard.ui.workers import Worker

logger = logging.getLogger(__name__)

_DEFAULT_GRIDS: dict[str, dict] = {
    "Random Forest": {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_split": [2, 5],
    },
    "Gradient Boosting": {
        "model__n_estimators": [50, 100],
        "model__learning_rate": [0.05, 0.1, 0.2],
        "model__max_depth": [3, 5],
    },
    "Logistic Regression": {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__solver": ["lbfgs", "saga"],
    },
    "Decision Tree": {
        "model__max_depth": [None, 3, 5, 10],
        "model__min_samples_split": [2, 5, 10],
    },
    "K-Nearest Neighbours": {
        "model__n_neighbors": [3, 5, 7, 11],
        "model__weights": ["uniform", "distance"],
    },
    "Ridge Regression": {
        "model__alpha": [0.1, 1.0, 10.0, 100.0],
    },
}


class _GridResultModel(QAbstractTableModel):
    _COLS = ["Rank", "Params", "Mean Score", "Std Score"]

    def __init__(self, rows: list[dict], parent=None) -> None:
        super().__init__(parent)
        self._rows = rows

    def rowCount(self, p=QModelIndex()):
        return len(self._rows)

    def columnCount(self, p=QModelIndex()):
        return len(self._COLS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._COLS[section]
        return None

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        col = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0:
                return str(row.get("rank_test_score", ""))
            if col == 1:
                # Pretty-print params without "model__" prefix clutter
                params = row.get("params", {})
                clean = {k.replace("model__", ""): v for k, v in params.items()}
                return str(clean)
            if col == 2:
                return f"{row.get('mean_test_score', 0):.4f}"
            if col == 3:
                return f"{row.get('std_test_score', 0):.4f}"
        if role == Qt.ItemDataRole.BackgroundRole and index.row() == 0:
            from PySide6.QtGui import QColor
            return QColor("#2a2a4a")
        return None


class HyperparamPanel(QWidget):
    """GridSearchCV hyperparameter tuning panel."""

    def __init__(self, data_manager: DataManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dm = data_manager
        self._pool = QThreadPool.globalInstance()
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)

        root.addWidget(SectionHeader("Hyperparameter Tuning"))
        root.addWidget(Divider())

        root.addWidget(MutedLabel(
            "GridSearchCV exhaustively searches the parameter grid below using "
            "5-fold cross-validation. Edit the JSON grid to customise the search space."
        ))

        # Controls
        ctrl_group = QGroupBox("Search Configuration")
        ctrl_form = QFormLayout(ctrl_group)

        from PySide6.QtWidgets import QComboBox, QSpinBox
        self._task_combo = QComboBox()
        self._task_combo.addItems(["Classification", "Regression"])
        self._task_combo.currentTextChanged.connect(self._refresh_model_list)
        ctrl_form.addRow("Task:", self._task_combo)

        self._model_combo = QComboBox()
        self._model_combo.currentTextChanged.connect(self._load_default_grid)
        ctrl_form.addRow("Model:", self._model_combo)

        self._cv_spin = QSpinBox()
        self._cv_spin.setRange(2, 10)
        self._cv_spin.setValue(5)
        ctrl_form.addRow("CV folds:", self._cv_spin)

        root.addWidget(ctrl_group)

        # Grid editor
        grid_group = QGroupBox("Parameter Grid  (Python dict literal)")
        grid_layout = QVBoxLayout(grid_group)
        self._grid_editor = QPlainTextEdit()
        self._grid_editor.setFont(__import__("PySide6.QtGui", fromlist=["QFont"]).QFont("Courier New", 11))
        self._grid_editor.setMaximumHeight(160)
        grid_layout.addWidget(self._grid_editor)
        root.addWidget(grid_group)

        # Run button + progress
        run_row = QHBoxLayout()
        self._run_btn = PrimaryButton("🔍  Run Grid Search")
        self._run_btn.clicked.connect(self._run)
        run_row.addWidget(self._run_btn)
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        run_row.addWidget(self._progress)
        run_row.addStretch()
        root.addLayout(run_row)

        # Results table
        self._results_label = QLabel("")
        root.addWidget(self._results_label)
        self._table = QTableView()
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self._table, stretch=1)

        # Initialise
        self._refresh_model_list()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _refresh_model_list(self) -> None:
        task = self._task_combo.currentText().lower()
        catalogue = CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        self._model_combo.addItems(list(catalogue.keys()))
        self._model_combo.blockSignals(False)
        self._load_default_grid(self._model_combo.currentText())

    def _load_default_grid(self, model_name: str) -> None:
        grid = _DEFAULT_GRIDS.get(model_name, {"model__C": [0.1, 1.0, 10.0]})
        self._grid_editor.setPlainText(repr(grid))

    def _run(self) -> None:
        if not self._dm.is_loaded:
            QMessageBox.warning(self, "No Data", "Load a dataset first.")
            return
        if not self._dm.target_column:
            QMessageBox.warning(self, "No Target", "Select a target column on the Data tab.")
            return

        try:
            X, y = self._dm.get_X_y()
            grid_text = self._grid_editor.toPlainText().strip()
            param_grid = eval(grid_text, {"__builtins__": {}})  # safe-ish literal eval
            if not isinstance(param_grid, dict):
                raise ValueError("Grid must be a Python dict")
        except Exception as exc:
            QMessageBox.critical(self, "Invalid Grid", f"Could not parse parameter grid:\n{exc}")
            return

        import copy
        task = self._task_combo.currentText().lower()
        model_name = self._model_combo.currentText()
        catalogue = CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS
        estimator = copy.deepcopy(catalogue[model_name])
        cv = self._cv_spin.value()

        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)

        def _search():
            from sklearn.model_selection import GridSearchCV
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import LabelEncoder, StandardScaler

            Xp = X.copy()
            yp = y.copy()
            le = None

            for col in Xp.select_dtypes(include=["object", "category"]).columns:
                Xp[col] = LabelEncoder().fit_transform(Xp[col].astype(str))
            if task == "classification" and yp.dtype == object:
                le = LabelEncoder()
                yp = pd.Series(le.fit_transform(yp), name=yp.name)

            pipeline = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
            scoring = "accuracy" if task == "classification" else "r2"
            gs = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
            gs.fit(Xp, yp)
            return gs.cv_results_, gs.best_params_, gs.best_score_

        worker = Worker(_search)
        worker.signals.finished.connect(self._on_done)
        worker.signals.error.connect(self._on_error)
        self._pool.start(worker)

    def _on_done(self, result: tuple) -> None:
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)

        cv_results, best_params, best_score = result

        # Build list of dicts for table model
        n = len(cv_results["rank_test_score"])
        rows = []
        for i in range(n):
            rows.append({
                "rank_test_score": cv_results["rank_test_score"][i],
                "params": cv_results["params"][i],
                "mean_test_score": cv_results["mean_test_score"][i],
                "std_test_score": cv_results["std_test_score"][i],
            })
        rows.sort(key=lambda r: r["rank_test_score"])

        self._table.setModel(_GridResultModel(rows))
        self._table.resizeColumnsToContents()

        clean_best = {k.replace("model__", ""): v for k, v in best_params.items()}
        self._results_label.setText(
            f"✅  Best score: {best_score:.4f}  |  Best params: {clean_best}"
        )
        self._results_label.setStyleSheet("color: #a6e3a1; font-weight: 600;")

    def _on_error(self, exc: Exception, tb: str) -> None:
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)
        logger.error("Grid search error:\n%s", tb)
        QMessageBox.critical(self, "Search Error", str(exc))
