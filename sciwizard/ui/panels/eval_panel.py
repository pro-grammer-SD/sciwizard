"""Evaluation panel — confusion matrix, ROC curve, metrics dashboard."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QHBoxLayout,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from sciwizard.core.model_trainer import TrainingResult
from sciwizard.ui.widgets.common import Divider, MetricCard, MutedLabel, SectionHeader
from sciwizard.ui.widgets.plot_canvas import PlotCanvas

logger = logging.getLogger(__name__)


class EvaluationPanel(QWidget):
    """Visualise model performance after training."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: TrainingResult | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)

        root.addWidget(SectionHeader("Model Evaluation"))
        root.addWidget(Divider())

        self._info_label = MutedLabel("Train a model to populate evaluation charts.")
        root.addWidget(self._info_label)

        self._metrics_container = QWidget()
        self._metrics_layout = QHBoxLayout(self._metrics_container)
        self._metrics_layout.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._metrics_container)

        self._tabs = QTabWidget()
        root.addWidget(self._tabs, stretch=1)

        self._cm_canvas  = PlotCanvas(show_toolbar=False)
        self._roc_canvas = PlotCanvas(show_toolbar=False)
        self._cv_canvas  = PlotCanvas(show_toolbar=False)

        self._tabs.addTab(self._cm_canvas,  "Confusion Matrix")
        self._tabs.addTab(self._roc_canvas, "ROC Curve")
        self._tabs.addTab(self._cv_canvas,  "CV Distribution")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def display_result(self, result: TrainingResult) -> None:
        """Render all evaluation charts for the given result.

        Each chart is drawn independently inside a try/except so a failure
        in one never prevents the others from rendering.

        Args:
            result: A completed TrainingResult from the trainer.
        """
        self._result = result
        self._info_label.setText(
            f"Showing evaluation for: {result.model_name}  |  Task: {result.task_type}"
        )

        # Metrics row
        for i in reversed(range(self._metrics_layout.count())):
            w = self._metrics_layout.itemAt(i).widget()
            if w:
                w.deleteLater()
        self._metrics_layout.addWidget(MetricCard.row(result.metrics))

        for draw_fn in (self._draw_confusion_matrix, self._draw_roc, self._draw_cv):
            try:
                draw_fn(result)
            except Exception:
                logger.exception("Eval draw failed in %s", draw_fn.__name__)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numeric_labels(y: pd.Series) -> tuple[np.ndarray, list | None]:
        """Coerce y to a numeric array, returning (array, classes_list_or_None).

        Works regardless of whether y is already int/float, or still contains
        string labels (can happen with older pandas object dtype or newer
        StringDtype when the LabelEncoder was not applied).
        """
        if pd.api.types.is_numeric_dtype(y):
            return np.asarray(y), None
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        arr = le.fit_transform(y.astype(str))
        return arr, list(le.classes_)

    # ------------------------------------------------------------------
    # Plot implementations
    # ------------------------------------------------------------------

    def _draw_confusion_matrix(self, result: TrainingResult) -> None:
        self._cm_canvas.clear()
        ax = self._cm_canvas.get_ax()

        if result.task_type != "classification":
            ax.text(0.5, 0.5, "Confusion matrix: classification only.",
                    ha="center", va="center", fontsize=12, transform=ax.transAxes)
            self._cm_canvas.draw()
            return

        from sklearn.metrics import confusion_matrix

        y_true_num, fallback_classes = self._to_numeric_labels(result.y_test)
        y_pred_num, _               = self._to_numeric_labels(
            pd.Series(result.y_pred, name="pred")
        )

        cm = confusion_matrix(y_true_num, y_pred_num)
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        self._cm_canvas.figure.colorbar(im, ax=ax)

        classes = result.classes or fallback_classes or list(range(cm.shape[0]))
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels([str(c) for c in classes], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels([str(c) for c in classes], fontsize=8)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "#cdd6f4", fontsize=9)

        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        ax.set_title("Confusion Matrix")
        self._cm_canvas.draw()

    def _draw_roc(self, result: TrainingResult) -> None:
        self._roc_canvas.clear()
        ax = self._roc_canvas.get_ax()

        if result.task_type != "classification" or result.y_prob is None:
            ax.text(0.5, 0.5,
                    "ROC curve requires a classification model\nwith probability output.",
                    ha="center", va="center", fontsize=12, transform=ax.transAxes)
            self._roc_canvas.draw()
            return

        from sklearn.metrics import roc_auc_score, roc_curve
        from sklearn.preprocessing import label_binarize

        y_prob = result.y_prob

        # Coerce y_test to numeric — required by roc_curve regardless of sklearn version
        y_true_num, fallback_classes = self._to_numeric_labels(result.y_test)
        classes = result.classes or fallback_classes or sorted(set(y_true_num.tolist()))

        if len(classes) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true_num, y_prob[:, 1], pos_label=1)
            auc = roc_auc_score(y_true_num, y_prob[:, 1])
            ax.plot(fpr, tpr, color="#7c6af7", lw=2, label=f"AUC = {auc:.3f}")
        else:
            # Multi-class: one-vs-rest
            n_classes = len(classes)
            y_bin = label_binarize(y_true_num, classes=list(range(n_classes)))
            colours = ["#7c6af7", "#63d4c4", "#a6e3a1", "#f9e2af", "#f38ba8"]
            for i, cls in enumerate(classes[:5]):
                if y_bin.shape[1] <= i or y_prob.shape[1] <= i:
                    continue
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
                ax.plot(fpr, tpr, color=colours[i % len(colours)], lw=1.5,
                        label=f"{cls} (AUC={auc:.2f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.2)
        self._roc_canvas.draw()

    def _draw_cv(self, result: TrainingResult) -> None:
        self._cv_canvas.clear()
        ax = self._cv_canvas.get_ax()

        scores = result.cv_scores
        if scores is None or len(scores) == 0:
            ax.text(0.5, 0.5, "No CV scores available.",
                    ha="center", va="center", transform=ax.transAxes)
            self._cv_canvas.draw()
            return

        folds = list(range(1, len(scores) + 1))
        ax.bar(folds, scores, color="#7c6af7", edgecolor="#1e1e2e", alpha=0.85)
        ax.axhline(scores.mean(), color="#63d4c4", linestyle="--", lw=1.5,
                   label=f"Mean: {scores.mean():.4f}")
        ax.fill_between(
            [0.5, len(scores) + 0.5],
            scores.mean() - scores.std(),
            scores.mean() + scores.std(),
            alpha=0.15, color="#63d4c4",
        )
        ax.set_xticks(folds)
        ax.set_xticklabels([f"Fold {f}" for f in folds])
        ax.set_ylabel("Score")
        ax.set_title(
            f"Cross-Validation Scores  (mean={scores.mean():.4f} ± {scores.std():.4f})"
        )
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        self._cv_canvas.draw()
