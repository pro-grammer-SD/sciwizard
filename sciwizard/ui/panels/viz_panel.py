"""Visualization panel — histograms, scatter, heatmap, distributions, PCA."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from sciwizard.core.data_manager import DataManager
from sciwizard.ui.widgets.common import Divider, PrimaryButton, SectionHeader
from sciwizard.ui.widgets.plot_canvas import PlotCanvas

logger = logging.getLogger(__name__)


class VisualizationPanel(QWidget):
    """Interactive data visualization panel."""

    def __init__(self, data_manager: DataManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dm = data_manager
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)

        root.addWidget(SectionHeader("Visualization"))
        root.addWidget(Divider())

        # Controls row
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(10)

        ctrl_row.addWidget(QLabel("Plot type:"))
        self._plot_combo = QComboBox()
        self._plot_combo.addItems(
            ["Histogram", "Scatter", "Correlation Heatmap", "Feature Distribution", "PCA (2D)"]
        )
        self._plot_combo.currentIndexChanged.connect(self._update_col_visibility)
        ctrl_row.addWidget(self._plot_combo)

        ctrl_row.addWidget(QLabel("X column:"))
        self._x_combo = QComboBox()
        self._x_combo.setMinimumWidth(140)
        ctrl_row.addWidget(self._x_combo)

        self._y_label = QLabel("Y column:")
        ctrl_row.addWidget(self._y_label)
        self._y_combo = QComboBox()
        self._y_combo.setMinimumWidth(140)
        ctrl_row.addWidget(self._y_combo)

        plot_btn = PrimaryButton("Plot")
        plot_btn.clicked.connect(self._plot)
        ctrl_row.addWidget(plot_btn)
        ctrl_row.addStretch()
        root.addLayout(ctrl_row)

        self._canvas = PlotCanvas(self, show_toolbar=True)
        root.addWidget(self._canvas, stretch=1)

    def refresh_columns(self) -> None:
        """Sync column selectors from the data manager."""
        cols = self._dm.numeric_columns
        for combo in (self._x_combo, self._y_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(cols)
            combo.blockSignals(False)
        if len(cols) >= 2:
            self._y_combo.setCurrentIndex(1)

    def _update_col_visibility(self) -> None:
        plot_type = self._plot_combo.currentText()
        scatter_only = plot_type == "Scatter"
        self._y_label.setVisible(scatter_only)
        self._y_combo.setVisible(scatter_only)
        x_visible = plot_type not in ("Correlation Heatmap", "Feature Distribution", "PCA (2D)")
        self._x_combo.setVisible(x_visible)

    def _plot(self) -> None:
        if not self._dm.is_loaded:
            QMessageBox.warning(self, "No Data", "Please load a dataset first.")
            return

        df = self._dm.data
        numeric_df = df[self._dm.numeric_columns] if self._dm.numeric_columns else df

        plot_type = self._plot_combo.currentText()
        x_col = self._x_combo.currentText()
        y_col = self._y_combo.currentText()

        self._canvas.clear()

        try:
            if plot_type == "Histogram":
                self._histogram(numeric_df, x_col)
            elif plot_type == "Scatter":
                self._scatter(numeric_df, x_col, y_col)
            elif plot_type == "Correlation Heatmap":
                self._heatmap(numeric_df)
            elif plot_type == "Feature Distribution":
                self._distribution(numeric_df)
            elif plot_type == "PCA (2D)":
                self._pca(numeric_df)
        except Exception as exc:
            logger.exception("Plot failed")
            QMessageBox.critical(self, "Plot Error", str(exc))

    # ------------------------------------------------------------------
    # Plot implementations
    # ------------------------------------------------------------------

    def _histogram(self, df: pd.DataFrame, col: str) -> None:
        ax = self._canvas.get_ax()
        if col not in df.columns:
            col = df.columns[0]
        ax.hist(df[col].dropna(), bins=30, color="#7c6af7", edgecolor="#1e1e2e", alpha=0.85)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.set_title(f"Histogram — {col}")
        ax.grid(True, axis="y", alpha=0.3)
        self._canvas.draw()

    def _scatter(self, df: pd.DataFrame, x_col: str, y_col: str) -> None:
        ax = self._canvas.get_ax()
        x = df[x_col].dropna()
        y = df[y_col].dropna()
        min_len = min(len(x), len(y))
        ax.scatter(x.iloc[:min_len], y.iloc[:min_len], alpha=0.6, color="#7c6af7", s=20)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"Scatter — {x_col} vs {y_col}")
        ax.grid(True, alpha=0.3)
        self._canvas.draw()

    def _heatmap(self, df: pd.DataFrame) -> None:

        corr = df.corr(numeric_only=True)
        ax = self._canvas.get_ax()
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(corr.columns, fontsize=9)
        self._canvas.figure.colorbar(im, ax=ax)
        ax.set_title("Correlation Heatmap")
        self._canvas.draw()

    def _distribution(self, df: pd.DataFrame) -> None:
        cols = list(df.columns[:9])  # cap at 9
        n = len(cols)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        for i, col in enumerate(cols):
            ax = self._canvas.figure.add_subplot(nrows, ncols, i + 1)
            ax.hist(df[col].dropna(), bins=20, color="#63d4c4", edgecolor="#1e1e2e", alpha=0.8)
            ax.set_title(col, fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)

        self._canvas.figure.suptitle("Feature Distributions", y=1.02)
        self._canvas.draw()

    def _pca(self, df: pd.DataFrame) -> None:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        X = df.dropna()
        if X.shape[1] < 2:
            raise ValueError("Need at least 2 numeric columns for PCA")

        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)

        ax = self._canvas.get_ax()

        target = self._dm.target_column
        if target and target in self._dm.data.columns:
            labels = self._dm.data[target].iloc[X.index].values
            unique = np.unique(labels)
            colours = ["#7c6af7", "#63d4c4", "#a6e3a1", "#f9e2af", "#f38ba8"]
            for i, lbl in enumerate(unique[:len(colours)]):
                mask = labels == lbl
                ax.scatter(
                    components[mask, 0],
                    components[mask, 1],
                    label=str(lbl),
                    alpha=0.7,
                    s=20,
                    color=colours[i],
                )
            ax.legend(title=target, markerscale=2)
        else:
            ax.scatter(components[:, 0], components[:, 1], alpha=0.6, color="#7c6af7", s=20)

        var = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({var[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({var[1]:.1%} variance)")
        ax.set_title("PCA — 2D Projection")
        ax.grid(True, alpha=0.3)
        self._canvas.draw()
