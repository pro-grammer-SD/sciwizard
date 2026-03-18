"""Matplotlib canvas embedded in a Qt widget."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

matplotlib.use("QtAgg")

# Use dark matplotlib style to match the app theme
plt.rcParams.update(
    {
        "axes.facecolor": "#1e1e2e",
        "figure.facecolor": "#1e1e2e",
        "axes.edgecolor": "#45475a",
        "axes.labelcolor": "#cdd6f4",
        "xtick.color": "#a6adc8",
        "ytick.color": "#a6adc8",
        "text.color": "#cdd6f4",
        "grid.color": "#313244",
        "grid.linewidth": 0.5,
        "lines.color": "#7c6af7",
        "patch.edgecolor": "#45475a",
        "legend.facecolor": "#27273a",
        "legend.edgecolor": "#45475a",
    }
)


class PlotCanvas(QWidget):
    """A QWidget wrapping a Matplotlib Figure with a navigation toolbar.

    Args:
        parent: Optional parent widget.
        show_toolbar: Whether to include the matplotlib navigation toolbar.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        show_toolbar: bool = True,
        figsize: tuple[int, int] = (8, 5),
    ) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if show_toolbar:
            toolbar = NavToolbar(self.canvas, self)
            toolbar.setStyleSheet(
                "background-color: #27273a; border-bottom: 1px solid #313244;"
            )
            layout.addWidget(toolbar)

        layout.addWidget(self.canvas)

    def clear(self) -> None:
        """Clear all axes on the figure."""
        self.figure.clear()
        self.canvas.draw()

    def get_ax(self, nrows: int = 1, ncols: int = 1, index: int = 1):
        """Return (and create if needed) a subplot axes.

        Args:
            nrows: Number of subplot rows.
            ncols: Number of subplot columns.
            index: 1-based subplot index.

        Returns:
            Matplotlib Axes object.
        """
        return self.figure.add_subplot(nrows, ncols, index)

    def draw(self) -> None:
        """Refresh the canvas."""
        self.canvas.draw()
