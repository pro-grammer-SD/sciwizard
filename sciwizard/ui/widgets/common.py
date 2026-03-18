"""Shared reusable widgets used across multiple panels."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class SectionHeader(QLabel):
    """A styled section header label."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("section_header")
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)


class MutedLabel(QLabel):
    """A subdued secondary label."""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("muted")
        self.setWordWrap(True)


class Divider(QFrame):
    """A horizontal rule divider."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)
        self.setStyleSheet("color: #313244; margin: 4px 0;")


class PrimaryButton(QPushButton):
    """A prominently styled action button."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("primary_btn")
        self.setCursor(Qt.CursorShape.PointingHandCursor)


class MetricCard(QFrame):
    """A small card displaying a metric name and value.

    Args:
        name: Metric label.
        value: Value to display (will be converted to str).
        color: Accent hex colour for the value text.
    """

    def __init__(
        self,
        name: str,
        value: float | str,
        color: str = "#7c6af7",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "QFrame { background-color: #27273a; border: 1px solid #313244;"
            " border-radius: 8px; padding: 10px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        val_label = QLabel(str(value))
        val_label.setStyleSheet(
            f"font-size: 22px; font-weight: 700; color: {color}; border: none;"
        )
        val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        name_label = QLabel(name)
        name_label.setStyleSheet("color: #a6adc8; font-size: 11px; border: none;")
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(val_label)
        layout.addWidget(name_label)

    @staticmethod
    def row(
        metrics: dict[str, float], parent: QWidget | None = None
    ) -> QWidget:
        """Build a horizontal row of MetricCards from a metrics dict."""
        COLOURS = ["#7c6af7", "#63d4c4", "#a6e3a1", "#f9e2af", "#f38ba8"]
        container = QWidget(parent)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        for i, (name, val) in enumerate(metrics.items()):
            display = f"{val:.4f}" if isinstance(val, float) else str(val)
            layout.addWidget(MetricCard(name, display, COLOURS[i % len(COLOURS)]))
        layout.addStretch()
        return container


class StatusBadge(QLabel):
    """Inline coloured badge for status indicators."""

    _COLOURS = {
        "success": ("#a6e3a1", "#1e3329"),
        "error": ("#f38ba8", "#3b1e27"),
        "warning": ("#f9e2af", "#3b3319"),
        "info": ("#89b4fa", "#1e2a3b"),
    }

    def __init__(
        self,
        text: str,
        status: str = "info",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(text, parent)
        fg, bg = self._COLOURS.get(status, self._COLOURS["info"])
        self.setStyleSheet(
            f"background-color: {bg}; color: {fg}; border-radius: 4px;"
            f" padding: 2px 8px; font-size: 11px; font-weight: 600;"
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
