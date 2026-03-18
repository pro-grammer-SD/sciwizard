"""Main application window — sidebar + stacked panel layout."""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from sciwizard.config import (
    APP_NAME,
    APP_VERSION,
    ICON_PATH,
    SIDEBAR_WIDTH,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
)
from sciwizard.core.data_manager import DataManager
from sciwizard.core.experiment_tracker import ExperimentTracker
from sciwizard.core.model_registry import ModelRegistry
from sciwizard.core.model_trainer import TrainingResult
from sciwizard.core.plugin_loader import PluginLoader

logger = logging.getLogger(__name__)

_NAV_ITEMS = [
    ("📊", "Data"),
    ("🔧", "Preprocess"),
    ("🎨", "Visualize"),
    ("🚀", "Train"),
    ("🔍", "Hyperparams"),
    ("⚡", "AutoML"),
    ("📈", "Evaluate"),
    ("🔮", "Predict"),
    ("📦", "Registry"),
    ("🧪", "Experiments"),
]


class _SidebarButton(QPushButton):
    def __init__(self, icon: str, label: str, parent=None) -> None:
        super().__init__(f"  {icon}  {label}", parent)
        self.setObjectName("sidebar_btn")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(44)
        self.setFont(QFont("Segoe UI", 12))


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME}  v{APP_VERSION}")
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))

        self._dm = DataManager()
        self._registry = ModelRegistry()
        self._tracker = ExperimentTracker()

        self._plugin_registry: dict = {"models": {}, "preprocessors": {}}
        loader = PluginLoader()
        loader.load_all(self._plugin_registry)
        if loader.loaded_plugins:
            logger.info("Loaded plugins: %s", loader.loaded_plugins)

        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        from sciwizard.ui.panels.automl_panel import AutoMLPanel
        from sciwizard.ui.panels.data_panel import DataPanel
        from sciwizard.ui.panels.eval_panel import EvaluationPanel
        from sciwizard.ui.panels.experiments_panel import ExperimentsPanel
        from sciwizard.ui.panels.hyperparam_panel import HyperparamPanel
        from sciwizard.ui.panels.prediction_panel import PredictionPanel
        from sciwizard.ui.panels.preprocessing_panel import PreprocessingPanel
        from sciwizard.ui.panels.registry_panel import RegistryPanel
        from sciwizard.ui.panels.training_panel import TrainingPanel
        from sciwizard.ui.panels.viz_panel import VisualizationPanel

        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Sidebar
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(SIDEBAR_WIDTH)
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(10, 16, 10, 16)
        sl.setSpacing(4)

        logo = QLabel(f"🧙‍♂️  {APP_NAME}")
        logo.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        logo.setStyleSheet("color: #7c6af7; padding: 4px 8px 12px 8px;")
        sl.addWidget(logo)

        ver = QLabel(f"v{APP_VERSION}")
        ver.setStyleSheet("font-size: 11px; color: #585b70; padding-left: 10px;")
        sl.addWidget(ver)
        sl.addSpacing(8)

        self._nav_buttons: list[_SidebarButton] = []
        for icon, label in _NAV_ITEMS:
            btn = _SidebarButton(icon, label)
            btn.clicked.connect(self._make_nav(len(self._nav_buttons)))
            sl.addWidget(btn)
            self._nav_buttons.append(btn)

        sl.addStretch()
        self._beginner_btn = QPushButton("💡  Beginner Mode: OFF")
        self._beginner_btn.setObjectName("sidebar_btn")
        self._beginner_btn.setCheckable(True)
        self._beginner_btn.clicked.connect(self._toggle_beginner)
        sl.addWidget(self._beginner_btn)
        outer.addWidget(sidebar)

        # Content stack — must match _NAV_ITEMS order
        self._stack = QStackedWidget()
        outer.addWidget(self._stack, stretch=1)

        self._data_panel        = DataPanel(self._dm)
        self._preprocess_panel  = PreprocessingPanel(self._dm)
        self._viz_panel         = VisualizationPanel(self._dm)
        self._training_panel    = TrainingPanel(
            self._dm, self._registry, self._tracker,
            plugin_models=self._plugin_registry["models"],
        )
        self._hyperparam_panel  = HyperparamPanel(self._dm)
        self._automl_panel      = AutoMLPanel(self._dm)
        self._eval_panel        = EvaluationPanel()
        self._predict_panel     = PredictionPanel(self._dm)
        self._registry_panel    = RegistryPanel(self._registry)
        self._experiments_panel = ExperimentsPanel(self._tracker)

        for panel in (
            self._data_panel, self._preprocess_panel, self._viz_panel,
            self._training_panel, self._hyperparam_panel, self._automl_panel,
            self._eval_panel, self._predict_panel, self._registry_panel,
            self._experiments_panel,
        ):
            self._stack.addWidget(panel)

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready — load a CSV to begin")

        self._nav_buttons[0].setChecked(True)
        self._stack.setCurrentIndex(0)

    def _connect_signals(self) -> None:
        self._data_panel.data_loaded.connect(self._on_data_loaded)
        self._training_panel.training_finished.connect(self._on_training_done)
        self._registry_panel.model_loaded.connect(self._on_registry_model_loaded)

    def _make_nav(self, idx: int):
        def _handler():
            for i, b in enumerate(self._nav_buttons):
                b.setChecked(i == idx)
            self._stack.setCurrentIndex(idx)
        return _handler

    def _on_data_loaded(self, dm: DataManager) -> None:
        shape = dm.data.shape
        self._status_bar.showMessage(
            f"Loaded: {dm.file_name}  —  {shape[0]:,} rows × {shape[1]} cols"
        )
        self._viz_panel.refresh_columns()
        self._preprocess_panel.refresh_columns()

    def _on_training_done(self, result: TrainingResult) -> None:
        import logging
        _log = logging.getLogger(__name__)
        # Each panel update is guarded independently — a rendering failure in one
        # must never prevent the others from loading (e.g. predict panel).
        for label, fn in [
            ("eval_panel",        lambda: self._eval_panel.display_result(result)),
            ("predict_panel",     lambda: self._predict_panel.load_result(result)),
            ("registry_panel",    lambda: self._registry_panel.refresh()),
            ("experiments_panel", lambda: self._experiments_panel.refresh()),
        ]:
            try:
                fn()
            except Exception as exc:
                _log.exception("Post-training update failed in %s: %s", label, exc)

        metric_str = "  |  ".join(f"{k}: {v}" for k, v in result.metrics.items())
        self._status_bar.showMessage(f"✅  {result.model_name} trained  —  {metric_str}")

    def _on_registry_model_loaded(self, pipeline, meta: dict) -> None:
        self._status_bar.showMessage(
            f"Registry model loaded: {meta.get('alias')}  ({meta.get('model_id')})"
        )

    def _toggle_beginner(self, checked: bool) -> None:
        label = "ON" if checked else "OFF"
        self._beginner_btn.setText(f"💡  Beginner Mode: {label}")
        self._status_bar.showMessage(
            "Beginner mode enabled — tooltips active" if checked else "Beginner mode disabled"
        )
