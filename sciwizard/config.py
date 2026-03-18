"""Application-wide configuration and constants."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR: Path = Path(__file__).parent.parent
ICON_PATH: Path = ROOT_DIR / "icon" / "icon.ico"
PLUGIN_DIR: Path = ROOT_DIR / "plugins"
MODEL_REGISTRY_DIR: Path = Path.home() / ".sciwizard" / "models"
EXPERIMENT_LOG_PATH: Path = Path.home() / ".sciwizard" / "experiments.jsonl"

# Ensure user dirs exist
MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# App identity
# ---------------------------------------------------------------------------

APP_NAME = "SciWizard"
APP_ID = "SciWizard.App"
APP_VERSION = "1.0.0"
ORGANIZATION = "SciWizard"

# ---------------------------------------------------------------------------
# UI defaults
# ---------------------------------------------------------------------------

WINDOW_MIN_WIDTH = 1280
WINDOW_MIN_HEIGHT = 800
SIDEBAR_WIDTH = 220

DARK_PALETTE = {
    "background": "#1e1e2e",
    "surface": "#27273a",
    "surface_alt": "#313145",
    "primary": "#7c6af7",
    "primary_hover": "#9b8df8",
    "accent": "#63d4c4",
    "text": "#cdd6f4",
    "text_muted": "#6c7086",
    "border": "#45475a",
    "error": "#f38ba8",
    "warning": "#f9e2af",
    "success": "#a6e3a1",
}

# ---------------------------------------------------------------------------
# ML defaults
# ---------------------------------------------------------------------------

DEFAULT_TEST_SIZE: float = 0.2
DEFAULT_RANDOM_STATE: int = 42
MAX_AUTOML_MODELS: int = 8
CV_FOLDS: int = 5

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

BEGINNER_MODE_DEFAULT: bool = False
ENABLE_ANIMATIONS: bool = True
