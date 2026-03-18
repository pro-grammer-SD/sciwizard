"""Dynamic plugin loader — discovers and loads Python modules from /plugins."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

from sciwizard.config import PLUGIN_DIR

logger = logging.getLogger(__name__)


class PluginLoader:
    """Scan the plugins directory and load every valid plugin module.

    A valid plugin module must expose a ``register(registry: dict) -> None``
    function. The registry dict has two keys:

    * ``"models"``  — dict[str, sklearn-compatible estimator]
    * ``"preprocessors"`` — dict[str, sklearn Transformer]

    Example plugin::

        # plugins/my_model.py
        from sklearn.ensemble import ExtraTreesClassifier

        def register(registry):
            registry["models"]["Extra Trees"] = ExtraTreesClassifier()
    """

    def __init__(self, plugin_dir: Path | None = None) -> None:
        self._plugin_dir = plugin_dir or PLUGIN_DIR
        self._loaded: list[str] = []

    def load_all(self, registry: dict) -> None:
        """Load all plugins into the provided registry.

        Args:
            registry: Mutable dict with keys ``"models"`` and ``"preprocessors"``.
        """
        if not self._plugin_dir.exists():
            logger.debug("Plugin directory %s does not exist — skipping", self._plugin_dir)
            return

        for path in sorted(self._plugin_dir.glob("*.py")):
            if path.name.startswith("_"):
                continue
            try:
                self._load_plugin(path, registry)
            except Exception as exc:
                logger.warning("Failed to load plugin %s: %s", path.name, exc)

    def _load_plugin(self, path: Path, registry: dict) -> None:
        module_name = f"sciwizard.plugin.{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        if not hasattr(module, "register"):
            logger.warning("Plugin %s has no register() — skipping", path.name)
            return

        module.register(registry)
        self._loaded.append(path.name)
        logger.info("Plugin loaded: %s", path.name)

    @property
    def loaded_plugins(self) -> list[str]:
        """Names of successfully loaded plugin files."""
        return list(self._loaded)
