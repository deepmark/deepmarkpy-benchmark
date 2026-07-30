"""Plugin discovery for the deepmarkpy package.

Walks the packaged ``deepmarkpy/plugins`` tree (and, optionally, an external
directory of third-party plugins) and registers every ``BaseAttack``/
``BaseModel`` subclass found in files named exactly ``attack.py`` or
``model.py``, keyed by class name, with the containing directory's
``config.json`` attached. Import failures are logged and recorded in the
``failed`` mapping; the failing plugin is skipped.
"""

import importlib
import importlib.util
import inspect
import json
import logging
import os

from deepmarkpy.core.base_attack import BaseAttack
from deepmarkpy.core.base_model import BaseModel

logger = logging.getLogger(__name__)

# Environment variable naming a directory of external plugin directories.
EXTERNAL_PLUGINS_ENV = "DEEPMARK_PLUGINS_DIR"


class PluginManager:
    def __init__(self, plugins_dir=None, external_plugins_dir=None):
        """Discover attack and model plugins.

        Args:
            plugins_dir: Path to the packaged ``plugins`` directory. Defaults
                to the ``plugins`` directory next to this module.
            external_plugins_dir: Directory containing third-party plugin
                directories (each with ``attack.py``/``model.py`` +
                ``config.json``). Defaults to the ``DEEPMARK_PLUGINS_DIR``
                environment variable when set. External plugin files are
                loaded by file path, so they work outside site-packages.
        """
        if plugins_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.plugins_dir = os.path.join(current_dir, "plugins")
        else:
            self.plugins_dir = plugins_dir

        if external_plugins_dir is None:
            external_plugins_dir = os.environ.get(EXTERNAL_PLUGINS_ENV)
        self.external_plugins_dir = external_plugins_dir

        self.attacks = {}
        self.models = {}
        # Import failures by module path (or file path for external
        # plugins) -> error string. Programmatic record only.
        self.failed = {}

        self._load_attacks()
        self._load_models()
        if self.external_plugins_dir:
            self._load_external(self.external_plugins_dir)

    def _load_attacks(self):
        """
        Recursively load and register all classes inheriting from BaseAttack
        under the plugins/attacks/ directory.
        """
        attacks_path = os.path.join(self.plugins_dir, "attacks")
        self._load_classes_from_directory(
            directory=attacks_path,
            base_class=BaseAttack,
            storage_dict=self.attacks,
        )

    def _load_models(self):
        """
        Recursively load and register all classes inheriting from BaseModel
        under the plugins/models/ directory.
        """
        models_path = os.path.join(self.plugins_dir, "models")
        self._load_classes_from_directory(
            directory=models_path,
            base_class=BaseModel,
            storage_dict=self.models,
        )

    @staticmethod
    def _load_config(root):
        """Return the directory's parsed config.json, or None."""
        config_path = os.path.join(root, "config.json")
        config_data = None
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load config.json at {config_path} ({e})")
        return config_data

    def _register_members(self, module, base_class, storage_dict, config_data):
        """Register every ``base_class`` subclass found in ``module``."""
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, base_class) and obj is not base_class:
                storage_dict[name] = {"class": obj, "config": config_data}

    def _load_classes_from_directory(self, directory, base_class, storage_dict):
        # The import prefix derives from this module's package, so plugins
        # always load under one module identity (deepmarkpy.plugins.*).
        package_prefix = f"{__package__}.plugins"
        for root, _, files in os.walk(directory):
            config_data = self._load_config(root)

            for filename in files:
                # Only load if it's literally named attack.py or model.py
                if filename not in ["attack.py", "model.py"]:
                    continue  # skip everything else

                rel_path = os.path.relpath(
                    os.path.join(root, filename), self.plugins_dir
                )
                module_name = os.path.splitext(rel_path)[0].replace(os.path.sep, ".")
                full_module_path = f"{package_prefix}.{module_name}"

                try:
                    module = importlib.import_module(full_module_path)
                    self._register_members(module, base_class, storage_dict, config_data)
                except Exception as e:
                    logger.error(f"Failed to import {full_module_path}: {e}")
                    self.failed[full_module_path] = str(e)

    def _load_external(self, directory):
        """Load third-party plugins from ``directory`` by file path.

        Any ``attack.py``/``model.py`` under the directory is loaded under a
        synthetic ``deepmarkpy_external_plugins`` module name and registered
        exactly like a packaged plugin.
        """
        for root, _, files in os.walk(directory):
            config_data = self._load_config(root)

            for filename in files:
                if filename not in ["attack.py", "model.py"]:
                    continue
                base_class = BaseAttack if filename == "attack.py" else BaseModel
                storage_dict = self.attacks if filename == "attack.py" else self.models

                file_path = os.path.join(root, filename)
                rel = os.path.relpath(file_path, directory)
                module_name = "deepmarkpy_external_plugins." + (
                    os.path.splitext(rel)[0].replace(os.path.sep, ".")
                )

                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self._register_members(module, base_class, storage_dict, config_data)
                except Exception as e:
                    logger.error(f"Failed to import {file_path}: {e}")
                    self.failed[file_path] = str(e)

    def get_attacks(self):
        """Return a dict of {class_name: {"class": class, "config": config_data}} for all discovered attacks."""
        return self.attacks

    def get_models(self):
        """Return a dict of {class_name: {"class": class, "config": config_data}} for all discovered models."""
        return self.models
