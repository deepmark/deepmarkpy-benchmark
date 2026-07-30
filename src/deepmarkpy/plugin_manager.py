import importlib
import inspect
import json
import logging
import os
import sys

from deepmarkpy.core.base_attack import BaseAttack
from deepmarkpy.core.base_model import BaseModel

logger = logging.getLogger(__name__)

class PluginManager:
    def __init__(self, plugins_dir=None):
        """
        :param plugins_dir: Path to the 'plugins' directory.
                           If None, we assume it is in the same directory
                           as this PluginManager or we dynamically compute it.
        """
        if plugins_dir is None:
            # Dynamically locate the "plugins" directory relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.plugins_dir = os.path.join(current_dir, "plugins")
        else:
            self.plugins_dir = plugins_dir

        self.attacks = {}
        self.models = {}

        project_root = os.path.dirname(os.path.abspath(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        self._load_attacks()
        self._load_models()

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
            package_prefix="plugins",
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
            package_prefix="plugins",
        )

    def _load_classes_from_directory(
        self, directory, base_class, storage_dict, package_prefix
    ):
        for root, _, files in os.walk(directory):
            config_path = os.path.join(root, "config.json")
            config_data = None
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config_data = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load config.json at {config_path} ({e})")

            for filename in files:
                # Only load if it's literally named attack.py or model.py
                if filename not in ["attack.py", "model.py"]:
                    continue  # skip everything else

                # Build module path, e.g. "plugins.attacks.some_attack.attack"
                rel_path = os.path.relpath(
                    os.path.join(root, filename), self.plugins_dir
                )
                module_name = os.path.splitext(rel_path)[0].replace(os.path.sep, ".")
                full_module_path = f"{package_prefix}.{module_name}"

                try:
                    module = importlib.import_module(full_module_path)

                    # Inspect all classes defined in the module
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, base_class) and obj is not base_class:
                            storage_dict[name] = {"class": obj, "config": config_data}

                except Exception as e:
                    logger.error(f"Failed to import {full_module_path}: {e}")

    def get_attacks(self):
        """Return a dict of {class_name: {"class": class, "config": config_data}} for all discovered attacks."""
        return self.attacks

    def get_models(self):
        """Return a dict of {class_name: {"class": class, "config": config_data}} for all discovered models."""
        return self.models
