import abc
import inspect
import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

class BaseAttack(abc.ABC):
    """
    Abstract base class for an Attack module.
    
    All attacks must implement the `apply` method.
    Each attack should have its own `config.json` stored in its respective folder.
    """

    def __init__(self, version=None):
        """
        Initializes the attack by loading its configuration file.

        Args:
            version: Optional version name to load from a multi-version
                config.json. When None, loads 'default'. For single-version
                configs (no 'default' key), the entire config is used.
        """
        model_file = inspect.getfile(self.__class__)
        model_dir = os.path.dirname(os.path.abspath(model_file))

        self.config_path = os.path.join(model_dir, "config.json")
        self._version = version or "default"

        if not os.path.exists(self.config_path):
            logger.warning(f"config.json not found in {self.config_path}")
            self._config = None
        else:
            with open(self.config_path, "r") as json_file:
                raw = json.load(json_file)

            if "default" in raw and isinstance(raw["default"], dict):
                # Multi-version config
                if self._version not in raw:
                    available = [k for k in raw if not k.startswith("_")]
                    raise ValueError(
                        f"{self.__class__.__name__} has no version '{self._version}'. "
                        f"Available versions: {available}"
                    )
                self._config = raw[self._version]
            else:
                # Single-version config: the whole file is the parameter set.
                self._config = raw

    @abc.abstractmethod
    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Applies the attack to the given `audio` signal.

        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters that specific attacks may require.

        Returns:
            np.ndarray: The attacked (modified) audio signal.

        An attack may instead return ``(audio, extra)`` only if its class name
        is listed in ``benchmark._TUPLE_RETURNING_ATTACKS``; the run loop
        unpacks those and nothing else. ``CrossModelAttack`` is the only one
        today, returning the watermark it embedded alongside the audio.

        This method must be implemented by all subclasses.
        """
        pass

    @property
    def name(self) -> str:
        """
        Returns a short identifier name for this attack.

        Returns:
            str: The class name of the attack instance.
        """
        return self.__class__.__name__

    @property
    def config(self) -> dict:
        """
        Provides read-only access to the attack configuration.

        Returns:
            dict: The attack's configuration loaded from `config.json`,
                  or None if the file does not exist.
        """
        return self._config