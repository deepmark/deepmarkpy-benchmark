import abc
import inspect
import json
import logging
import os

import numpy as np

from deepmarkpy.utils.param_aliases import expand_attack_config, expand_attack_kwargs

logger = logging.getLogger(__name__)

class BaseAttack(abc.ABC):
    """
    Abstract base class for an Attack module.
    
    All attacks must implement the `apply` method.
    Each attack should have its own `config.json` stored in its respective folder.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        apply_method = cls.__dict__.get("apply")
        if apply_method is None or getattr(apply_method, "_deepmark_alias_wrapped", False):
            return

        def wrapped_apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
            return apply_method(
                self,
                audio,
                **expand_attack_kwargs(self.attack_key, kwargs),
            )

        wrapped_apply.__name__ = apply_method.__name__
        wrapped_apply.__doc__ = apply_method.__doc__
        wrapped_apply._deepmark_alias_wrapped = True
        cls.apply = wrapped_apply

    def __init__(self):
        """
        Initializes the attack by loading its configuration file.
        
        - Determines the file path of the subclass implementing this base class.
        - Constructs the path to `config.json` in the attack's directory.
        - Loads the configuration if the file exists, otherwise sets `_config` to None.
        """
        model_file = inspect.getfile(self.__class__)
        model_dir = os.path.dirname(os.path.abspath(model_file))
        self.attack_key = os.path.basename(model_dir)

        self.config_path = os.path.join(model_dir, "config.json")

        if not os.path.exists(self.config_path):
            logger.warning(f"config.json not found in {self.config_path}")
            self._config = None
        else:
            with open(self.config_path, "r") as json_file:
                self._config = expand_attack_config(self.attack_key, json.load(json_file))

    @abc.abstractmethod
    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Applies the attack to the given `audio` signal.

        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters that specific attacks may require.

        Returns:
            np.ndarray: The attacked (modified) audio signal.

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
