import numpy as np

from deepmarkpy.core.base_attack import BaseAttack

class SignInversionAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        This attacks performs a simple sign inversion attack, that's inaudible to human ears
        (we detect amplitude with our ears, not the phase). 
        Args:
            audio (np.ndarray): The input audio signal.
        Returns:
            np.ndarray: The processed audio signal.

        """
        return -audio