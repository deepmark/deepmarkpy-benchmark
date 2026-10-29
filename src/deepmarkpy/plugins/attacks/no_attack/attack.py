import numpy as np

from deepmarkpy.core.base_attack import BaseAttack


class NoAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Identity attack: returns the audio unchanged.

        The baseline row of a benchmark run: what the detector reads back from
        the watermarked audio with nothing in between, measured and reported
        like every other attack.

        Args:
            audio (np.ndarray): The input audio signal.

        Returns:
            np.ndarray: The same audio signal.
        """
        return audio
