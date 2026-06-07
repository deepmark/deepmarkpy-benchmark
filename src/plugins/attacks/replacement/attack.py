import numpy as np

from core.base_attack import BaseAttack
from plugins.attacks.replacement.replacement_attack import replacement_attack


class ReplacementAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform a replacement attack on an audio signal.

        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters for the replacement attack:
                - sampling_rate (int): The sampling rate of the audio signal in Hz (required).
                - block_size_replacement (int): Size of each block for processing in samples (default: 1024).
                - overlap_factor_replacement (float): Overlap factor between consecutive blocks (default: 0.75).
                Must be in the range [0, 1), where 0 means no overlap and values closer to 1
                indicate higher overlap.
                - lower_bound_replacement (float): The lower bound of the similarity distance for considering a block as a candidate (default: 0).
                - upper_bound_replacement (float): The upper bound of the similarity distance for considering a block as a candidate (default: 10).
                - use_masking_replacement (bool): Whether to use psychoacoustic masking for distance calculation (default: False).

        Returns:
            np.ndarray: The processed audio signal with the replacement attack applied.

        Raises:
            ValueError: If the `sampling_rate` is not provided in `kwargs`.

        """
        sampling_rate = kwargs.get("sampling_rate", None)
        block_size = kwargs.get(
            "block_size_replacement", self.config.get("block_size_replacement")
        )
        overlap_factor = kwargs.get(
            "overlap_factor_replacement", self.config.get("overlap_factor_replacement")
        )
        lower_bound = kwargs.get(
            "lower_bound_replacement", self.config.get("lower_bound_replacement")
        )
        upper_bound = kwargs.get(
            "upper_bound_replacement", self.config.get("upper_bound_replacement")
        )
        use_masking = kwargs.get(
            "use_masking_replacement", self.config.get("use_masking_replacement")
        )
        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")
        return replacement_attack(
            x=audio,
            sampling_rate=sampling_rate,
            block_size=block_size,
            overlap_factor=overlap_factor,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            use_masking=use_masking,
        )
