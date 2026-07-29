import numpy as np

from deepmarkpy.core.base_attack import BaseAttack
from deepmarkpy.plugins.attacks.replacement_2.replacement2_attack import replacement2_attack


class Replacement2Attack(BaseAttack):
    """Faster, scalable variant of :class:`ReplacementAttack`.

    Produces the same kind of replacement attack (blocks substituted by a
    least-squares combination of spectrally similar blocks) but uses a
    vectorised, BLAS-backed similarity search so runtime grows manageably with
    file length instead of blowing up in a pure-Python ``O(N^2)`` double loop.
    """

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform a fast replacement attack on an audio signal.

        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters for the replacement attack:
                - sampling_rate (int): Sampling rate of the audio in Hz (required).
                - block_size (int): Block size in samples.
                - overlap_factor (float): Overlap factor in [0, 1).
                - lower_bound (float): Lower similarity-distance bound.
                - upper_bound (float): Upper similarity-distance bound.
                - k (int): Max number of similar blocks used.
                - use_masking (bool): Use psychoacoustic masking.
                - search_window_sec (float): Restrict candidates to
                  +/- this many seconds (0 = whole file).
                - search_dims (int): Rank on the first N magnitude
                  bins only (0 = all bins).
                - tile_size (int): Query blocks per tile (memory vs.
                  speed; does not change the result).

        Returns:
            np.ndarray: The processed audio signal.

        Raises:
            ValueError: If ``sampling_rate`` is not provided in ``kwargs``.
        """
        sampling_rate = kwargs.get("sampling_rate", None)
        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")

        def cfg(name):
            return kwargs.get(name, self.config.get(name))

        return replacement2_attack(
            x=audio,
            sampling_rate=sampling_rate,
            block_size=cfg("block_size"),
            overlap_factor=cfg("overlap_factor"),
            lower_bound=cfg("lower_bound"),
            upper_bound=cfg("upper_bound"),
            k=cfg("k"),
            use_masking=cfg("use_masking"),
            search_window_sec=cfg("search_window_sec"),
            search_dims=cfg("search_dims"),
            tile_size=cfg("tile_size"),
        )
