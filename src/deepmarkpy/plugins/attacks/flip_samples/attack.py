import numpy as np

from deepmarkpy.core.base_attack import BaseAttack


class FlipSamplesAttack(BaseAttack):
    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform a "Flip Samples" attack by randomly exchanging the positions
        of selected samples in a randomly chosen segment of the audio signal.

        Args:
            audio (np.ndarray): Input audio signal.
            **kwargs: Additional parameters.
                - sampling_rate (int): Sampling rate of the audio in Hz (required).
                - num_flip_samples (int): Number of sample pairs to flip in the selected segment. Default is 100 (Optional).
                - duration_flip_samples (float): Duration (in seconds) of the segment where flips should occur. Default is 0.5 seconds (Optional).

        Returns:
            np.ndarray: Audio signal with flipped sample positions.

        Raises:
            ValueError: If 'sampling_rate' is not provided in kwargs.
        """
        sampling_rate = kwargs.get("sampling_rate", None)
        num_flips = kwargs.get("num_flips", self.config.get("num_flips"))
        duration = kwargs.get("duration", self.config.get("duration"))

        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")

        total_samples = int(duration * sampling_rate)

        if total_samples >= len(audio):
            start_offset = 0
            end_offset = len(audio)
        else:
            start_offset = np.random.randint(0, len(audio) - total_samples)
            end_offset = start_offset + total_samples

        segment = audio[start_offset:end_offset]

        flip_indices = np.random.choice(
            range(len(segment)), size=min(num_flips * 2, len(segment)), replace=False
        )
        flip_indices = flip_indices.reshape(-1, 2)

        modified_segment = segment.copy()
        for idx1, idx2 in flip_indices:
            modified_segment[idx1], modified_segment[idx2] = (
                modified_segment[idx2],
                modified_segment[idx1],
            )

        modified_audio = audio.copy()
        modified_audio[start_offset:end_offset] = modified_segment

        return modified_audio
