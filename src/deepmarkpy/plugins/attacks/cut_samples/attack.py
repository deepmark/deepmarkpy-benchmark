import numpy as np

from deepmarkpy.core.base_attack import BaseAttack


class CutSamplesAttack(BaseAttack):
    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform a "Cut Samples" attack by randomly deleting short sequences of samples
        within a randomly chosen segment of the audio.

        Args:
            audio (np.ndarray): Input audio signal.
            **kwargs: Additional parameters.
                - sampling_rate (int): Sampling rate of the audio in Hz (required).
                - max_sequence_length_cut (int): Maximum length of each cut sequence. Default is 50 samples (Optional).
                - num_sequences_cut (int): Number of sequences to cut in the specified duration. Default is 20 (Optional).
                - duration_cut (float): Duration (in seconds) over which cuts should occur. Default is 0.5 seconds (Optional).
                - max_value_difference_cut (float): Maximum allowed difference between start and end sample of a cut. Default is 0.1 (Optional).

        Returns:
            np.ndarray: Audio signal with random samples cut.

        Raises:
            ValueError: If 'sampling_rate' is not provided in kwargs.
        """
        sampling_rate = kwargs.get("sampling_rate", None)
        max_sequence_length = kwargs.get(
            "max_sequence_length", self.config.get("max_sequence_length")
        )
        num_sequences = kwargs.get(
            "num_sequences", self.config.get("num_sequences")
        )
        duration = kwargs.get("duration", self.config.get("duration"))
        max_value_difference = kwargs.get(
            "max_value_difference", self.config.get("max_value_difference")
        )

        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")

        total_samples = len(audio)
        segment_samples = int(duration * sampling_rate)

        if segment_samples >= total_samples:
            start_offset = 0
        else:
            start_offset = np.random.randint(0, total_samples - segment_samples)

        end_offset = start_offset + segment_samples
        segment_audio = audio[start_offset:end_offset]

        cut_positions = np.random.choice(
            range(len(segment_audio) - max_sequence_length),
            size=num_sequences,
            replace=False,
        )
        cut_positions = np.sort(cut_positions)

        modified_audio = []
        prev_cut_end = 0

        for start_pos in cut_positions:
            sequence_length = np.random.randint(1, max_sequence_length + 1)
            end_pos = start_pos + sequence_length

            if end_pos >= len(segment_audio):
                break

            if abs(segment_audio[start_pos] - segment_audio[end_pos]) > max_value_difference:
                continue

            modified_audio.extend(segment_audio[prev_cut_end:start_pos])
            prev_cut_end = end_pos

        modified_audio.extend(segment_audio[prev_cut_end:])

        final_audio = np.concatenate([audio[:start_offset], modified_audio, audio[end_offset:]])
        return final_audio
