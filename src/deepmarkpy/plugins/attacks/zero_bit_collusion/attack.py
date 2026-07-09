import numpy as np

from deepmarkpy.core.base_attack import BaseAttack

class ZeroBitCollusionAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform a collusion attack on an audio signal. Modification of the collusion attack for zero-bit watermarking models.
        This attack takes x% of the original (not watermarked) audio and (100-x)% of the watermarked audio and concatenates them.
        Args:
            audio (np.ndarray): The input audio signal that's watermarked.
            **kwargs: Additional parameters for the collusion modification attack:
                - sampling_rate (int): The sampling rate of the audio signal in Hz (required).
                - original_audio (np.ndarray): The original audio signal, that's not watermarked.
                - x (int): percentage of the non_watermarked_audio.
                - position (string): possibilities are ['random_samples','random_segment','front','end']. This explains how parts of the watermarked signal are replaced by using the original signal.
                    - 'random_samples': replaces random individual samples
                    - 'random_segment': replaces a contiguous segment at a random position
                    - 'front': replaces samples from the beginning
                    - 'end': replaces samples from the end
        Returns:
            np.ndarray: The processed audio signal.

        Raises:
            ValueError: If the `sampling_rate` is not provided in `kwargs`.

        """

        sampling_rate = kwargs.get("sampling_rate", None)
        original_audio = kwargs.get("original_audio", None)
        # original_audio=original_audio.copy()
        x = kwargs.get(
            "x", self.config.get("x")
        )
        position = kwargs.get("position", self.config.get("position"))

        if position not in ["random_samples", "random_segment", "front", "end"]:
            raise ValueError(f"Invalid position: '{position}'. Must be one of 'random_samples', 'random_segment', 'front', or 'end'.")
       
        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")
        
        if original_audio is None:
            raise ValueError("'original_audio' must be provided in kwargs.")

        original_audio = original_audio.copy()

        wm_len = len(audio)
        orig_len = len(original_audio)
        min_len = min(wm_len, orig_len)

        num_samples = int(min_len * x / 100)

        if num_samples <= 0:
            return audio

        reconstructed_audio = audio.copy()
    
        if (position=="front"):
           
            audio_first_part = original_audio[:num_samples]
            audio_second_part = audio[num_samples:]
            reconstructed_audio = np.concatenate((audio_first_part, audio_second_part), axis=0)
           
        elif (position=="end"):
            
            audio_first_part = audio[:wm_len - num_samples]
            audio_second_part = original_audio[-num_samples:]
            reconstructed_audio = np.concatenate((audio_first_part, audio_second_part), axis=0)
            
        elif (position=="random_samples"):
            replace_indices = np.random.choice(min_len, size=num_samples, replace=False)
            reconstructed_audio[replace_indices] = original_audio[replace_indices]

        elif (position=="random_segment"):
            # Replace a contiguous segment at a random position
            max_start = min_len - num_samples
            start_index = np.random.randint(0, max_start + 1)
            end_index = start_index + num_samples
            reconstructed_audio[start_index:end_index] = original_audio[start_index:end_index]

        return reconstructed_audio
