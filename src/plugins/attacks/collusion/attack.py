import numpy as np
from core.base_attack import BaseAttack

class CollusionAttack(BaseAttack):
    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform a collusion attack by embedding a second watermark into the audio and
        recombining segments of both watermarked audios into one.
        Args:
            audio (np.ndarray): The original watermarked audio.
            **kwargs: Additional parameters for the attack.
                - model (BaseModel): BaseModel instance of the watermarking model.
                - orig_audio (np.ndarray): Original audio file.
                - sampling_rate (int): Sampling rate of the audio.
                - size_collusion(int): Size of the collusion segment.
        Returns:
            np.ndarray: The recombined audio after the collusion attack.
        """
        model = kwargs.get("model", None)
        orig_audio = kwargs.get("orig_audio", None)
        sampling_rate = kwargs.get("sampling_rate", None)
        
        if model is None or orig_audio is None or sampling_rate is None:
            raise ValueError(
                "'model', 'orig_audio' and 'sampling_rate' must be provided for the collusion attack."
            )
        
        second_audio = model.embed(orig_audio, model.generate_watermark(), sampling_rate)

        if second_audio is None or len(second_audio) == 0:
            raise ValueError(
                f"Collusion attack failed: model.embed() returned empty audio. "
                f"Check that the watermarking service is running and responding correctly."
            )

        segment_size = kwargs.get("size_collusion", 25)
        num_segments = len(audio) // segment_size
        mixed_audio = np.zeros_like(audio)
        second_len = len(second_audio)

        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size
            # Use second_audio only if the full segment is available
            if np.random.rand() > 0.5 or end_idx > second_len:
                mixed_audio[start_idx:end_idx] = audio[start_idx:end_idx]
            else:
                mixed_audio[start_idx:end_idx] = second_audio[start_idx:end_idx]
        
        # Handle remaining samples
        if num_segments * segment_size < len(audio):
            remaining_start = num_segments * segment_size
            remaining_length = len(audio) - remaining_start
            
            if np.random.rand() > 0.5:
                mixed_audio[remaining_start:] = audio[remaining_start:]
            else:
                # Only use available samples from second_audio
                available_from_second = min(remaining_length, len(second_audio) - remaining_start) #CHANGE
                if available_from_second > 0:
                    mixed_audio[remaining_start:remaining_start + available_from_second] = second_audio[remaining_start:remaining_start + available_from_second] #CHANGE
                    # If second_audio is shorter, fill the rest from original audio
                    if available_from_second < remaining_length:
                        mixed_audio[remaining_start + available_from_second:] = audio[remaining_start + available_from_second:]
                else:
                    # If remaining_start is beyond second_audio length, use original audio
                    mixed_audio[remaining_start:] = audio[remaining_start:]
        
        return mixed_audio