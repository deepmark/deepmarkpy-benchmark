import numpy as np
import torch

from core.base_attack import BaseAttack


class EncodecAttack(BaseAttack):
    def __init__(self):
        super().__init__()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self):
        """Lazy load the Encodec model."""
        if self.model is not None:
            return

        try:
            from encodec import EncodecModel
        except ImportError:
            raise RuntimeError(
                "encodec not found. Please install it: pip install encodec"
            )

        model_name = self.config.get("model_name_encodec", "encodec_24khz")
        bandwidth = self.config.get("bandwidth_encodec", 6.0)

        # Load pre-trained Encodec model, options: "encodec_24khz" or "encodec_48khz"
        self.model = EncodecModel.encodec_model_24khz() if "24khz" in model_name else EncodecModel.encodec_model_48khz()
        self.model.set_target_bandwidth(bandwidth)
        self.model = self.model.to(self.device)
        self.model.eval()

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply Encodec neural codec compression attack.

        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters:
                - sampling_rate (int): The sampling rate of the audio signal in Hz (required).

        Returns:
            np.ndarray: The processed audio signal after Encodec compression.
        """
        sampling_rate = kwargs.get("sampling_rate", 16000)

        # Load model on first use
        self._load_model()

        # Convert numpy to torch tensor [batch, channels, time]
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        waveform = waveform.to(self.device)

        # Resample if needed (Encodec expects 24kHz or 48kHz)
        target_sr = self.config.get("target_sampling_rate_encodec", 24000)
        if sampling_rate != target_sr:
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=sampling_rate,
                new_freq=target_sr
            ).to(self.device)
            waveform = resampler(waveform)

        # Apply Encodec compression and decompression
        with torch.no_grad():
            # Encode to discrete codes and decode back to audio
            encoded_frames = self.model.encode(waveform)
            reconstructed = self.model.decode(encoded_frames)

        # Resample back to original sampling rate if needed
        if sampling_rate != target_sr:
            resampler_back = torchaudio.transforms.Resample(
                orig_freq=target_sr,
                new_freq=sampling_rate
            ).to(self.device)
            reconstructed = resampler_back(reconstructed)

        # Convert back to numpy
        result = reconstructed.squeeze().cpu().numpy()

        return result
