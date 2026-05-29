import logging
import math
import numpy as np
import torch

from core.base_attack import BaseAttack

logger = logging.getLogger(__name__)


class DescriptAudioCodecAttack(BaseAttack):
    def __init__(self):
        super().__init__()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.codebook_size = None
        self.downsampling_ratio = None
        self.n_codebooks = None
        self.supported_n_codebooks = None
        self.supported_bandwidths = None
        self.bandwith_to_ncodebook = None

    def _load_model(self):
        """Lazy load the DAC model and calculate bandwidth properties."""
        if self.model is not None:
            return

        try:
            import dac
        except ImportError:
            raise RuntimeError(
                "dac not found. Please install it: pip install descript-audio-codec"
            )

        model_type = self.config.get("model_type_dac", "44khz")

        # Get codec sample rate
        type_to_sr = {'44khz': 44100, '24khz': 24000, '16khz': 16000}
        self.codec_sr = type_to_sr[model_type]

        # Load pre-trained DAC model, options: "16khz", "24khz", "44khz"
        logger.info(f"Downloading DAC model: {model_type} (this may take a few minutes on first run)...")

        model_path = dac.utils.download(model_type=model_type)

        logger.info(f"Download complete. Loading model from {model_path}")

        self.model = dac.DAC.load(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Calculate codebook properties
        self.codebook_size = self.model.codebook_size
        self.downsampling_ratio = math.prod(
            [block.block[-1].stride[0]
             for block in self.model.encoder.block
             if 'EncoderBlock' in str(block.__class__)]
        )

        # Set supported number of codebooks and bandwidths
        self.n_codebooks = self.model.n_codebooks
        self.supported_n_codebooks = [i + 1 for i in range(self.model.n_codebooks)]
        self.supported_bandwidths = [
            self.codec_sr / self.downsampling_ratio * math.log2(self.codebook_size) * i
            for i in self.supported_n_codebooks
        ]

        # Map bandwidth to the number of codebooks
        self.bandwith_to_ncodebook = {
            bandwidth: n_codebook
            for bandwidth, n_codebook in zip(self.supported_bandwidths, self.supported_n_codebooks)
        }

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply Descript Audio Codec (DAC) neural codec compression attack.

        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters:
                - sampling_rate (int): The sampling rate of the audio signal in Hz (required).
                - n_codebooks (int): Number of codebooks to use (optional, from config if not provided).

        Returns:
            np.ndarray: The processed audio signal after DAC compression.
        """
        sampling_rate = kwargs.get("sampling_rate", 16000)

        # Load model on first use
        self._load_model()

        # Get number of codebooks to use
        n_codebooks = kwargs.get("n_codebooks_dac", self.config.get("n_codebooks_dac"))
        if n_codebooks is None:
            # Use maximum codebooks by default
            n_codebooks = self.n_codebooks

        if n_codebooks not in self.supported_n_codebooks:
            logger.warning(f"n_codebooks={n_codebooks} not in supported range {self.supported_n_codebooks}, using {self.n_codebooks}")
            n_codebooks = self.n_codebooks

        # Convert numpy to torch tensor [batch, channels, time]
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        waveform = waveform.to(self.device)

        # Resample if needed (DAC expects 16kHz, 24kHz, or 44.1kHz depending on model)
        target_sr = self.config.get("target_sampling_rate_dac", 44100)
        if sampling_rate != target_sr:
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=sampling_rate,
                new_freq=target_sr
            ).to(self.device)
            waveform = resampler(waveform)

        # Apply DAC compression and decompression
        try:
            with torch.no_grad():
                original_length = waveform.shape[-1]
                reconstructed = self.model(waveform, n_quantizers=n_codebooks)['audio']
                reconstructed = reconstructed[..., :original_length]
        except Exception as e:
            logger.error(f"DAC encoding/decoding failed: {e}")
            raise

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
