"""Descript Audio Codec (DAC) compression attack inference, HTTP-free.

Audio is resampled to the codec rate from the plugin config, passed through
an eval-mode DAC round trip quantized to the requested number of codebooks,
and resampled back to the request rate.
"""

import logging
import math

import numpy as np
import torch
import dac

from deepmarkpy.core.inference import BaseAttackEngine

logger = logging.getLogger(__name__)


class DescriptAudioCodecEngine(BaseAttackEngine):
    """DAC encode/decode round-trip with a selectable codebook count.

    The model variant from ``config['model_type_dac']`` downloads and loads
    once at construction, together with the derived codebook/bandwidth
    properties used to validate requests.
    """

    def __init__(self, config: dict, device: str | None = None):
        """Download and load the configured DAC model onto ``device``."""
        self.config = config
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        self.device = device

        model_type = config.get("model_type_dac", "44khz")

        type_to_sr = {'44khz': 44100, '24khz': 24000, '16khz': 16000}
        codec_sr = type_to_sr[model_type]

        logger.info(f"Downloading DAC model: {model_type} (this may take a few minutes on first run)...")

        model_path = dac.utils.download(model_type=model_type)

        logger.info(f"Download complete. Loading model from {model_path}")

        model = dac.DAC.load(model_path)
        model = model.to(self.device)
        model.eval()

        codebook_size = model.codebook_size
        downsampling_ratio = math.prod(
            [block.block[-1].stride[0]
             for block in model.encoder.block
             if 'EncoderBlock' in str(block.__class__)]
        )

        n_codebooks = model.n_codebooks
        supported_n_codebooks = [i + 1 for i in range(model.n_codebooks)]
        supported_bandwidths = [
            codec_sr / downsampling_ratio * math.log2(codebook_size) * i
            for i in supported_n_codebooks
        ]

        self.model = model
        self.codec_sr = codec_sr
        self.codebook_size = codebook_size
        self.downsampling_ratio = downsampling_ratio
        self.n_codebooks = n_codebooks
        self.supported_n_codebooks = supported_n_codebooks
        self.supported_bandwidths = supported_bandwidths
        self.bandwith_to_ncodebook = {
            bandwidth: n_codebook
            for bandwidth, n_codebook in zip(supported_bandwidths, supported_n_codebooks)
        }

    def apply(self, audio: list, sampling_rate: int, **params) -> np.ndarray:
        """Run the DAC round trip on ``audio`` with ``params['n_codebooks_dac']``."""
        audio_arr = np.array(audio)

        n_codebooks = params.get("n_codebooks_dac", self.config.get("n_codebooks_dac"))
        if n_codebooks is None:
            n_codebooks = self.n_codebooks

        if n_codebooks not in self.supported_n_codebooks:
            logger.warning(f"n_codebooks={n_codebooks} not in supported range {self.supported_n_codebooks}, using {self.n_codebooks}")
            n_codebooks = self.n_codebooks

        waveform = torch.tensor(audio_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        waveform = waveform.to(self.device)

        target_sr = self.config.get("target_sampling_rate_dac", 44100)
        if sampling_rate != target_sr:
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=sampling_rate,
                new_freq=target_sr
            ).to(self.device)
            waveform = resampler(waveform)

        try:
            with torch.no_grad():
                original_length = waveform.shape[-1]
                reconstructed = self.model(waveform, n_quantizers=n_codebooks)['audio']
                reconstructed = reconstructed[..., :original_length]
        except Exception as e:
            logger.error(f"DAC encoding/decoding failed: {e}")
            raise

        if sampling_rate != target_sr:
            resampler_back = torchaudio.transforms.Resample(
                orig_freq=target_sr,
                new_freq=sampling_rate
            ).to(self.device)
            reconstructed = resampler_back(reconstructed)

        result = reconstructed.squeeze().cpu().numpy()
        return result

