"""SpeechBrain enhancement attack inference, HTTP-free.

The noise term draws from the unseeded global NumPy RNG with a
request-controlled ``noise_strength``; the magnitude assert propagates as
the service's 500 error.
"""

import numpy as np
import torch
from speechbrain.inference.enhancement import (
    SpectralMaskEnhancement,
    WaveformEnhancement,
)
from speechbrain.utils.fetching import FetchConfig

from deepmarkpy.utils.utils import resample_audio

# Pinned so a rebuild cannot silently pick up a different upstream checkpoint.
# SpeechBrain takes the revision through FetchConfig rather than a from_hparams
# keyword; a bare revision= lands in **kwargs and reaches Pretrained.__init__,
# which rejects it.
WAVEFORM_REVISION = "5142933779578738d62d0d1f79290e824c8cd2fb"
SPECTRAL_MASK_REVISION = "a196ce26b3bdace6fa1d819017584bdbcce462a8"

from deepmarkpy.core.inference import BaseAttackEngine


class SpeechEnhancement1Engine(BaseAttackEngine):
    """SpeechBrain enhancement at 16 kHz with request-rate round-trips.

    The enhancement model loads once at construction. ``device`` is
    accepted for interface uniformity and unused — no device placement
    happens here.
    """

    def __init__(self, config: dict, device: str | None = None):
        """Load the SpeechBrain model selected by ``config['type_se1']``."""
        self.config = config
        type = config["type_se1"]
        assert type=="waveform" or type=="spectral_mask", "type must be either 'waveform' or 'spectral_mask'."

        if type=="waveform":
            self.model = WaveformEnhancement.from_hparams(
                source="speechbrain/mtl-mimic-voicebank",
                fetch_config=FetchConfig(revision=WAVEFORM_REVISION),
            )
        else:
            self.model = SpectralMaskEnhancement.from_hparams(
                source="speechbrain/metricgan-plus-voicebank",
                fetch_config=FetchConfig(revision=SPECTRAL_MASK_REVISION),
            )

        self.model.eval()

    def apply(self, audio: list, sampling_rate: int, **params) -> np.ndarray:
        """Enhance ``audio`` after adding ``params['noise_strength']`` noise."""
        audio_arr = np.array(audio)
        noise_strength = params["noise_strength"]
        assert abs(noise_strength) <= 0.01, "noise_strength should not be greater than 0.01."
        audio_arr = resample_audio(audio_arr, input_sr=sampling_rate, target_sr=16000)
        noisy = audio_arr +noise_strength*np.random.normal(0, 1, size=(len(audio_arr)))
        noisy = np.expand_dims(noisy, axis=[0])
        noisy = torch.FloatTensor(noisy)
        lengths = torch.FloatTensor([1.0])
        with torch.no_grad():
            enhanced = self.model.enhance_batch(noisy, lengths=lengths)
            enhanced = enhanced.squeeze().detach().numpy()

        enhanced = resample_audio(enhanced, input_sr=16000, target_sr=sampling_rate)
        return enhanced

