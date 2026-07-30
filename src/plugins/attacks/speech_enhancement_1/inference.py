"""All inference for the speech_enhancement_1 attack service (REORG_PLAN.md §5.1).

No FastAPI/HTTP imports. Logic moved verbatim from speech_brain.py.
Conditionally deterministic (§4.3): the noise term draws from the unseeded
global NumPy RNG, but ``noise_strength`` is request-controlled — contract
fixtures are recorded with ``noise_strength=0.0``. Do not seed (REORG_PLAN
§4.1); the ``noise_strength`` assert's failure shape (HTTP 500) is current
behavior and stays.
"""

import numpy as np
import torch
from speechbrain.inference.enhancement import (
    SpectralMaskEnhancement,
    WaveformEnhancement,
)

from utils.utils import resample_audio


class Engine:
    """SpeechBrain enhancement attack.

    The enhancement model loads at construction (startup-loaded stays
    startup-loaded). ``device`` is accepted for signature uniformity but
    unused — the current code manages no device placement, and that is
    preserved.
    """

    def __init__(self, config: dict, device: str | None = None):
        """Load the SpeechBrain model selected by ``config['type_se1']``."""
        self.config = config
        type = config["type_se1"]
        assert type=="waveform" or type=="spectral_mask", "type must be either 'waveform' or 'spectral_mask'."

        if type=="waveform":
            self.model = WaveformEnhancement.from_hparams(source="speechbrain/mtl-mimic-voicebank")
        else:
            self.model = SpectralMaskEnhancement.from_hparams(source="speechbrain/metricgan-plus-voicebank")

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
