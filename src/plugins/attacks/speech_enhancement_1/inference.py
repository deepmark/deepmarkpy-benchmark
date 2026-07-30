import numpy as np
import torch
from speechbrain.inference.enhancement import (
    SpectralMaskEnhancement,
    WaveformEnhancement,
)

from utils.utils import resample_audio


class SpeechBrain:
    def __init__(self, type):
        assert type=="waveform" or type=="spectral_mask", "type must be either 'waveform' or 'spectral_mask'."

        if type=="waveform":
            self.model = WaveformEnhancement.from_hparams(source="speechbrain/mtl-mimic-voicebank")
        else:
            self.model = SpectralMaskEnhancement.from_hparams(source="speechbrain/metricgan-plus-voicebank")
        
        self.model.eval()

    def inference(self, audio, sampling_rate, noise_strength):
        assert abs(noise_strength) <= 0.01, "noise_strength should not be greater than 0.01."
        audio = resample_audio(audio, input_sr=sampling_rate, target_sr=16000)
        noisy = audio +noise_strength*np.random.normal(0, 1, size=(len(audio)))
        noisy = np.expand_dims(noisy, axis=[0])
        noisy = torch.FloatTensor(noisy)
        lengths = torch.FloatTensor([1.0])
        with torch.no_grad():
            enhanced = self.model.enhance_batch(noisy, lengths=lengths)
            enhanced = enhanced.squeeze().detach().numpy()

        enhanced = resample_audio(enhanced, input_sr=16000, target_sr=sampling_rate)
        return enhanced