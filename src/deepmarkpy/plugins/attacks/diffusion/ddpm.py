import numpy as np
import torch
from diffusers import DiffusionPipeline

from deepmarkpy.utils.utils import renormalize_audio, resample_audio


class DDPM:
    def __init__(self, model_name, device):
        self.device = device
        self.model = DiffusionPipeline.from_pretrained(
            model_name, cache_dir="diffusers"
        )
        self.model.to(self.device)

    def inference(self, audio, sampling_rate, diffusion_steps):
        mel = self.model.mel
        mel_sample_rate = mel.get_sample_rate()
        slice_size = mel.x_res * mel.hop_length

        audio = resample_audio(audio, sampling_rate, mel_sample_rate)

        overlap_secs = 2
        overlap_samples = overlap_secs * mel_sample_rate
        stride = min(slice_size - overlap_samples, len(audio))

        generator = torch.Generator(device=self.device)
        seed = generator.seed()

        output = np.array([])
        new_audio_slice = np.array([])
        not_first = 0
        for sample in range(len(audio) // stride):
            generator.manual_seed(seed)
            audio_slice = np.array(
                audio[sample * stride : sample * stride + slice_size]
            )
            if not_first:
                # Normalize and re-insert generated audio
                audio_slice[:overlap_samples] = (
                    new_audio_slice[-overlap_samples:]
                    * np.max(audio_slice[:overlap_samples])
                    / np.max(new_audio_slice[-overlap_samples:])
                )
            with torch.no_grad():
                slice = self.model(
                    raw_audio=audio_slice,
                    start_step=diffusion_steps,
                    generator=generator,
                    mask_start_secs=overlap_secs * not_first,
                )
                new_audio_slice = slice.audios[0, 0]
                output = np.concatenate(
                    [output, new_audio_slice[overlap_samples * not_first :]]
                )
                not_first = 1
        output = resample_audio(output, mel_sample_rate, sampling_rate)
        return renormalize_audio(audio, output)
