import os
import sys

import torch
from app_utils.utils import resample_audio

sys.path.append("BigVGAN")

import bigvgan
from meldataset import get_mel_spectrogram


class BigVGAN:
    def __init__(self, model_name, device):

        self.device = device
        self.model = bigvgan.BigVGAN.from_pretrained(model_name)
        self.model.remove_weight_norm()
        self.model.eval().to(self.device)

    def inference(self, audio, sampling_rate):
        audio = resample_audio(audio, input_sr=sampling_rate, target_sr=44100)
        audio = torch.FloatTensor(audio).unsqueeze(0)

        mel = get_mel_spectrogram(audio, self.model.h).to(self.device) 
        
        with torch.inference_mode():
            output = self.model(mel) 

        output = output.squeeze().cpu().numpy() 
        
        return resample_audio(output, input_sr=44100, target_sr=sampling_rate)
