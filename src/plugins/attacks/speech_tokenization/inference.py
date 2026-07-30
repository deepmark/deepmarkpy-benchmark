import torch
from xcodec2.modeling_xcodec2 import XCodec2Model

from utils.utils import resample_audio


class XCodec:
    def __init__(self, model_name, device):
        self.device = device

        self.model = XCodec2Model.from_pretrained(model_name)
        self.model.eval().to(self.device)

    def inference(self, audio, sampling_rate):
        audio = resample_audio(audio, input_sr=sampling_rate, target_sr=16000)

        audio = torch.from_numpy(audio).float().unsqueeze(0)

        with torch.no_grad():
            vq_code = self.model.encode_code(input_waveform=audio)

            output = self.model.decode_code(vq_code)[0, 0, :].cpu().numpy()

        return resample_audio(output, input_sr=16000, target_sr=sampling_rate)


