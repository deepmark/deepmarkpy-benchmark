"""XCodec2 tokenize/detokenize attack inference, HTTP-free.

The 16 kHz resample here intentionally duplicates app.py's request-level
resample — a preserved redundancy.
"""

import torch
from xcodec2.modeling_xcodec2 import XCodec2Model

from deepmarkpy.utils.utils import resample_audio


class Engine:
    """XCodec2 tokenize/detokenize round-trip attack.

    The codec loads once at construction onto the ``device`` chosen by
    app.py.
    """

    def __init__(self, config: dict, device: str | None = None):
        """Load XCodec2 from ``config['model_name_speech_tokenization']``."""
        self.config = config
        self.device = device

        self.model = XCodec2Model.from_pretrained(config["model_name_speech_tokenization"])
        self.model.eval().to(self.device)

    def apply(self, audio, sampling_rate: int, **params):
        """Encode ``audio`` to VQ codes and decode back."""
        audio = resample_audio(audio, input_sr=sampling_rate, target_sr=16000)

        audio = torch.from_numpy(audio).float().unsqueeze(0)

        with torch.no_grad():
            vq_code = self.model.encode_code(input_waveform=audio)

            output = self.model.decode_code(vq_code)[0, 0, :].cpu().numpy()

        return resample_audio(output, input_sr=16000, target_sr=sampling_rate)
