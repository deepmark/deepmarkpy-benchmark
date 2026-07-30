"""All inference for the speech_tokenization attack service (REORG_PLAN.md §5.1).

No FastAPI/HTTP imports. Logic moved verbatim from xcodec.py. The 16 kHz
resample here duplicates app.py's request-level resample (defect D9,
docs/KNOWN_DEFECTS.md) — preserved exactly, do not de-duplicate. app.py
also keeps its request-path resample block and INFO logs: they are visible
container output under its ``logging.basicConfig`` (moving them would
change the logged logger name — user-visible text).
"""

import torch
from xcodec2.modeling_xcodec2 import XCodec2Model

from utils.utils import resample_audio


class Engine:
    """XCodec2 tokenize/detokenize round-trip attack.

    The codec loads at construction (startup-loaded stays startup-loaded).
    ``device`` is passed by app.py, which keeps its own device selection and
    its visible "Using device" log line.
    """

    def __init__(self, config: dict, device: str | None = None):
        """Load XCodec2 from ``config['model_name_speech_tokenization']``."""
        self.config = config
        self.device = device

        self.model = XCodec2Model.from_pretrained(config["model_name_speech_tokenization"])
        self.model.eval().to(self.device)

    def apply(self, audio, sampling_rate: int, **params):
        """Encode ``audio`` to VQ codes and decode back (includes the D9 resample)."""
        audio = resample_audio(audio, input_sr=sampling_rate, target_sr=16000)

        audio = torch.from_numpy(audio).float().unsqueeze(0)

        with torch.no_grad():
            vq_code = self.model.encode_code(input_waveform=audio)

            output = self.model.decode_code(vq_code)[0, 0, :].cpu().numpy()

        return resample_audio(output, input_sr=16000, target_sr=sampling_rate)
