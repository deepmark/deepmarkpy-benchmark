"""XCodec2 tokenize/detokenize attack inference, HTTP-free.

The 16 kHz resample here intentionally duplicates app.py's request-level
resample — a preserved redundancy.
"""

import torch
from xcodec2.modeling_xcodec2 import XCodec2Model

from deepmarkpy.utils.utils import resample_audio

from deepmarkpy.core.inference import BaseAttackEngine

# The snapshot the Dockerfile bakes into the image. Loading without it let a
# newer upstream "main" replace the weights at service start.
MODEL_REVISION = "e412427ed30f0cf9d5e3c95562113deb10a32d03"

# At every load, transformers warns that 74 CodecEnc weights are "newly
# initialized". The checkpoint stores those weight-normed convolutions as
# weight_g / weight_v, and the model names them parametrizations...original0 /
# original1. transformers compares the names before loading. PyTorch's
# weight-norm parametrization then renames the old keys in a load hook, so the
# weights do load: all 74 tensors equal the checkpoint's. The warning is a
# false alarm. GPU study §4.1 read it as a random encoder; that was wrong.
# Correctness rests on that hook, so it is checked rather than assumed, and a
# version that stops honouring it fails here instead of running half-random.
_WEIGHT_NORM_KEYS = (
    (".parametrizations.weight.original0", ".weight_g"),
    (".parametrizations.weight.original1", ".weight_v"),
)


def load_codec(name: str, revision: str = MODEL_REVISION) -> XCodec2Model:
    """Load XCodec2, and raise unless every reportedly missing weight holds the checkpoint's value."""
    model, info = XCodec2Model.from_pretrained(name, revision=revision, output_loading_info=True)
    missing = info["missing_keys"]
    if not missing:
        return model

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    state = model.state_dict()
    unexplained = []
    with safe_open(hf_hub_download(name, "model.safetensors", revision=revision), framework="pt") as checkpoint:
        stored = set(checkpoint.keys())
        for key in missing:
            source = next((key[: -len(new)] + old for new, old in _WEIGHT_NORM_KEYS if key.endswith(new)), None)
            if source not in stored or not torch.equal(checkpoint.get_tensor(source), state[key]):
                unexplained.append(key)
    if unexplained:
        raise RuntimeError(
            f"XCodec2 {name}@{revision}: {len(unexplained)} weights do not hold the "
            f"checkpoint's values, e.g. {unexplained[:3]}"
        )
    return model


class SpeechTokenizationEngine(BaseAttackEngine):
    """XCodec2 tokenize/detokenize round-trip attack.

    The codec loads once at construction onto the ``device`` chosen by
    app.py.
    """

    def __init__(self, config: dict, device: str | None = None):
        """Load XCodec2 from ``config['model_name_speech_tokenization']``."""
        self.config = config
        self.device = device

        self.model = load_codec(config["model_name_speech_tokenization"])
        self.model.eval().to(self.device)

    def apply(self, audio, sampling_rate: int, **params):
        """Encode ``audio`` to VQ codes and decode back."""
        audio = resample_audio(audio, input_sr=sampling_rate, target_sr=16000)

        audio = torch.from_numpy(audio).float().unsqueeze(0)

        with torch.no_grad():
            vq_code = self.model.encode_code(input_waveform=audio)

            output = self.model.decode_code(vq_code)[0, 0, :].cpu().numpy()

        return resample_audio(output, input_sr=16000, target_sr=sampling_rate)

