import logging

import numpy as np
import pycodec2

from deepmarkpy.core.base_attack import BaseAttack
from deepmarkpy.utils.utils import resample_audio

logger = logging.getLogger(__name__)

CODEC2_SAMPLE_RATE = 8000
SUPPORTED_BITRATES = {700, 1200, 1300, 1400, 1600, 2400, 3200}


class Codec2VocoderAttack(BaseAttack):
    """Low-bitrate parametric vocoder attack using Codec2.

    Encodes audio at a specified bitrate (700-3200 bps) then decodes back
    to PCM. This simulates transmission through a low-bitrate voice channel
    (similar to MELP/MELPe military vocoders).

    Codec2 requires 8kHz mono input, so the attack resamples internally.
    """

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        sampling_rate = kwargs.get("sampling_rate", 16000)
        bitrate = kwargs.get("bitrate", self.config.get("bitrate", 1300))

        if isinstance(bitrate, list):
            bitrate = bitrate[0]

        if bitrate not in SUPPORTED_BITRATES:
            logger.warning(
                f"Codec2 bitrate {bitrate} not supported. "
                f"Supported: {sorted(SUPPORTED_BITRATES)}. Returning audio unchanged."
            )
            return audio

        if sampling_rate != CODEC2_SAMPLE_RATE:
            audio_8k = resample_audio(audio, sampling_rate, CODEC2_SAMPLE_RATE)
        else:
            audio_8k = audio

        audio_8k = np.clip(audio_8k, -1.0, 1.0)
        audio_int16 = (audio_8k * 32767).astype(np.int16)

        codec = pycodec2.Codec2(bitrate)
        samples_per_frame = codec.samples_per_frame()

        # Pad to multiple of frame size
        n_samples = len(audio_int16)
        remainder = n_samples % samples_per_frame
        if remainder != 0:
            pad_len = samples_per_frame - remainder
            audio_int16 = np.pad(audio_int16, (0, pad_len), mode='constant')

        # Encode then decode frame by frame
        decoded_frames = []
        for i in range(0, len(audio_int16), samples_per_frame):
            frame = audio_int16[i:i + samples_per_frame]
            encoded = codec.encode(frame)
            decoded = codec.decode(encoded)
            decoded_frames.append(decoded)

        decoded_audio = np.concatenate(decoded_frames)[:n_samples]
        decoded_float = decoded_audio.astype(np.float32) / 32767.0

        if sampling_rate != CODEC2_SAMPLE_RATE:
            decoded_float = resample_audio(decoded_float, CODEC2_SAMPLE_RATE, sampling_rate)

        # Match original length
        if len(decoded_float) > len(audio):
            decoded_float = decoded_float[:len(audio)]
        elif len(decoded_float) < len(audio):
            decoded_float = np.pad(decoded_float, (0, len(audio) - len(decoded_float)))

        return decoded_float
