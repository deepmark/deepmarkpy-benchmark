"""TimbreWatermarking embed/detect inference, HTTP-free.

Imports follow the container layout: the ``wm_model`` variant modules and
checkpoints resolve from the image's ``TimbreWatermarking/watermarking_model``
clone via the working-directory-relative ``sys.path.append``. Both endpoints
intentionally feed ``resample_audio`` the raw request list.
"""

import logging
import os
import sys

import numpy as np
import torch
import yaml

from deepmarkpy.utils.utils import resample_audio

from deepmarkpy.core.inference import BaseModelEngine

# Named "app" so the debug lines below keep their existing format in the
# service log output ("DEBUG:app:...").
logger = logging.getLogger("app")

sys.path.append("TimbreWatermarking/watermarking_model")


def load_model(process_config, model_config, train_config, device):
    """Build the configured Encoder/Decoder variant and load its checkpoint.

    The variant is dispatched from the model config's structure flags; the
    checkpoint is either the configured name or the index-th file of the
    model directory sorted by mtime.
    """
    if model_config["structure"]["transformer"]:
        if model_config["structure"]["mel"]:
            from wm_model.mel_modules import Encoder, Decoder
        else:
            from wm_model.modules import Encoder, Decoder
    elif model_config["structure"].get("conv2", False):
        from wm_model.conv2_modules import Encoder, Decoder
    elif model_config["structure"].get("conv2mel", False):
        if not model_config["structure"].get("ab", False):
            from wm_model.conv2_mel_modules import Encoder, Decoder
        else:
            from wm_model.conv2_mel_modules_ab import Encoder, Decoder
    else:
        from wm_model.conv_modules import Encoder, Decoder
    win_dim = process_config["audio"]["win_len"]
    embedding_dim = model_config["dim"]["embedding"]
    nlayers_encoder = model_config["layer"]["nlayers_encoder"]
    nlayers_decoder = model_config["layer"]["nlayers_decoder"]
    attention_heads_encoder = model_config["layer"]["attention_heads_encoder"]
    attention_heads_decoder = model_config["layer"]["attention_heads_decoder"]
    msg_length = train_config["watermark"]["length"]
    if model_config["structure"].get("mel", False) or model_config["structure"].get("conv2", False):
        encoder = Encoder(process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_encoder=nlayers_encoder, attention_heads=attention_heads_encoder).to(device)
        decoder = Decoder(process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder=nlayers_decoder, attention_heads=attention_heads_decoder).to(device)
    else:
        encoder = Encoder(model_config, msg_length, win_dim, embedding_dim, nlayers_encoder=nlayers_encoder, attention_heads=attention_heads_encoder).to(device)
        decoder = Decoder(model_config, msg_length, win_dim, embedding_dim, nlayers_decoder=nlayers_decoder, attention_heads=attention_heads_decoder).to(device)
    path_model = os.path.join("TimbreWatermarking/watermarking_model", model_config["test"]["model_path"])
    model_name = model_config["test"]["model_name"]
    if model_name:
        model = torch.load(os.path.join(path_model, model_name), map_location=device)
    else:
        index = model_config["test"]["index"]
        model_list = os.listdir(path_model)
        model_list = sorted(model_list, key=lambda x: os.path.getmtime(os.path.join(path_model, x)))
        model_path = os.path.join(path_model, model_list[index])
        model = torch.load(model_path, map_location=device)
    encoder.load_state_dict(model["encoder"])
    decoder.load_state_dict(model["decoder"], strict=False)
    encoder.eval()
    decoder.eval()
    return encoder, decoder, msg_length


class TimbreWMEngine(BaseModelEngine):
    """TimbreWatermarking embed/detect with bipolar message mapping.

    The Encoder/Decoder pair loads once at construction from the upstream
    YAML configs. embed maps request bits to {-1, 1}; detect thresholds the
    decoder output back to {0, 1}.
    """

    def __init__(self, config: dict, device: str | None = None):
        """Load the configured Encoder/Decoder variant onto ``device``."""
        self.config = config
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        self.device = device

        process_config = yaml.load(open("TimbreWatermarking/watermarking_model/config/process.yaml", "r"), Loader=yaml.FullLoader)
        model_config = yaml.load(open("TimbreWatermarking/watermarking_model/config/model.yaml", "r"), Loader=yaml.FullLoader)
        train_config = yaml.load(open("TimbreWatermarking/watermarking_model/config/train.yaml", "r"), Loader=yaml.FullLoader)
        self.embedder, self.detector, self.msg_length = load_model(process_config, model_config, train_config, device)

    def embed(self, audio: list, watermark_data: list, sampling_rate: int) -> np.ndarray:
        """Embed ``watermark_data`` (bits, mapped to bipolar); returns the signal."""
        audio_arr = np.array(audio)
        watermark_arr = np.array(watermark_data, dtype=np.float32)
        logger.debug(f"Audio shape: {audio_arr.shape}, Sampling rate: {sampling_rate}")

        if sampling_rate != self.config["sampling_rate"]:
            logger.debug(f"Resampling from {sampling_rate} to {self.config['sampling_rate']}")
            audio_arr = resample_audio(audio, sampling_rate, self.config["sampling_rate"])

        wav = torch.tensor(audio_arr, dtype=torch.float32)
        wav = wav.unsqueeze(0).unsqueeze(0).to(self.device)
        logger.debug(f"WAV tensor shape: {wav.shape}, dtype: {wav.dtype}")

        msg = torch.from_numpy(watermark_arr).float().unsqueeze(0).unsqueeze(0).to(self.device)
        msg = msg * 2 - 1
        logger.debug(f"MSG tensor shape: {msg.shape}, dtype: {msg.dtype}")

        logger.debug("Starting model inference")
        with torch.no_grad():
            watermarked_audio, _ = self.embedder.test_forward(wav, msg)
        logger.debug(f"Model inference complete. Result shape: {watermarked_audio.shape}")

        watermarked_audio = watermarked_audio.squeeze().cpu().numpy()

        if sampling_rate != self.config["sampling_rate"]:
            logger.debug(f"Resampling output from {self.config['sampling_rate']} to {sampling_rate}")
            watermarked_audio = resample_audio(watermarked_audio, self.config["sampling_rate"], sampling_rate)

        return watermarked_audio

    def detect(self, audio: list, sampling_rate: int) -> np.ndarray:
        """Decode the watermark; returns the thresholded {0, 1} message."""
        audio_arr = np.array(audio)
        logger.debug(f"Audio shape: {audio_arr.shape}, Sampling rate: {sampling_rate}")

        if sampling_rate != self.config["sampling_rate"]:
            logger.debug(f"Resampling from {sampling_rate} to {self.config['sampling_rate']}")
            audio_arr = resample_audio(audio, sampling_rate, self.config["sampling_rate"])

        wav = torch.tensor(audio_arr, dtype=torch.float32)
        wav = wav.unsqueeze(0).unsqueeze(0).to(self.device)
        logger.debug(f"WAV tensor shape: {wav.shape}, dtype: {wav.dtype}")

        logger.debug("Starting model inference")
        with torch.no_grad():
            message = self.detector.test_forward(wav)
        logger.debug(f"Model inference complete. Result shape: {message.shape}, dtype: {message.dtype}")

        message = torch.where(message >= 0, 1, -1)
        message = (message + 1) / 2
        message = message.squeeze().cpu().numpy()
        logger.debug(f"Processed message shape: {message.shape}")

        return message


# Stable import alias.
Engine = TimbreWMEngine
