import logging
import os
import subprocess
import tempfile
import time

import numpy as np
import soundfile as sf

from deepmarkpy.core.base_attack import BaseAttack

logger = logging.getLogger(__name__)


class AacCompressionAttack(BaseAttack):
    """Lossy AAC transcode-and-decode attack.

    Encodes with ffmpeg's built-in AAC-LC encoder at the configured target
    bitrate (64 kbps by default), then decodes back to PCM with ffmpeg.

    Uses an MP4 (`.m4a`) container so the decoder can account for encoder
    priming delay. Raw ADTS (`.aac`) does not carry the edit list needed
    for this alignment. Trailing frame padding is trimmed to the input
    length after decoding.
    """

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters for the AAC compression:
                - sampling_rate (int): The sampling rate of the audio signal in Hz (required).
                - bitrate_aac (int): Target AAC bitrate in kbps.

        Returns:
            np.ndarray: The audio after an AAC encode/decode round trip,
                trimmed or zero-padded back to the input length (safe here
                because the `.m4a` round trip is sample-aligned; any length
                change is trailing frame padding, not a leading shift).

        Raises:
            ValueError: If `sampling_rate` is not provided in `kwargs`.
            RuntimeError: If ffmpeg is unavailable or the transcode fails.
        """
        sampling_rate = kwargs.get("sampling_rate", None)
        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided.")

        bitrate = kwargs.get("bitrate_aac", self.config.get("bitrate_aac"))

        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("FFmpeg not found. Please install FFmpeg or skip AAC tests.")

        wav_in_fd, wav_in_path = tempfile.mkstemp(suffix=".wav")
        m4a_fd, m4a_path = tempfile.mkstemp(suffix=".m4a")
        wav_out_fd, wav_out_path = tempfile.mkstemp(suffix=".wav")

        try:
            # Close file descriptors immediately to avoid conflicts
            os.close(wav_in_fd)
            os.close(m4a_fd)
            os.close(wav_out_fd)

            audio = np.asarray(audio, dtype=np.float32)
            sf.write(wav_in_path, audio, sampling_rate)

            # Encode to AAC-in-MP4. -y overwrites the pre-created temp file.
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", wav_in_path,
                    "-c:a", "aac", "-b:a", f"{bitrate}k",
                    m4a_path,
                ],
                capture_output=True, check=True,
            )

            # Decode back to PCM via ffmpeg -- libsndfile has no AAC decoder,
            # and going through the MP4 demuxer is what applies the edit
            # list that keeps this round trip sample-aligned (see class
            # docstring).
            subprocess.run(
                ["ffmpeg", "-y", "-i", m4a_path, wav_out_path],
                capture_output=True, check=True,
            )

            decoded_audio, _ = sf.read(wav_out_path, dtype="float32")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"FFmpeg AAC transcode failed: {e.stderr.decode(errors='replace') if e.stderr else str(e)}"
            )
        finally:
            # Clean up temporary files with retry logic
            self.safe_delete(wav_in_path)
            self.safe_delete(m4a_path)
            self.safe_delete(wav_out_path)

        if len(decoded_audio) > len(audio):
            decoded_audio = decoded_audio[: len(audio)]
        elif len(decoded_audio) < len(audio):
            decoded_audio = np.pad(decoded_audio, (0, len(audio) - len(decoded_audio)))

        return decoded_audio

    def safe_delete(self, filepath: str, max_retries: int = 5) -> None:
        """Safely delete a file with retries for Windows file locking issues"""
        for attempt in range(max_retries):
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                return
            except PermissionError:
                if attempt < max_retries - 1:
                    time.sleep(0.1)  # Wait 100ms before retry
                else:
                    logger.warning(
                        "Could not delete %s after %d attempts", filepath, max_retries
                    )
