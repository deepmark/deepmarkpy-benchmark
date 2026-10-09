import os
import subprocess
import tempfile
import time

import numpy as np
import soundfile as sf

from deepmarkpy.core.base_attack import BaseAttack


class AacCompressionAttack(BaseAttack):
    """Lossy AAC transcode-and-decode attack.

    Mp3CompressionAttack models "the file got lossily recompressed," but AAC,
    not MP3, is the codec that actually sits in most of today's real
    distribution paths for a watermarked file: YouTube/M4A, Apple Music,
    the MP4 audio track TikTok/Instagram/WhatsApp re-encode into, and the
    Bluetooth A2DP AAC profile. This attack encodes with ffmpeg's built-in
    `aac` encoder (LC profile; avoids the nonfree `libfdk_aac` build) at a
    configurable bitrate and decodes it back to PCM, the same
    encode-then-decode round trip Mp3CompressionAttack applies for the
    codec AAC has largely displaced in that role.

    The container matters here, not just the codec: AAC's encoder has an
    inherent ~1024-sample priming delay, and only a container with an edit
    list (MP4/`.m4a`) records where real audio starts so a decoder can trim
    it back out. Measured on this machine's ffmpeg (8.1.2): round-tripping
    through raw ADTS (`.aac`) leaves a 1024-sample leading offset with zero
    trimming, which would register as a desync artifact having nothing to
    do with AAC's actual lossy damage; round-tripping through `.m4a` came
    back sample-aligned (0-sample lag), which is also what every player on
    the platforms above actually does. This attack therefore transcodes
    through `.m4a`, not `.aac`.

    libsndfile has no AAC decoder, so — unlike Mp3CompressionAttack, whose
    MP3 decode goes through `soundfile.read` — the decode step here also
    goes through ffmpeg.
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

            # Small delay to ensure FFmpeg fully releases files
            time.sleep(0.1)

            # Decode back to PCM via ffmpeg -- libsndfile has no AAC decoder,
            # and going through the MP4 demuxer is what applies the edit
            # list that keeps this round trip sample-aligned (see class
            # docstring).
            subprocess.run(
                ["ffmpeg", "-y", "-i", m4a_path, wav_out_path],
                capture_output=True, check=True,
            )

            time.sleep(0.1)

            decoded_audio, _ = sf.read(wav_out_path, dtype="float32")

            time.sleep(0.1)

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"FFmpeg AAC transcode failed: {e.stderr.decode(errors='replace') if e.stderr else str(e)}"
            )
        except Exception as e:
            raise e
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
                    print(f"Warning: Could not delete {filepath} after {max_retries} attempts")
