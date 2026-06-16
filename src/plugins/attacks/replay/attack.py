import logging
import numpy as np
import os
import random
import librosa
from scipy.signal import butter, sosfilt

from core.base_attack import BaseAttack

logger = logging.getLogger(__name__)


class ReplayAttack(BaseAttack):

    def _bandpass_filter(self, audio, sr, low_freq, high_freq, order=4):
        """Apply bandpass filter to audio."""
        nyquist = sr / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        low = max(0.001, min(low, 0.999))
        high = max(0.001, min(high, 0.999))
        sos = butter(order, [low, high], btype='band', output='sos')
        return sosfilt(sos, audio)

    def _add_gaussian_noise(self, audio, snr_db):
        """Add Gaussian noise to audio at specified SNR."""
        signal_power = np.mean(audio ** 2)
        if signal_power == 0:
            return audio
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        return audio + noise

    def _convolve_with_air(self, audio, audio_sr, air_path, air_sr=48000):
        """Convolve audio signal with an impulse response."""

        ir, ir_sr = librosa.load(air_path, sr=None, mono=True)
        if ir_sr != audio_sr:
            ir = librosa.resample(ir, orig_sr=ir_sr, target_sr=audio_sr)

        convolved = np.convolve(audio, ir, mode='full')
        convolved = convolved[:len(audio)]

        max_val = np.max(np.abs(convolved))
        if max_val > 0:
            convolved = convolved / max_val * 0.95

        return convolved

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply a replay attack to an audio signal.

        Simulates recording audio through a phone in a room by:
        1) Convolve with a random Room Impulse Response (RIR)
        2) Apply a bandpass filter
        3) Add Gaussian noise

        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters for the replay attack:

        Returns:
            np.ndarray: The processed audio signal.
        """
        sampling_rate = kwargs.get("sampling_rate", None)
        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")

        # get parameters from kwargs or config
        air_folder = kwargs.get("air_folder_replay", self.config.get("air_folder_replay", "AIR_wav_files"))
        air_sr = kwargs.get("air_sr_replay", self.config.get("air_sr_replay", 48000))
        bandpass = kwargs.get("bandpass_replay", self.config.get("bandpass_replay", True))
        low_freq = kwargs.get("low_freq_replay", self.config.get("low_freq_replay", 50))
        high_freq = kwargs.get("high_freq_replay", self.config.get("high_freq_replay", 8000))
        filter_order = kwargs.get("filter_order_replay", self.config.get("filter_order_replay", 4))
        add_noise = kwargs.get("add_noise_replay", self.config.get("add_noise_replay", True))
        snr_db = kwargs.get("snr_db_replay", self.config.get("snr_db_replay", 50))

        if not os.path.isabs(air_folder):
            # get project root (4 levels up from attack.py)
            attack_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(attack_dir))))
            air_folder = os.path.join(project_root, air_folder)

        # get list of AIR files
        air_files = [f for f in os.listdir(air_folder) if f.endswith('.wav')]
        if not air_files:
            raise ValueError(f"No .wav files found in {air_folder}")

        # pick random AIR
        air_file = random.choice(air_files)
        air_path = os.path.join(air_folder, air_file)

        output = self._convolve_with_air(audio, sampling_rate, air_path, air_sr=air_sr)
        if bandpass:
            output = self._bandpass_filter(output, sampling_rate, low_freq, high_freq, order=filter_order)
        if add_noise:
            output = self._add_gaussian_noise(output, snr_db=snr_db)

        return output