import numpy as np
import os
import random
import librosa
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfilt
from audiocomplib import AudioCompressor


from deepmarkpy.core.base_attack import BaseAttack
from deepmarkpy.plugins.attacks.equalizer.attack import EqualizerAttack
from deepmarkpy.plugins.attacks.highpass_filter.attack import HighpassFilterAttack


class MixingAttack(BaseAttack):


    def rms_dbfs(self, audio):
        """Compute RMS level in dBFS."""
        rms = np.sqrt(np.mean(audio**2))
        if rms == 0:
            return -np.inf
        return 20 * np.log10(rms)


    def apply_gain_staging(self, audio, target_dbfs=-18.0):
        """
        Adjust audio gain so its RMS level matches target dBFS.

        Args:
            audio (np.ndarray): Input audio signal (-1 to 1 range)
            target_dbfs (float): Desired RMS level in dBFS

        Returns:
            adjusted_audio (np.ndarray)
            applied_gain_db (float)
        """
        current_dbfs = self.rms_dbfs(audio)
        gain_db = target_dbfs - current_dbfs
        gain_linear = 10 ** (gain_db / 20)
        adjusted_audio = audio * gain_linear
        
        return adjusted_audio, gain_db


    def _k_weighting_filter(self, audio, sr):
        """Apply K-weighting filter for LUFS measurement.

        K-weighting consists of:
        1. High-shelf filter (+4dB above 1500Hz)
        2. High-pass filter (removing <38Hz)

        Args:
            audio: Input audio signal
            sr: Sample rate

        Returns:
            K-weighted audio
        """
        # Stage 1: High-shelf filter (+4dB at high frequencies)
        # Approximation using a high-shelf biquad
        f0 = 1500.0  # Shelf frequency
        G = 4.0  # Gain in dB
        Q = 0.707

        A = 10 ** (G / 40)
        w0 = 2 * np.pi * f0 / sr
        alpha = np.sin(w0) / (2 * Q)

        b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
        a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

        # Normalize coefficients
        b = np.array([b0/a0, b1/a0, b2/a0])
        a = np.array([1, a1/a0, a2/a0])

        # Apply high-shelf filter
        from scipy.signal import lfilter
        audio_hs = lfilter(b, a, audio)

        # Stage 2: High-pass filter at 38Hz
        hp_freq = 38.0
        nyquist = sr / 2
        if hp_freq < nyquist:
            sos_hp = butter(2, hp_freq / nyquist, btype='high', output='sos')
            audio_k = sosfilt(sos_hp, audio_hs)
        else:
            audio_k = audio_hs

        return audio_k

    def _detect_voice_activity_lufs(self, audio, sr, threshold=-40, window_seconds=0.4):
        """Detect voice activity using short-term LUFS (Loudness Units Full Scale).
        Args:
            audio: Input audio signal
            sr: Sample rate
            threshold: LUFS threshold relative to max (in dB, e.g., -40 means 40dB below max)
            window_seconds: Window length for short-term LUFS (default 400ms per standard)

        Returns:
            voice_mask: Binary mask of voice activity
            lufs_envelope: Short-term LUFS values (normalized 0-1)
        """
        # Apply K-weighting filter
        audio_k = self._k_weighting_filter(audio, sr)

        # Calculate short-term loudness using sliding window
        window_samples = int(window_seconds * sr)

        # Calculate mean square in sliding window
        audio_k_squared = audio_k ** 2
        mean_square = uniform_filter1d(audio_k_squared, size=window_samples, mode='constant')

        # Avoid log of zero
        mean_square = np.maximum(mean_square, 1e-10)

        # Convert to LUFS (dB scale, relative to full scale)
        lufs = -0.691 + 10 * np.log10(mean_square)

        # Normalize LUFS to 0-1 range for envelope
        max_lufs = np.max(lufs)
        min_lufs = max_lufs + threshold  # threshold dB below max

        # Map LUFS to 0-1 range
        lufs_normalized = (lufs - min_lufs) / (max_lufs - min_lufs + 1e-6)
        lufs_normalized = np.clip(lufs_normalized, 0, 1)

        # Create binary voice activity mask
        voice_mask = (lufs > min_lufs).astype(float)

        return voice_mask, lufs_normalized

    def _smooth_envelope(self, envelope, sr, window_seconds=0.5):
        """Smooth the voice envelope for gradual transitions.

        Args:
            envelope: Voice activity envelope
            sr: Sample rate
            window_seconds: Smoothing window in seconds

        Returns:
            Smoothed envelope
        """
        window_samples = int(window_seconds * sr)
        if window_samples < 1:
            window_samples = 1

        # Apply multiple passes of smoothing for very gradual transitions
        smoothed = uniform_filter1d(envelope, size=window_samples, mode='constant')
        smoothed = uniform_filter1d(smoothed, size=window_samples, mode='constant')
        smoothed = uniform_filter1d(smoothed, size=window_samples // 2, mode='constant')

        return smoothed

    def _load_random_music(self, music_folder, target_length, sr):
        """Load a random music file and match it to target length.

        Args:
            music_folder: Folder containing music files
            target_length: Target length in samples
            sr: Target sample rate

        Returns:
            Music audio array of target_length
        """
        # Get list of music files
        music_files = [f for f in os.listdir(music_folder)
                      if f.endswith(('.wav', '.mp3'))]

        if not music_files:
            raise ValueError(f"No music files found in {music_folder}")

        # Pick random music file
        music_file = random.choice(music_files)
        music_path = os.path.join(music_folder, music_file)

        # Load music at target sample rate
        music, _ = librosa.load(music_path, sr=sr, mono=True)

        # Match length to target
        if len(music) >= target_length:
            # Randomly pick a starting point
            max_start = len(music) - target_length
            start = random.randint(0, max_start) if max_start > 0 else 0
            music = music[start:start + target_length]
        else:
            # Loop music to fill target length
            repeats = int(np.ceil(target_length / len(music)))
            music = np.tile(music, repeats)[:target_length]

        return music

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply mixing attack by adding background music with automatic ducking.
        Music volume is automatically adjusted based on voice activity:
        - Higher volume where there's no speech (but not exceeding audio level)
        - Lower volume where there's speech (as percentage of audio level)

        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters:
                - sampling_rate_mixing (int): Sample rate of the audio
                - music_folder_mixing (str): Folder containing music files
                - music_volume_high_mixing (float): Max music volume as fraction of audio max (0-1)
                - music_volume_low_mixing (float): Music volume during speech as fraction of audio max (0-1)
                - smoothing_window_mixing (float): Smoothing window in seconds

        Returns:
            np.ndarray: Audio mixed with ducked background music
        """
        sampling_rate = kwargs.get("sampling_rate", 16000)
        music_folder = kwargs.get("music_folder_mixing", self.config.get("music_folder_mixing", "music"))
        volume_high_ratio = kwargs.get("music_volume_high_mixing", self.config.get("music_volume_high_mixing", 0.5))
        volume_low_ratio = kwargs.get("music_volume_low_mixing", self.config.get("music_volume_low_mixing", 0.1))
        smoothing_window = kwargs.get("smoothing_window_mixing", self.config.get("smoothing_window_mixing", 0.5))

        if not os.path.isabs(music_folder):
            attack_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(attack_dir)))))
            music_folder = os.path.join(project_root, music_folder)

        audio, _ = self.apply_gain_staging(audio, -18)

        # Apply highpass filter
        hp_cutoff = kwargs.get("highpass_cutoff_mixing", self.config.get("highpass_cutoff_mixing", 50))
        highpass = HighpassFilterAttack()
        audio = highpass.apply(audio, sampling_rate=sampling_rate, cutoff_freq_highpass=hp_cutoff)

        # Apply equalizer
        eq_gains = kwargs.get("eq_gains_mixing", self.config.get("eq_gains_mixing", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        equalizer = EqualizerAttack()
        audio = equalizer.apply(audio, sampling_rate=sampling_rate, gains_equalizer=eq_gains)

        # Ensure audio is 2D: (channels, samples)
        if audio.ndim == 1:
            audio_2d = audio[np.newaxis, :]  # shape becomes (1, num_samples)
        else:
            audio_2d = audio

        # Process with compressor
        compressor = AudioCompressor(
            threshold=-17.0,
            ratio=4.0,
            attack_time_ms=30.0,
            release_time_ms=70.0
        )
        processed_audio_2d = compressor.process(audio_2d, sample_rate=sampling_rate)

        # Convert back to original shape (1D)
        if audio.ndim == 1:
            audio = processed_audio_2d[0]
        else:
            audio = processed_audio_2d

        # Reference the music level to the speech as it is *after* the
        # processing chain. Taking it from the raw input peak instead made the
        # music-to-speech ratio a property of the recording's level (measured
        # ~19 dB of swing across a corpus) rather than of this attack's config.
        audio_max = np.max(np.abs(audio))
        if audio_max == 0:
            audio_max = 1.0

        volume_high = audio_max * volume_high_ratio  # e.g., 50% of max audio when silent
        volume_low = audio_max * volume_low_ratio    # e.g., 10% of max audio during speech

        # Detect voice activity using LUFS (perceived loudness)
        voice_mask, lufs_envelope = self._detect_voice_activity_lufs(
            audio, sampling_rate, threshold=-40, window_seconds=0.4
        )

        # Apply additional smoothing for gradual transitions
        voice_envelope = self._smooth_envelope(lufs_envelope, sampling_rate, window_seconds=smoothing_window)

        # Create music volume envelope (inverse of voice activity)
        music_gain = volume_high - (voice_envelope * (volume_high - volume_low))

        # Load random music
        music = self._load_random_music(music_folder, len(audio), sampling_rate)

        # Normalize music to unit amplitude before applying gain
        music_max = np.max(np.abs(music))
        if music_max > 0:
            music = music / music_max

        # Apply dynamic gain to music
        music_ducked = music * music_gain

        # Mix with original audio
        output = audio + music_ducked

        # Normalize to prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val * 0.95

        return output
