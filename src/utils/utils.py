import json
import logging
import os
from typing import Dict
import librosa
import numpy as np
from scipy.signal import resample_poly


logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.json") -> Dict:
    """
    Loads a JSON configuration file.

    Args:
        config_path (str): The path to the configuration file. Defaults to "config.json".

    Returns:
        Dict: The loaded configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file is not found at the specified path.
        ValueError: If the file cannot be decoded as JSON.
        IOError: If there's an error reading the file (e.g., permissions).
    """
    if not os.path.exists(config_path):
        error_msg = f"Configuration file not found: {config_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)
        logger.info(f"Successfully loaded configuration from: {config_path}")
        return config_data
    except json.JSONDecodeError as e:
        error_msg = f"Error decoding JSON from configuration file {config_path}: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e
    except IOError as e:
        error_msg = f"An I/O error occurred while reading {config_path}: {e}"
        logger.error(error_msg)
        raise IOError(error_msg) from e

def load_audio(file_path, target_sr=None, mono=True):
    """
    Load an audio file and resample it to the specified sampling rate.

    Args:
        file_path (str): Path to the audio file.
        target_sr (int, optional): Target sampling rate for the audio. If None, the original sampling rate is used.
        mono (bool, optional): If True, the audio is converted to mono. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - y (np.ndarray): The audio signal as a NumPy array.
            - sr (int): The sampling rate of the loaded audio.
    """
    y, sr = librosa.load(file_path, sr=target_sr, mono=mono)
    return y, sr


def snr(signal, noisy_signal):
    """
    Calculate the Signal-to-Noise Ratio (SNR) in decibels (dB) between a clean signal
    and a noisy signal.

    Args:
        signal (np.ndarray): The original clean signal.
        noisy_signal (np.ndarray): The noisy version of the clean signal.

    Returns:
        float: The SNR value in decibels (dB).
    """
    if signal.shape != noisy_signal.shape:
        min_len = np.min([len(signal), len(noisy_signal)])
        signal = signal[:min_len]
        noisy_signal = noisy_signal[:min_len]

    noise = noisy_signal - signal

    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)

    if noise_power == 0:
        return np.inf

    snr = 10 * np.log10(signal_power / noise_power)

    return snr

def resample_audio(audio, input_sr, target_sr):
    """
    Resamples an audio signal from input_sr to target_sr using polyphase filtering.

    Args:
        audio (np.ndarray): The input audio signal.
        input_sr (int): The original sampling rate of the audio.
        target_sr (int, optional): The target sampling rate. Default is 48000 Hz.

    Returns:
        np.ndarray: The resampled audio signal.
    """
    if input_sr != target_sr:
        gcd = np.gcd(input_sr, target_sr)
        up = target_sr // gcd
        down = input_sr // gcd
        audio = resample_poly(audio, up, down)
    
    return audio

def renormalize_audio(original_audio: np.ndarray, processed_audio: np.ndarray) -> np.ndarray:
    """
    Renormalizes processed_audio to match the min and max range of original_audio.

    Args:
        original_audio (np.ndarray): The original input audio signal.
        processed_audio (np.ndarray): The model output (quieter audio).

    Returns:
        np.ndarray: The renormalized audio.
    """
    # Get min and max of the original and processed audio
    orig_min, orig_max = original_audio.min(), original_audio.max()
    proc_min, proc_max = processed_audio.min(), processed_audio.max()

    # Avoid division by zero if processed audio is silent
    if proc_max - proc_min == 0:
        return processed_audio  # Return as-is if it's completely silent
    
    # Scale processed audio to match original range
    renormalized_audio = (processed_audio - proc_min) / (proc_max - proc_min)  # Normalize to [0, 1]
    renormalized_audio = renormalized_audio * (orig_max - orig_min) + orig_min  # Scale to original range
    
    return renormalized_audio