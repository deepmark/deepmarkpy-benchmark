import logging

import librosa
import numpy as np
from pystoi import stoi
from pesq import pesq

from visqol import VisqolApi

logger = logging.getLogger(__name__)


def trim_audio_to_match(audio1: np.ndarray, audio2: np.ndarray) -> tuple:
    """
    Trim the longer audio to match the length of the shorter one.
    Args:
        audio1 (np.ndarray): First audio signal
        audio2 (np.ndarray): Second audio signal
    
    Returns:
        tuple: (trimmed_audio1, trimmed_audio2) - both with matching lengths
    """
    len1 = len(audio1)
    len2 = len(audio2)
    
    if len1 == len2:
        return audio1, audio2
    
    if len1 > len2:
        samples_trimmed = len1 - len2
        logger.debug(f"Trimming audio1: {len1} → {len2} samples (removed {samples_trimmed} samples)")
        return audio1[:len2], audio2
    else:
        samples_trimmed = len2 - len1
        logger.debug(f"Trimming audio2: {len2} → {len1} samples (removed {samples_trimmed} samples)")
        return audio1, audio2[:len1]


def psnr(original: np.ndarray, watermarked: np.ndarray, max_value: float = 1.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between original and watermarked audio.
        
    Args:
        original: Original audio signal
        watermarked: Watermarked audio signal
        max_value: Maximum possible value in the signal (default: 1.0 for normalized audio)
        
    Returns:
        PSNR value in dB
    """
    original, watermarked = trim_audio_to_match(original, watermarked)
    mse = np.mean((original - watermarked) ** 2)
    if mse == 0:
        return float('inf')
    
    return 10 * np.log10((max_value ** 2) / mse)


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """
    Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
        
    Args:
        reference: Reference (original) signal
        estimate: Estimated (watermarked) signal
        
    Returns:
        SI-SDR value in dB
    """
    # Ensure signals are 1D
    reference, estimate = trim_audio_to_match(reference, estimate) 
    reference = reference.flatten()
    estimate = estimate.flatten()
        
    # Zero-mean normalization
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
        
    # Calculate SI-SDR
    alpha = np.dot(estimate, reference) / (np.linalg.norm(reference) ** 2 + 1e-8)
    projection = alpha * reference
    noise = estimate - projection
        
    si_sdr_value = 10 * np.log10(
        (np.linalg.norm(projection) ** 2) / (np.linalg.norm(noise) ** 2 + 1e-8)
    )
        
    return si_sdr_value
    

def stoi_wrapper(reference: np.ndarray, degraded: np.ndarray,
                    fs: int = 16000) -> float:
        """
        Simplified Short-Time Objective Intelligibility (STOI) implementation.
        Args:
            reference: Clean reference signal
            degraded: Degraded signal
            fs: Sampling frequency (Hz)

        Returns:
            STOI score (0-1, higher is better), or None if calculation fails
        """
        reference, degraded = trim_audio_to_match(reference, degraded)

        # STOI requires minimum audio length
        min_samples = fs // 4
        if len(reference) < min_samples or len(degraded) < min_samples:
            logger.warning(f"STOI: Audio too short ({len(reference)} samples), skipping")
            return None

        try:
            return stoi(reference, degraded, fs)
        except Exception as e:
            logger.warning(f"STOI calculation failed: {e}")
            return None



def pesq_wrapper(reference: np.ndarray, degraded: np.ndarray,
                     fs: int = 16000, mode: str = 'wb') -> float:
    """
    PESQ - Perceptual Evaluation of Speech Quality.

    Args:
        reference: Reference signal
        degraded: Degraded signal
        fs: Sampling rate (8000 or 16000 Hz)
        mode: 'wb' (wideband) for 16kHz or 'nb' (narrowband) for 8kHz

    Returns:
        PESQ score (narrowband: 0.5-4.5, wideband: 1.0-4.5), or None if calculation fails
    """
    if fs not in [8000, 16000]:
        return None
    reference, degraded = trim_audio_to_match(reference, degraded)

    # PESQ requires minimum audio length (roughly 0.25 seconds)
    min_samples = fs // 4
    if len(reference) < min_samples or len(degraded) < min_samples:
        logger.warning(f"PESQ: Audio too short ({len(reference)} samples), skipping")
        return None

    try:
        return pesq(fs, reference, degraded, mode)
    except Exception as e:
        logger.warning(f"PESQ calculation failed: {e}")
        return None

def visqol_wrapper(reference: np.ndarray, degraded: np.ndarray,
                   fs: int = 16000) -> float:
    """
    ViSQOL - Virtual Speech Quality Objective Listener.

    Args:
        reference: Reference signal
        degraded: Degraded signal
        fs: Sampling rate

    Returns:
        ViSQOL MOS score (1.0-5.0), or None if calculation fails
    """
    reference, degraded = trim_audio_to_match(reference, degraded)

    try:
        api = VisqolApi()
        if fs >= 48000:
            api.create(mode="audio")
        else:
            api.create(mode="speech")
        result = api.measure_from_arrays(reference, degraded, fs)
        return result.moslqo
    except Exception as e:
        logger.warning(f"ViSQOL calculation failed: {e}")
        return None


def sii(reference: np.ndarray, degraded: np.ndarray,
        fs: int = 16000, n_bands: int = 20) -> float:
    """
    Speech Intelligibility Index (SII) based on ANSI S3.5-1997.

    Estimates speech intelligibility by computing band-level SNR
    weighted by perceptual importance.

    Args:
        reference: Reference (clean) signal
        degraded: Degraded signal
        fs: Sampling rate
        n_bands: Number of frequency bands

    Returns:
        SII score (0.0-1.0, higher is better), or None if calculation fails
    """
    reference, degraded = trim_audio_to_match(reference, degraded)

    try:
        n_fft = 2048
        ref_spec = np.abs(np.fft.rfft(reference, n=n_fft)) ** 2
        deg_spec = np.abs(np.fft.rfft(degraded, n=n_fft)) ** 2

        freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)

        # Critical band center frequencies (ANSI S3.5 simplified)
        min_freq = 150.0
        max_freq = min(fs / 2.0, 8500.0)
        if max_freq <= min_freq:
            logger.warning(f"SII: Sampling rate too low ({fs} Hz), skipping")
            return None

        band_centers = np.logspace(
            np.log10(min_freq), np.log10(max_freq), n_bands
        )
        band_edges = np.sqrt(band_centers[:-1] * band_centers[1:])
        band_edges = np.concatenate([[min_freq], band_edges, [max_freq]])

        # Importance weights (approximate equal weighting, normalized)
        weights = np.ones(n_bands) / n_bands

        sii_val = 0.0
        for i in range(n_bands):
            mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
            if not np.any(mask):
                continue

            signal_power = np.mean(ref_spec[mask])
            noise = deg_spec[mask] - ref_spec[mask]
            noise_power = np.mean(np.maximum(noise, 0) + 1e-12)

            band_snr = 10.0 * np.log10(signal_power / (noise_power + 1e-12))
            band_snr_clamped = np.clip(band_snr, -15.0, 15.0)
            audibility = (band_snr_clamped + 15.0) / 30.0

            sii_val += weights[i] * audibility

        return float(np.clip(sii_val, 0.0, 1.0))
    except Exception as e:
        logger.warning(f"SII calculation failed: {e}")
        return None


def mcd(reference: np.ndarray, degraded: np.ndarray,
        sr: int = 16000, n_mfcc: int = 13) -> float:
    """
    Mel Cepstral Distortion (MCD) between reference and degraded audio.

    Args:
        reference: Reference signal
        degraded: Degraded signal
        sr: Sampling rate
        n_mfcc: Number of MFCC coefficients

    Returns:
        MCD value in dB (lower is better), or None if calculation fails
    """
    reference, degraded = trim_audio_to_match(reference, degraded)

    try:
        mfcc_ref = librosa.feature.mfcc(y=reference, sr=sr, n_mfcc=n_mfcc)
        mfcc_deg = librosa.feature.mfcc(y=degraded, sr=sr, n_mfcc=n_mfcc)
        min_len = min(mfcc_ref.shape[1], mfcc_deg.shape[1])
        diff = mfcc_ref[:, :min_len] - mfcc_deg[:, :min_len]
        return float(np.mean(np.sqrt(np.sum(diff ** 2, axis=0))))
    except Exception as e:
        logger.warning(f"MCD calculation failed: {e}")
        return None


def ncm(reference: np.ndarray, degraded: np.ndarray,
        fs: int = 16000, n_bands: int = 20) -> float:
    """
    Normalized Covariance Metric (NCM) for speech intelligibility.

    Evaluates intelligibility by computing the normalized covariance
    between clean and processed speech in frequency bands, weighted
    by band importance.

    Args:
        reference: Reference (clean) signal
        degraded: Degraded signal
        fs: Sampling rate
        n_bands: Number of frequency bands

    Returns:
        NCM score (0.0-1.0, higher is better), or None if calculation fails
    """
    reference, degraded = trim_audio_to_match(reference, degraded)

    try:
        n_fft = 2048
        hop_length = n_fft // 2

        ref_stft = librosa.stft(reference, n_fft=n_fft, hop_length=hop_length)
        deg_stft = librosa.stft(degraded, n_fft=n_fft, hop_length=hop_length)

        freqs = librosa.fft_frequencies(sr=fs, n_fft=n_fft)

        min_freq = 100.0
        max_freq = min(fs / 2.0, 8000.0)
        if max_freq <= min_freq:
            logger.warning(f"NCM: Sampling rate too low ({fs} Hz), skipping")
            return None

        band_edges = np.logspace(
            np.log10(min_freq), np.log10(max_freq), n_bands + 1
        )

        weights = np.ones(n_bands) / n_bands
        ncm_val = 0.0

        for i in range(n_bands):
            mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
            if not np.any(mask):
                continue

            ref_band = np.abs(ref_stft[mask, :]).flatten()
            deg_band = np.abs(deg_stft[mask, :]).flatten()

            ref_std = np.std(ref_band)
            deg_std = np.std(deg_band)

            if ref_std < 1e-12 or deg_std < 1e-12:
                continue

            cov = np.mean((ref_band - np.mean(ref_band)) * (deg_band - np.mean(deg_band)))
            norm_cov = cov / (ref_std * deg_std)
            norm_cov = np.clip(norm_cov, 0.0, 1.0)

            ncm_val += weights[i] * norm_cov

        return float(np.clip(ncm_val, 0.0, 1.0))
    except Exception as e:
        logger.warning(f"NCM calculation failed: {e}")
        return None


def compute_metrics(reference, degraded, sr):
    """
    Compute audio quality and speech intelligibility metrics.

    Args:
        reference: Original clean audio signal
        degraded: Degraded audio signal
        sr: Sampling rate

    Returns:
        dict with keys:
            Quality: pesq, psnr, si_sdr, mcd, visqol (optional)
            Intelligibility: stoi, sii
    """
    ref_trimmed, deg_trimmed = trim_audio_to_match(reference, degraded)

    # PESQ/STOI require 8kHz or 16kHz
    metrics_sr = 16000 if sr not in [8000, 16000] else sr
    if metrics_sr != sr:
        ref_resampled = librosa.resample(ref_trimmed, orig_sr=sr, target_sr=metrics_sr)
        deg_resampled = librosa.resample(deg_trimmed, orig_sr=sr, target_sr=metrics_sr)
    else:
        ref_resampled = ref_trimmed
        deg_resampled = deg_trimmed

    result = {
        # Quality metrics
        "pesq": pesq_wrapper(ref_resampled, deg_resampled, metrics_sr, 'wb'),
        "psnr": psnr(ref_trimmed, deg_trimmed),
        "si_sdr": si_sdr(ref_trimmed, deg_trimmed),
        "mcd": mcd(ref_trimmed, deg_trimmed, sr),
        "visqol": visqol_wrapper(ref_trimmed, deg_trimmed, sr),
        # Speech intelligibility metrics
        "stoi": stoi_wrapper(ref_resampled, deg_resampled, metrics_sr),
        "sii": sii(ref_resampled, deg_resampled, metrics_sr),
        "ncm": ncm(ref_resampled, deg_resampled, metrics_sr),
    }

    return result
