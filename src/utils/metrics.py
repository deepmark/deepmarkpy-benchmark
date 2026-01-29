import numpy as np
from pystoi import stoi
from pesq import pesq


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
        print(f"Trimming audio1: {len1} → {len2} samples (removed {samples_trimmed} samples)")
        return audio1[:len2], audio2
    else:
        samples_trimmed = len2 - len1
        print(f"Trimming audio2: {len2} → {len1} samples (removed {samples_trimmed} samples)")
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
            STOI score (0-1, higher is better)
        """
        reference, degraded = trim_audio_to_match(reference, degraded)
        return stoi(reference, degraded, fs)



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
        PESQ score (narrowband: 0.5-4.5, wideband: 1.0-4.5)
    """
    reference, degraded = trim_audio_to_match(reference, degraded)
    return pesq(fs, reference, degraded, mode)
