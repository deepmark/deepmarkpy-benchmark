import functools
import logging
import os
import tempfile
from typing import Dict, Iterable, Optional

import librosa
import numpy as np
import soundfile as sf
from pystoi import stoi
from pesq import pesq

logger = logging.getLogger(__name__)


def _safe_metric(name: str):
    """Decorator for metric functions added in this branch.

    Applies the shared trim + narrow-exception pattern so each new
    wrapper can focus on the actual computation. Wrappers that existed
    before this branch keep their inline try/except so the change is
    scoped to new code only.
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(reference, degraded, *args, **kwargs):
            reference, degraded = trim_audio_to_match(reference, degraded)
            try:
                return fn(reference, degraded, *args, **kwargs)
            except (RuntimeError, ValueError, IndexError) as e:
                logger.warning(f"{name} calculation failed: {e}")
                return None
        return wrapper
    return deco


# ANSI S3.5-1997 Table B.2: 1/3-octave band center frequencies (Hz) and
# corresponding band importance weights for average speech. SII and NCM
# share this band set so their frequency resolution is identical.
_ANSI_BAND_CENTERS_HZ = np.array([
    160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0,
    1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0,
    5000.0, 6300.0, 8000.0,
])
_ANSI_BAND_IMPORTANCE = np.array([
    0.0083, 0.0095, 0.0150, 0.0289, 0.0440, 0.0578, 0.0653, 0.0711,
    0.0818, 0.0844, 0.0882, 0.0898, 0.0868, 0.0844, 0.0771,
    0.0527, 0.0364, 0.0185,
])

# 1/3-octave factor: edges at center * 2^(±1/6)
_THIRD_OCTAVE_FACTOR = 2.0 ** (1.0 / 6.0)

# STFT parameters shared by SII and NCM. 2048 gives ~4 Hz resolution at
# 8 kHz which is enough to isolate 1/3-octave bands at the low end.
_SPECTRAL_N_FFT = 2048
_SPECTRAL_HOP_LENGTH = _SPECTRAL_N_FFT // 2

# SII audibility mapping per ANSI S3.5-1997 maps a -15..+15 dB SNR range
# linearly to the [0, 1] audibility range (30 dB dynamic range).
_SII_SNR_CLAMP_DB = 15.0
_SII_DYNAMIC_RANGE_DB = 2.0 * _SII_SNR_CLAMP_DB  # 30 dB

# Small numerical floor to prevent log10(0) and divide-by-zero.
_EPSILON = 1e-12

# Minimum audio duration (in seconds) required by pystoi/pesq. Shorter
# clips cause the underlying C libraries to raise; we bail out early.
_MIN_METRIC_DURATION_S = 0.25


def _check_min_length(reference: np.ndarray, degraded: np.ndarray,
                      fs: int, metric_name: str) -> bool:
    """Return True if both signals meet the minimum-length requirement.

    Logs a warning and returns False otherwise so callers can skip the
    computation instead of crashing inside the underlying C extension.
    """
    min_samples = int(fs * _MIN_METRIC_DURATION_S)
    if len(reference) < min_samples or len(degraded) < min_samples:
        logger.warning(
            f"{metric_name}: Audio too short ({len(reference)} samples), skipping"
        )
        return False
    return True


def _ansi_bands(fs: int):
    """Return ANSI 1/3-octave band edges, centers, and importance weights
    that fit within the Nyquist limit for the given sampling rate.

    Bands whose upper edge exceeds fs/2 are dropped; importance weights
    are renormalized over the surviving bands so they sum to 1.
    """
    nyquist = fs / 2.0
    lower = _ANSI_BAND_CENTERS_HZ / _THIRD_OCTAVE_FACTOR
    upper = _ANSI_BAND_CENTERS_HZ * _THIRD_OCTAVE_FACTOR
    valid = upper <= nyquist
    centers = _ANSI_BAND_CENTERS_HZ[valid]
    weights = _ANSI_BAND_IMPORTANCE[valid]
    if weights.size == 0:
        return None, None, None, None
    weights = weights / weights.sum()
    return lower[valid], upper[valid], centers, weights


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


# SI-SDR uses a larger epsilon than the shared _EPSILON because its
# denominator is a squared L2-norm of the reference/noise. 1e-8 matches
# the convention from the original SI-SDR paper (Le Roux et al., 2019).
_SI_SDR_EPSILON = 1e-8


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """
    Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).

    Args:
        reference: Reference (original) signal
        estimate: Estimated (watermarked) signal

    Returns:
        SI-SDR value in dB
    """
    reference, estimate = trim_audio_to_match(reference, estimate)
    reference = reference.flatten()
    estimate = estimate.flatten()

    # Zero-mean normalization
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)

    alpha = np.dot(estimate, reference) / (np.linalg.norm(reference) ** 2 + _SI_SDR_EPSILON)
    projection = alpha * reference
    noise = estimate - projection

    return 10 * np.log10(
        (np.linalg.norm(projection) ** 2) / (np.linalg.norm(noise) ** 2 + _SI_SDR_EPSILON)
    )
    

def stoi_wrapper(reference: np.ndarray, degraded: np.ndarray,
                 fs: int = 16000) -> float:
    """
    Short-Time Objective Intelligibility (STOI) wrapper.

    Args:
        reference: Clean reference signal
        degraded: Degraded signal
        fs: Sampling frequency (Hz)

    Returns:
        STOI score (0-1, higher is better), or None if calculation fails
    """
    reference, degraded = trim_audio_to_match(reference, degraded)

    if not _check_min_length(reference, degraded, fs, "STOI"):
        return None

    try:
        return stoi(reference, degraded, fs)
    except (RuntimeError, ValueError, IndexError) as e:
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
        PESQ score (narrowband: -0.5-4.5, wideband: 1.0-4.66), or None if calculation fails
    """
    if fs not in (8000, 16000):
        logger.warning(f"PESQ: Unsupported sampling rate ({fs} Hz), skipping")
        return None
    reference, degraded = trim_audio_to_match(reference, degraded)

    if not _check_min_length(reference, degraded, fs, "PESQ"):
        return None

    try:
        return pesq(fs, reference, degraded, mode)
    except (RuntimeError, ValueError, IndexError) as e:
        logger.warning(f"PESQ calculation failed: {e}")
        return None

_visqol_cache = {}
_visqol_unavailable = False


def _get_visqol_api(mode: str):
    """Return a cached VisqolApi instance for the given mode.

    The ``visqol`` package is imported lazily so benchmarks and other
    metrics keep working even when ViSQOL is not installed. Returns
    ``None`` when the package is unavailable.
    """
    global _visqol_unavailable

    if _visqol_unavailable:
        return None
    if mode in _visqol_cache:
        return _visqol_cache[mode]

    try:
        from visqol import VisqolApi
    except ImportError:
        logger.info(
            "visqol package not installed; ViSQOL scores will be skipped. "
            "Install it to enable this metric."
        )
        _visqol_unavailable = True
        return None

    api = VisqolApi()
    api.create(mode=mode)
    _visqol_cache[mode] = api
    return api


@_safe_metric("ViSQOL")
def visqol_wrapper(reference: np.ndarray, degraded: np.ndarray,
                   fs: int = 16000) -> Optional[float]:
    """
    ViSQOL - Virtual Speech Quality Objective Listener.

    Args:
        reference: Reference signal
        degraded: Degraded signal
        fs: Sampling rate

    Returns:
        ViSQOL MOS score (1.0-5.0), ``None`` if the calculation fails
        or if the optional ``visqol`` package is not installed.
    """
    mode = "audio" if fs >= 48000 else "speech"
    api = _get_visqol_api(mode)
    if api is None:
        return None
    return api.measure_from_arrays(reference, degraded, fs).moslqo


@_safe_metric("SII")
def sii(reference: np.ndarray, degraded: np.ndarray,
        fs: int = 16000) -> Optional[float]:
    """
    Speech Intelligibility Index (SII) following ANSI S3.5-1997.

    Uses the standard 1/3-octave band set (160 Hz-8 kHz) with the ANSI
    Table B.2 band importance function for average speech. The per-band
    SNR is mapped to audibility via the ANSI 30 dB dynamic range
    (-15 dB .. +15 dB), and the weighted sum gives the final index.

    Args:
        reference: Reference (clean) signal
        degraded: Degraded signal
        fs: Sampling rate

    Returns:
        SII score (0.0-1.0, higher is better), or None if calculation fails
    """
    lower, upper, _, weights = _ansi_bands(fs)
    if weights is None:
        logger.warning(f"SII: Sampling rate too low ({fs} Hz), skipping")
        return None

    ref_stft = librosa.stft(reference, n_fft=_SPECTRAL_N_FFT,
                            hop_length=_SPECTRAL_HOP_LENGTH)
    ref_power = np.mean(np.abs(ref_stft) ** 2, axis=1)

    noise_signal = degraded - reference
    noise_stft = librosa.stft(noise_signal, n_fft=_SPECTRAL_N_FFT,
                              hop_length=_SPECTRAL_HOP_LENGTH)
    noise_power_spec = np.mean(np.abs(noise_stft) ** 2, axis=1)

    freqs = librosa.fft_frequencies(sr=fs, n_fft=_SPECTRAL_N_FFT)

    sii_val = 0.0
    for lo, hi, w in zip(lower, upper, weights):
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            continue

        signal_power = np.mean(ref_power[mask])
        noise_power = np.mean(noise_power_spec[mask]) + _EPSILON

        band_snr = 10.0 * np.log10(signal_power / noise_power)
        band_snr_clamped = np.clip(band_snr,
                                   -_SII_SNR_CLAMP_DB, _SII_SNR_CLAMP_DB)
        audibility = (band_snr_clamped + _SII_SNR_CLAMP_DB) / _SII_DYNAMIC_RANGE_DB

        sii_val += w * audibility

    return float(np.clip(sii_val, 0.0, 1.0))


_MCD_FACTOR = 10.0 / np.log(10.0) * np.sqrt(2.0)  # ≈ 6.1413 (ANSI standard)


@_safe_metric("MCD")
def mcd(reference: np.ndarray, degraded: np.ndarray,
        sr: int = 16000, n_mfcc: int = 13) -> Optional[float]:
    """
    Mel Cepstral Distortion (MCD) between reference and degraded audio.

    Uses the standard MCD scaling factor 10/ln(10) * sqrt(2) ≈ 6.1413,
    so values are directly comparable to published results. The DC
    coefficient (MFCC[0]) is excluded, following common MCD practice.

    Uses fixed index-wise alignment (no DTW), so MCD becomes unreliable
    for attacks that alter timing -- time-stretching, pitch-shifting or
    anything that inserts/removes samples -- because frames at the same
    index no longer represent the same phoneme.

    Args:
        reference: Reference signal
        degraded: Degraded signal
        sr: Sampling rate
        n_mfcc: Number of MFCC coefficients

    Returns:
        MCD value in dB (lower is better), or None if calculation fails
    """
    mfcc_ref = librosa.feature.mfcc(y=reference, sr=sr, n_mfcc=n_mfcc)[1:, :]
    mfcc_deg = librosa.feature.mfcc(y=degraded, sr=sr, n_mfcc=n_mfcc)[1:, :]
    min_len = min(mfcc_ref.shape[1], mfcc_deg.shape[1])
    diff = mfcc_ref[:, :min_len] - mfcc_deg[:, :min_len]
    return float(_MCD_FACTOR * np.mean(np.sqrt(np.sum(diff ** 2, axis=0))))


# --- NISQA (non-intrusive MOS prediction) ---------------------------
# NISQA is a deep model. Loading it is expensive (~1s + weights I/O), so
# we cache the model instance and the per-call result so that all five
# NISQA dimensions returned by a single inference (mos/noi/dis/col/loud)
# can be served from one prediction.
_NISQA_WEIGHTS_PATH = os.environ.get(
    "NISQA_WEIGHTS_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "weights", "nisqa.tar"),
)
_nisqa_model = None
_nisqa_unavailable = False


def _get_nisqa_model():
    """Return a cached nisqaModel, or None when unavailable."""
    global _nisqa_model, _nisqa_unavailable
    if _nisqa_unavailable:
        return None
    if _nisqa_model is not None:
        return _nisqa_model
    weights_abs = os.path.abspath(_NISQA_WEIGHTS_PATH)
    if not os.path.exists(weights_abs):
        logger.info(
            f"NISQA weights not found at {weights_abs}; NISQA scores will be "
            f"skipped. Set NISQA_WEIGHTS_PATH or place nisqa.tar there."
        )
        _nisqa_unavailable = True
        return None
    try:
        from nisqa.NISQA_model import nisqaModel
    except ImportError:
        logger.info(
            "nisqa package not installed; NISQA scores will be skipped."
        )
        _nisqa_unavailable = True
        return None
    try:
        import contextlib
        import io
        # The package always prints a yaml dump + 'Loaded pretrained model'
        # banner on init. Silence both so logs stay clean.
        with contextlib.redirect_stdout(io.StringIO()):
            _nisqa_model = nisqaModel({
                "mode": "predict_file",
                "pretrained_model": weights_abs,
                "deg": __file__,  # placeholder, overwritten before each predict
                "tr_bs_val": 1,
                "tr_num_workers": 0,
                "output_dir": None,
                "ms_channel": None,
            })
        return _nisqa_model
    except (RuntimeError, ValueError, FileNotFoundError, ImportError) as e:
        logger.warning(f"NISQA model could not be loaded: {e}")
        _nisqa_unavailable = True
        return None


_NISQA_MAX_SEC = 12.0
_NISQA_CHUNK_SEC = 10.0
_NISQA_MIN_CHUNK_SEC = 3.0


def _nisqa_predict_once(model, degraded: np.ndarray, sr: int
                        ) -> Dict[str, Optional[float]]:
    """Run NISQA on a single segment that fits in one forward pass."""
    none_result = {k: None for k in NISQA_METRICS}
    try:
        import contextlib
        import io
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, degraded, sr)
            model.args["deg"] = tmp_path
            with contextlib.redirect_stdout(io.StringIO()):
                model._loadDatasets()
                df = model.predict()
            row = df.iloc[0]
            return {
                "nisqa_mos": float(row["mos_pred"]),
                "nisqa_noi": float(row["noi_pred"]),
                "nisqa_dis": float(row["dis_pred"]),
                "nisqa_col": float(row["col_pred"]),
                "nisqa_loud": float(row["loud_pred"]),
            }
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except (RuntimeError, ValueError, KeyError) as e:
        logger.warning(f"NISQA prediction failed: {e}")
        return none_result


def compute_nisqa(degraded: np.ndarray, sr: int) -> Dict[str, Optional[float]]:
    """Run NISQA inference and return all five MOS dimensions.

    NISQA is non-intrusive: it scores ``degraded`` alone, no reference
    needed. The five dimensions (mos / noisiness / discontinuity /
    coloration / loudness) are produced together in one forward pass, so
    callers should request them as a group rather than five times.

    NISQA's mel-spec window buffer caps inference at ~13 s; longer signals
    are split into ~10 s chunks here and averaged so callers don't have to
    care about duration.
    """
    none_result = {k: None for k in NISQA_METRICS}
    model = _get_nisqa_model()
    if model is None:
        return none_result

    duration = len(degraded) / sr
    if duration <= _NISQA_MAX_SEC:
        return _nisqa_predict_once(model, degraded, sr)

    chunk_len = int(_NISQA_CHUNK_SEC * sr)
    min_len = int(_NISQA_MIN_CHUNK_SEC * sr)
    accum: Dict[str, list] = {k: [] for k in NISQA_METRICS}
    failed = 0
    for start in range(0, len(degraded), chunk_len):
        chunk = degraded[start:start + chunk_len]
        if len(chunk) < min_len:
            continue
        res = _nisqa_predict_once(model, chunk, sr)
        if res["nisqa_mos"] is None:
            failed += 1
            continue
        for k in NISQA_METRICS:
            accum[k].append(res[k])

    if not accum["nisqa_mos"]:
        logger.warning(
            f"NISQA: no successful chunks across {duration:.1f}s signal."
        )
        return none_result
    if failed:
        logger.info(
            f"NISQA: {failed} chunk(s) failed out of "
            f"{failed + len(accum['nisqa_mos'])} for {duration:.1f}s signal."
        )
    return {k: float(np.mean(accum[k])) for k in NISQA_METRICS}


@_safe_metric("NCM")
def ncm(reference: np.ndarray, degraded: np.ndarray,
        fs: int = 16000) -> Optional[float]:
    """
    Normalized Covariance Metric (NCM) for speech intelligibility.

    Uses the same ANSI S3.5-1997 1/3-octave band set as SII, together
    with the ANSI band importance weights. Per band, the absolute value
    of the normalized covariance between the envelopes of the clean and
    processed magnitude spectra is weighted by importance; `|r|` scores
    sign-inverted envelopes as preserved (matching Loizou's formulation).

    Args:
        reference: Reference (clean) signal
        degraded: Degraded signal
        fs: Sampling rate

    Returns:
        NCM score (0.0-1.0, higher is better), or None if calculation fails
    """
    lower, upper, _, weights = _ansi_bands(fs)
    if weights is None:
        logger.warning(f"NCM: Sampling rate too low ({fs} Hz), skipping")
        return None

    ref_stft = librosa.stft(reference, n_fft=_SPECTRAL_N_FFT,
                            hop_length=_SPECTRAL_HOP_LENGTH)
    deg_stft = librosa.stft(degraded, n_fft=_SPECTRAL_N_FFT,
                            hop_length=_SPECTRAL_HOP_LENGTH)

    freqs = librosa.fft_frequencies(sr=fs, n_fft=_SPECTRAL_N_FFT)

    ncm_val = 0.0
    for lo, hi, w in zip(lower, upper, weights):
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            continue

        ref_band = np.abs(ref_stft[mask, :]).flatten()
        deg_band = np.abs(deg_stft[mask, :]).flatten()

        ref_std = np.std(ref_band)
        deg_std = np.std(deg_band)

        if ref_std < _EPSILON or deg_std < _EPSILON:
            continue

        cov = np.mean((ref_band - np.mean(ref_band)) * (deg_band - np.mean(deg_band)))
        norm_cov = np.clip(np.abs(cov / (ref_std * deg_std)), 0.0, 1.0)

        ncm_val += w * norm_cov

    return float(np.clip(ncm_val, 0.0, 1.0))


NISQA_METRICS = [
    "nisqa_mos", "nisqa_noi", "nisqa_dis", "nisqa_col", "nisqa_loud",
]
QUALITY_METRICS = ["pesq", "psnr", "si_sdr", "mcd", "visqol"] + NISQA_METRICS
INTELLIGIBILITY_METRICS = ["stoi", "sii", "ncm"]
ALL_METRICS = QUALITY_METRICS + INTELLIGIBILITY_METRICS

# Human-readable labels used by report generators when rendering tables.
METRIC_LABELS = {
    "pesq": "PESQ (1--4.66)",
    "psnr": "PSNR (dB)",
    "si_sdr": "SI-SDR (dB)",
    "mcd": "MCD (dB)",
    "visqol": "ViSQOL (1--5)",
    "stoi": "STOI (0--1)",
    "sii": "SII (0--1)",
    "ncm": "NCM (0--1)",
    "nisqa_mos": "MOS (1--5)",
    "nisqa_noi": "Noisiness (1--5)",
    "nisqa_dis": "Discontinuity (1--5)",
    "nisqa_col": "Coloration (1--5)",
    "nisqa_loud": "Loudness (1--5)",
}

# Metrics that require 8 kHz or 16 kHz and are resampled accordingly
_NARROWBAND_METRICS = {"pesq", "stoi"}


def compute_metrics(
    reference: np.ndarray,
    degraded: np.ndarray,
    sr: int,
    metrics: Optional[Iterable[str]] = None,
) -> Dict[str, Optional[float]]:
    """
    Compute audio quality and speech intelligibility metrics.

    Measures the perceptual effect of the combined watermark + attack
    pipeline on the audio signal: ``reference`` is the original clean
    audio, ``degraded`` is the watermarked-then-attacked signal.

    Args:
        reference: Original clean audio signal
        degraded: Degraded audio signal
        sr: Sampling rate of both signals
        metrics: Optional iterable of metric names to compute. Defaults
            to all metrics in ``ALL_METRICS``. Only listed metrics are
            computed; any metric not requested is omitted from the
            result dict (avoids needless compute for grouped reports).

    Returns:
        Dict with the requested metric names as keys. Values that could
        not be computed are ``None``.

    Available metrics:
        Quality: pesq, psnr, si_sdr, mcd, visqol
            (ViSQOL returns ``None`` when the optional ``visqol`` package
            is not installed; see README.)
        Intelligibility: stoi, sii, ncm
    """
    ref_trimmed, deg_trimmed = trim_audio_to_match(reference, degraded)

    requested = set(metrics) if metrics is not None else set(ALL_METRICS)

    # Only pay the resample cost when a metric that needs it is requested
    need_narrowband = bool(requested & _NARROWBAND_METRICS)
    ref_nb = deg_nb = None
    nb_sr = sr
    if need_narrowband:
        nb_sr = 16000 if sr not in (8000, 16000) else sr
        if nb_sr != sr:
            ref_nb = librosa.resample(ref_trimmed, orig_sr=sr, target_sr=nb_sr)
            deg_nb = librosa.resample(deg_trimmed, orig_sr=sr, target_sr=nb_sr)
        else:
            ref_nb, deg_nb = ref_trimmed, deg_trimmed

    # PESQ mode is determined by the sampling rate PESQ is actually run at.
    pesq_mode = "wb" if nb_sr == 16000 else "nb"

    # NISQA returns all 5 dimensions in a single forward pass; cache the
    # result so requesting more than one of them only runs inference once.
    nisqa_cache: Dict[str, Optional[float]] = {}
    def _nisqa(key):
        if not nisqa_cache:
            nisqa_cache.update(compute_nisqa(deg_trimmed, sr))
        return nisqa_cache[key]

    computations = {
        "pesq": lambda: pesq_wrapper(ref_nb, deg_nb, nb_sr, pesq_mode),
        "psnr": lambda: psnr(ref_trimmed, deg_trimmed),
        "si_sdr": lambda: si_sdr(ref_trimmed, deg_trimmed),
        "mcd": lambda: mcd(ref_trimmed, deg_trimmed, sr),
        "visqol": lambda: visqol_wrapper(ref_trimmed, deg_trimmed, sr),
        "stoi": lambda: stoi_wrapper(ref_nb, deg_nb, nb_sr),
        "sii": lambda: sii(ref_trimmed, deg_trimmed, sr),
        "ncm": lambda: ncm(ref_trimmed, deg_trimmed, sr),
        "nisqa_mos": lambda: _nisqa("nisqa_mos"),
        "nisqa_noi": lambda: _nisqa("nisqa_noi"),
        "nisqa_dis": lambda: _nisqa("nisqa_dis"),
        "nisqa_col": lambda: _nisqa("nisqa_col"),
        "nisqa_loud": lambda: _nisqa("nisqa_loud"),
    }

    return {name: computations[name]() for name in ALL_METRICS if name in requested}
