"""Core routine for the fast replacement attack (Replacement2).

Same idea as the original :class:`ReplacementAttack` (Kirovski et al., 2007):
every short block of the signal is replaced by a least-squares combination of
other, spectrally similar blocks, which scrambles the watermark while
preserving perception. This variant only changes *how* the similar blocks are
found and how the spectra are computed, so that it scales to long files.

The original implementation is ``O(N^2)`` in the number of blocks ``N`` and,
worse, spends that budget in a pure-Python double loop calling
``distance_function`` once per block pair. Since ``N`` grows linearly with the
file length, doubling the audio roughly quadruples the runtime. Every
optimisation below attacks either the constant factor or the asymptotics:

  1. **BLAS distance matrix** -- the per-block Python loop that computed
     ``||abs(a) - abs(b)||`` one candidate at a time is replaced by the
     squared-distance identity ``||a||^2 + ||b||^2 - 2 a.b``, so all distances
     for a tile of query blocks come from a single matrix multiply. Same
     distances, vectorised by BLAS. This is the big, behaviour-preserving win.

  2. **Tiling** -- query blocks are processed in tiles so the distance
     sub-matrix is ``tile x N`` (or ``tile x window``) rather than ``N x N``,
     bounding peak memory on long signals. No effect on the result.

  3. **Pre-computed masking thresholds** -- when psychoacoustic masking is on,
     each block's masking threshold is computed once (``O(N)``) instead of
     re-deriving it for both operands on every one of the ``O(N^2)``
     comparisons.

  4. **Real, half-spectrum FFT** -- ``rfft``/``irfft`` over the
     ``block_size/2 + 1`` non-redundant bins replaces the full complex FFT,
     halving the work in the FFT, the distance matmul and the least squares.

  5. **Vectorised analysis** -- all per-block FFTs are computed in one batched
     ``rfft`` over a strided view instead of a Python loop.

  6. **Optional search window** -- restrict candidates to +/- a time window,
     turning the ``O(N^2)`` search into ``O(N*W)``. Off by default; when set it
     makes the attack faster but slightly weaker (distant similar blocks are no
     longer eligible).

  7. **Optional search-dimension reduction** -- rank candidates using only the
     first ``search_dims`` magnitude bins (which carry most speech energy). Off
     by default; when set it shrinks the matmul at a small fidelity cost in
     *which* blocks are deemed similar.

  8. **Bounded candidate set (k)** -- the least-squares solve uses at most ``k``
     similar blocks, bounding the cost of the per-block solve.

With windowing and dim-reduction off (the defaults), the candidate set and the
replacement match the original up to floating-point rounding in the distance
computation and the switch to the (equivalent) half magnitude spectrum -- only
much faster.
"""

import logging

import numpy as np
from tqdm import tqdm

from plugins.attacks.replacement.psychoacoustic_model import PsychoacousticModel

logger = logging.getLogger(__name__)


def signal_analysis(x, block_size, hop_size):
    """Batched STFT analysis using a real FFT.

    Parameters:
        x (np.ndarray): Input signal (1D array).
        block_size (int): Size of each block (must be even).
        hop_size (int): Step size between consecutive blocks.

    Returns:
        np.ndarray: Complex STFT coefficients of shape
            ``(N_blocks, block_size // 2 + 1)``.
    """
    N_blocks = np.ceil((len(x) - block_size) / hop_size).astype(np.int64) + 1

    padded_length = N_blocks * hop_size + block_size
    x = np.pad(x, (0, padded_length - len(x)), mode="constant")
    window = np.hanning(block_size + 2)[1:-1]

    # One strided view over all frames, then a single batched rfft (opt. 5).
    starts = np.arange(N_blocks) * hop_size
    frames = np.lib.stride_tricks.sliding_window_view(x, block_size)[starts]
    coeffs = np.fft.rfft(frames * window, axis=1)
    return coeffs


def signal_synthesis(coeffs, block_size, hop_size):
    """Inverse STFT synthesis (overlap-add) matching :func:`signal_analysis`.

    Parameters:
        coeffs (np.ndarray): Complex STFT coefficients (real-FFT layout).
        block_size (int): Size of each block (must be even).
        hop_size (int): Step size between consecutive blocks.

    Returns:
        np.ndarray: Reconstructed signal.
    """
    N_blocks = coeffs.shape[0]
    window = np.hanning(block_size + 2)[1:-1]

    blocks = np.fft.irfft(coeffs, n=block_size, axis=1) * window

    signal_length = (N_blocks - 1) * hop_size + block_size
    y = np.zeros(signal_length)
    norm_factor = np.zeros(signal_length)  # for overlap-add normalization
    win_sq = window**2
    for m in range(N_blocks):
        start = m * hop_size
        y[start : start + block_size] += blocks[m]
        norm_factor[start : start + block_size] += win_sq

    y /= norm_factor
    return y


def _least_squares(block, similar_blocks):
    """Least-squares projection of ``block`` onto the span of the candidates.

    Solves ``min_c || A c - block ||`` for ``A = similar_blocks.T`` and returns
    the fitted ``A c``. Instead of an SVD-based ``lstsq`` on the tall ``(F, num)``
    system, this solves the small ``(num, num)`` normal equations
    ``(A^H A) c = A^H block`` directly. The fitted vector ``A c`` (all we use) is
    identical to the ``lstsq`` projection, but the solve is on a matrix the size
    of the candidate count -- far cheaper, and it avoids the pathologically slow
    SVD path some BLAS builds take for tall complex systems. A tiny ridge term
    keeps the solve stable when candidates are near-collinear; if the system is
    still singular we fall back to ``lstsq``.

    Parameters:
        block (np.ndarray): Reference spectrum (1D complex array).
        similar_blocks (np.ndarray): Candidate spectra, shape ``(num, F)``.

    Returns:
        np.ndarray: The replacement spectrum.
    """
    A = similar_blocks.T  # (F, num)
    gram = A.conj().T @ A  # (num, num)
    rhs = A.conj().T @ block
    n = gram.shape[0]
    # Scale-aware ridge for numerical stability on collinear candidates.
    ridge = 1e-10 * (np.trace(gram).real / n if n else 0.0)
    gram[np.diag_indices(n)] += ridge
    try:
        coeffs = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(A, block, rcond=None)[0]
    return A @ coeffs


def replacement2_attack(
    x,
    sampling_rate=44100,
    block_size=1024,
    overlap_factor=0.75,
    lower_bound=0,
    upper_bound=10,
    k=100,
    use_masking=False,
    search_window_sec=0.0,
    search_dims=0,
    tile_size=256,
):
    """Perform a fast replacement attack on an audio signal.

    Drop-in, faster reimplementation of
    :func:`plugins.attacks.replacement.replacement_attack.replacement_attack`.
    See the module docstring for the optimisations applied.

    Parameters:
        x (np.ndarray): The input audio signal.
        sampling_rate (int): Sampling rate of the audio signal in Hz.
        block_size (int): Size of each block for processing.
        overlap_factor (float): Overlap factor between consecutive blocks,
            in ``[0, 1)``.
        lower_bound (float): Lower bound of the similarity distance for a block
            to be considered a candidate.
        upper_bound (float): Upper bound of the similarity distance for a block
            to be considered a candidate.
        k (int): Maximum number of similar blocks used in the least-squares
            approximation.
        use_masking (bool): Whether to use psychoacoustic masking when ranking
            candidate similarity.
        search_window_sec (float): If > 0, only blocks within +/- this many
            seconds are eligible candidates (turns the search into ``O(N*W)``).
        search_dims (int): If > 0, rank candidates using only the first
            ``search_dims`` magnitude bins.
        tile_size (int): Number of query blocks processed per tile (memory vs.
            speed trade-off; does not affect the result).

    Returns:
        np.ndarray: The processed audio signal with the replacement attack
            applied, truncated to the original length.
    """
    if block_size % 2 != 0:
        raise ValueError("block_size must be even (required by rfft).")
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0.")
    if k <= 0:
        raise ValueError("k must be > 0.")

    overlap = int(overlap_factor * block_size)
    hop_size = block_size - overlap
    if hop_size <= 0:
        raise ValueError("overlap_factor too large: hop_size must be > 0.")

    coeffs = signal_analysis(x, block_size, hop_size)  # (N, F) complex
    N = coeffs.shape[0]
    if N == 0:
        return x.copy()

    magnitudes = np.abs(coeffs)  # (N, F)

    # Opt. 3: compute each block's masking threshold once, then gate the
    # magnitudes so the distance stays a plain Euclidean norm (BLAS-friendly).
    masking_model = None
    if use_masking:
        masking_model = PsychoacousticModel(
            N=block_size, fs=sampling_rate, nfilts=24
        )
        thresholds = np.empty_like(magnitudes)
        for i in range(N):
            thresholds[i] = masking_model.maskingThreshold(magnitudes[i])
        work = magnitudes * (magnitudes > thresholds)
    else:
        work = magnitudes

    # Opt. 7: optionally rank on a reduced number of (low-frequency) bins.
    if search_dims and search_dims < work.shape[1]:
        work = np.ascontiguousarray(work[:, :search_dims])

    sq_norms = np.einsum("ij,ij->i", work, work)  # ||w_j||^2 per block

    guard = block_size // hop_size  # blocks overlapping block i in time
    window_blocks = (
        int(search_window_sec * sampling_rate / hop_size)
        if search_window_sec
        else 0
    )

    processed = np.empty_like(coeffs)
    cnt_replaced = 0

    # Opt. 1 + 2: distances for a whole tile of queries against the candidate
    # band come from one matmul, never the per-pair Python loop of the original.
    for q0 in tqdm(
        range(0, N, tile_size), desc="Replacement2 attack", unit="tile"
    ):
        qend = min(q0 + tile_size, N)
        queries = work[q0:qend]
        sq_queries = sq_norms[q0:qend]

        if window_blocks:
            lo = max(0, q0 - window_blocks)
            hi = min(N, qend + window_blocks)
        else:
            lo, hi = 0, N
        cand = work[lo:hi]
        sq_cand = sq_norms[lo:hi]

        gram = queries @ cand.T  # (tile, hi - lo)
        d2 = sq_queries[:, None] + sq_cand[None, :] - 2.0 * gram
        np.maximum(d2, 0.0, out=d2)
        dist = np.sqrt(d2)

        for r in range(qend - q0):
            i = q0 + r
            drow = dist[r].copy()

            # Exclude the block itself and its time-overlapping neighbours.
            local_i = i - lo
            drow[max(0, local_i - guard) : local_i + guard + 1] = np.inf

            if not np.isfinite(drow).any():
                processed[i] = coeffs[i]
                continue

            most_similar_local = int(np.argmin(drow))
            best_dist = drow[most_similar_local]

            valid = np.nonzero((drow >= lower_bound) & (drow <= upper_bound))[0]
            if valid.size == 0:
                processed[i] = coeffs[i]
                continue
            valid = valid[:k]  # opt. 8: bound the least-squares system

            block = coeffs[i]
            similar = coeffs[lo + valid]
            replacement = _least_squares(block, similar)

            # Accept the replacement only if it is at least as close as the
            # single best matching block (mirrors the original guard).
            repl_mag = np.abs(replacement)
            if masking_model is not None:
                repl_mag = repl_mag * (
                    repl_mag > masking_model.maskingThreshold(repl_mag)
                )
            if search_dims and search_dims < repl_mag.size:
                repl_mag = repl_mag[:search_dims]
            repl_dist = np.linalg.norm(work[i] - repl_mag)

            if repl_dist > best_dist:
                processed[i] = block
            else:
                processed[i] = replacement
                cnt_replaced += 1

    logger.info(f"Replaced:{(cnt_replaced / N * 100):.2f}% of blocks.")
    return signal_synthesis(processed, block_size, hop_size)[: len(x)]
