import logging

import numpy as np
from tqdm import tqdm

from deepmarkpy.plugins.attacks.replacement.psychoacoustic_model import PsychoacousticModel

logger = logging.getLogger(__name__)

def signal_analysis(x, block_size, hop_size):
    """
    Perform the Short-Time Fourier Transform (STFT) analysis with given overlap.

    Parameters:
        x (np.ndarray): Input signal (1D array).
        block_size (int): Size of each block (must be even).
        hop_size (int): Hop size, representing the step size between consecutive blocks.

    Returns:
        np.ndarray: STFT coefficients.
    """
    N_blocks = np.ceil((len(x) - block_size) / hop_size).astype(int) + 1

    padded_length = N_blocks * hop_size + block_size
    padding_needed = padded_length - len(x)
    x = np.pad(x, (0, padding_needed), mode="constant")
    window = np.hanning(block_size + 2)[1:-1]
    coeffs = []
    for m in range(N_blocks):
        start = m * hop_size
        block = x[start : start + block_size] * window
        spectrum = np.fft.fft(block)
        coeffs.append(spectrum)

    return np.array(coeffs)


def signal_synthesis(coeffs, block_size, hop_size):
    """
    Perform the inverse short-time Fourier transform (iSTFT) synthesis with given overlap.

    Parameters:
        coeffs (np.ndarray): STFT coefficients.
        block_size (int): Size of each block (must be even).
        hop_size (int): Hop size, representing the step size between consecutive blocks.

    Returns:
        np.ndarray: Reconstructed signal.
    """
    N_blocks = coeffs.shape[0]
    window = np.hanning(block_size + 2)[1:-1]

    signal_length = (N_blocks - 1) * hop_size + block_size
    y = np.zeros(signal_length)
    norm_factor = np.zeros(signal_length)  # for overlap-add normalization

    for m in range(N_blocks):
        spectrum = coeffs[m]
        block = np.real(np.fft.ifft(spectrum))
        block *= window

        start = m * hop_size
        y[start : start + block_size] += block
        norm_factor[start : start + block_size] += window**2

    y /= norm_factor
    return y


def distance_function(block1, block2, masking_model):
    """
    Calculate the distance between two audio blocks, optionally incorporating psychoacoustic masking.

    Parameters:
        block1 (np.ndarray): The first audio block (1D array).
        block2 (np.ndarray): The second audio block (1D array).
        masking_model: A model to apply psychoacoustic masking

    Returns:
        float: The computed distance between the two blocks.
    """

    block1 = np.abs(block1)
    block2 = np.abs(block2)

    mT1, mT2 = 0, 0
    if masking_model is not None:
        block1 = block1[: len(block1) // 2 + 1]
        block2 = block2[: len(block2) // 2 + 1]
        mT1 = masking_model.maskingThreshold(np.abs(block1))
        mT2 = masking_model.maskingThreshold(np.abs(block2))

    return np.linalg.norm(
        block1 * (block1 > mT1) - block2 * (block2 > np.maximum(mT1, mT2))
    )


def find_most_similar_blocks(
    block, blocks, overlap_indices, lower_bound, upper_bound, k, masking_model
):
    """
    Find the most similar blocks to a given block from a list of candidate blocks,
    excluding overlapping indices, based on a distance upper_bound.

    Parameters:
        block (np.ndarray): The reference block to compare against (1D array).
        blocks (list of np.ndarray): A list of candidate blocks for comparison.
        overlap_indices (list of int): Indices of blocks to exclude due to overlap.
        lower_bound (float): The lower bound of the similarity distance for considering a block as a candidate.
        upper_bound (float): The upper bound of the similarity distance for considering a block as a candidate.
        k (int): The maximum number of similar blocks to return.
        masking_model: A model to apply psychoacoustic masking

    Returns:
        tuple:
            - np.ndarray: Array of up to `k` blocks that are similar to the reference block.
            - np.ndarray: The block from the candidates with the smallest distance to the reference block.
    """
    candidates = [b for i, b in enumerate(blocks) if i not in overlap_indices]
    distances = [distance_function(block, b, masking_model) for b in candidates]
    most_similar_idx = np.argmin(distances)
    most_similar_indices = [
        i for i, dist in enumerate(distances) if dist <= upper_bound
    ]
    return np.array([candidates[i] for i in most_similar_indices])[:k], candidates[
        most_similar_idx
    ]


def least_squares_approximation(block, similar_blocks):
    """
    Perform a least-squares approximation of a given block using a set of similar blocks.

    Parameters:
        block (np.ndarray): The reference block to approximate (1D array).
        similar_blocks (list of np.ndarray): List of similar blocks (1D arrays)
                                             used for the approximation.

    Returns:
        np.ndarray: The replacement block obtained by least-squares approximation.
    """
    similar_blocks_matrix = np.vstack(similar_blocks).T
    coeffs = np.linalg.lstsq(similar_blocks_matrix, block, rcond=None)[0]
    replacement_block = similar_blocks_matrix @ coeffs
    return replacement_block


def replacement_attack(
    x,
    sampling_rate=44100,
    block_size=1024,
    overlap_factor=0.75,
    lower_bound=0,
    upper_bound=10,
    k=100,
    use_masking=False,
):
    """
    Perform a replacement attack on an audio signal by substituting blocks with similar ones
    based on a given distance upper_bound and masking condition.

    This implementation is based on:
    Darko Kirovski, Fabien A. P. Petitcolas, and Zeph Landau,
    "The Replacement Attack," IEEE Transactions on Audio, Speech, and Language Processing,
    vol. 15, no. 6, August 2007.

    Parameters:
        x (np.ndarray): The input audio signal.
        sampling_rate (int): Sampling rate of the audio signal (default: 44100 Hz).
        block_size (int): Size of each block for processing (default: 1024).
        overlap_factor (float): Overlap factor between consecutive blocks (default: 0.75).
        lower_bound (float): The lower bound of the similarity distance for considering a block as a candidate.
        upper_bound (float): The upper bound of the similarity distance for considering a block as a candidate.
        k (int): Maximum number of similar blocks to consider (default: 30).
        use_masking (bool): Whether to use psychoacoustic masking for distance calculation (default: True).

    Returns:
        np.ndarray: The processed audio signal with replacement attack applied.
    """

    overlap = int(overlap_factor * block_size)
    hop_size = block_size - overlap

    blocks = signal_analysis(x, block_size, hop_size)
    processed_blocks = []
    cnt_replaced, total = 0, 0
    masking_model = None

    if use_masking is True:
        masking_model = PsychoacousticModel(N=block_size, fs=sampling_rate, nfilts=24)

    for i in tqdm(range(len(blocks)), desc="Replacement attack", unit="blok"):
        block = blocks[i]
        overlap_indices = list(
            range(
                max(0, i - block_size // hop_size),
                min(len(blocks), i + block_size // hop_size + 1),
            )
        )
        similar_blocks, most_similar_block = find_most_similar_blocks(
            block, blocks, overlap_indices, lower_bound, upper_bound, k, masking_model
        )
        if len(similar_blocks) == 0:
            replacement_block = block
        else:
            replacement_block = least_squares_approximation(block, similar_blocks)
            dist = distance_function(block, replacement_block, masking_model)
            best_dist = distance_function(block, most_similar_block, masking_model)
            cnt_replaced += 1
            if dist > best_dist:
                replacement_block = block
                cnt_replaced -= 1

        total += 1
        processed_blocks.append(replacement_block)

    logger.info(f"Replaced:{(cnt_replaced / total * 100):.2f}% of blocks.")
    return signal_synthesis(np.array(processed_blocks), block_size, hop_size)[: len(x)]
