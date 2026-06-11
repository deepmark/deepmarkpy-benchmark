"""
Collusion 2 / Montage-Splicing Attack

Extended collusion attack that combines two differently-watermarked copies
of the same audio by splicing segments at natural silence boundaries.
Unlike the original CollusionAttack (fixed-size random segments), this
variant:
  - Detects silence/pause moments in the audio via short-term LUFS
  - Uses those moments as splice points (cuts are less audible in silence)
  - Picks random segment durations between 0.5s and 2s
  - Applies a short crossfade at each boundary so transitions are smooth

The goal is to make the splicing as imperceptible as possible to a human
listener while still disrupting watermark detection by mixing segments
from two different watermark embeddings.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfilt

from core.base_attack import BaseAttack


class Collusion2Attack(BaseAttack):
    """
    Montage/splicing collusion attack — splice at silence boundaries.

    Config parameters:
        - target_ratio_collusion2 (float): Target fraction (0-1) of the
            audio to replace with the second watermarked copy (default: 0.3)
        - min_segment_sec_collusion2 (float): Minimum splice segment
            duration in seconds (default: 0.5)
        - max_segment_sec_collusion2 (float): Maximum splice segment
            duration in seconds (default: 2.0)
        - crossfade_ms_collusion2 (float): Crossfade duration at splice
            boundaries in milliseconds (default: 50)
        - silence_threshold_db_collusion2 (float): LUFS threshold below
            max to consider as silence (default: -35)
    """

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform montage/splicing collusion attack.

        Args:
            audio (np.ndarray): The first watermarked audio (watermark A).
            **kwargs:
                - model: watermarking model instance
                - orig_audio: original (clean) audio
                - sampling_rate: sample rate in Hz
        Returns:
            np.ndarray: Spliced audio mixing segments from two watermarked copies.
        """
        model = kwargs.get("model")
        orig_audio = kwargs.get("orig_audio")
        sampling_rate = kwargs.get("sampling_rate")

        if model is None or orig_audio is None or sampling_rate is None:
            raise ValueError(
                "'model', 'orig_audio' and 'sampling_rate' must be provided."
            )

        # Create second watermarked copy with a different watermark
        second_watermark = model.generate_watermark()
        second_audio = model.embed(orig_audio, second_watermark, sampling_rate)

        if second_audio is None or len(second_audio) == 0:
            raise ValueError(
                "Collusion2 attack failed: model.embed() returned empty audio."
            )

        # Read parameters from config or kwargs
        target_ratio = kwargs.get(
            "target_ratio_collusion2",
            self.config.get("target_ratio_collusion2", 0.3),
        )
        min_seg_sec = kwargs.get(
            "min_segment_sec_collusion2",
            self.config.get("min_segment_sec_collusion2", 0.5),
        )
        max_seg_sec = kwargs.get(
            "max_segment_sec_collusion2",
            self.config.get("max_segment_sec_collusion2", 2.0),
        )
        crossfade_ms = kwargs.get(
            "crossfade_ms_collusion2",
            self.config.get("crossfade_ms_collusion2", 50),
        )
        silence_thresh_db = kwargs.get(
            "silence_threshold_db_collusion2",
            self.config.get("silence_threshold_db_collusion2", -35),
        )

        sr = int(sampling_rate)
        min_seg_samples = int(min_seg_sec * sr)
        max_seg_samples = int(max_seg_sec * sr)
        crossfade_samples = int(crossfade_ms * sr / 1000)

        # Work with the shorter of the two signals
        min_len = min(len(audio), len(second_audio))
        audio_a = audio[:min_len].copy()
        audio_b = second_audio[:min_len]

        # Detect silence positions and use them as segment boundaries.
        # This partitions the ENTIRE audio into contiguous segments
        # that start/end at silence moments (natural speech pauses).
        silence_mask = self._detect_silence(audio_a, sr, silence_thresh_db)
        boundaries = self._find_segment_boundaries(
            silence_mask, sr, min_seg_samples, max_seg_samples,
        )

        # Build list of contiguous segments covering the whole audio
        segments = []
        for i in range(len(boundaries) - 1):
            segments.append((boundaries[i], boundaries[i + 1]))

        # Select segments to replace, targeting a fraction of the total
        # DURATION (not segment count) so the replaced ratio is precise and
        # the resulting accuracy is interpretable. We never overshoot the
        # target -- it is better to replace slightly less than the requested
        # ratio than more, so a low accuracy can't be blamed on accidentally
        # replacing too much.
        rng = np.random.default_rng()
        target_samples = int(min_len * target_ratio)

        order = list(range(len(segments)))
        rng.shuffle(order)

        regions = []          # (start, end) regions to splice with B
        replaced_total = 0
        used = set()

        # Phase 1: greedily take whole segments (in random order) as long as
        # they don't push the total over the target. Picking the next random
        # segment and, when it would overshoot, skipping ahead to a smaller
        # one that still fits is exactly what this single pass does.
        for idx in order:
            s, e = segments[idx]
            seg_len = e - s
            if replaced_total + seg_len <= target_samples:
                regions.append((s, e))
                used.add(idx)
                replaced_total += seg_len

        # Phase 2: close the remaining gap by taking a partial slice of one
        # unused segment. After phase 1 no whole unused segment fits the gap
        # (any that fit were already taken), so splitting a segment is the
        # only way to reach the target exactly. The cut at ``s + gap`` is not
        # on a silence boundary, but the crossfade smooths it. If the gap is
        # too small for the crossfade to hide cleanly, skip the split and
        # undershoot -- an inaudible undershoot beats an audible artifact.
        gap = target_samples - replaced_total
        min_partial = max(2 * crossfade_samples, 1)
        if gap >= min_partial:
            for idx in order:
                if idx in used:
                    continue
                s, e = segments[idx]
                if (e - s) >= gap:
                    regions.append((s, s + gap))
                    replaced_total += gap
                    break

        result = audio_a.copy()
        splices = []
        for s, e in sorted(regions):
            result = self._splice_with_crossfade(
                result, audio_b, s, e, crossfade_samples,
            )
            splices.append((s, e, e - s))

        # Log splice details
        import logging
        logger = logging.getLogger(__name__)
        total_dur = min_len / sr
        replaced_total = sum(l for _, _, l in splices)
        replaced_dur = replaced_total / sr
        logger.info(
            f"Collusion2: {len(segments)} total segments, "
            f"{len(splices)} replaced, "
            f"{replaced_dur:.2f}s / {total_dur:.2f}s "
            f"({100*replaced_total/min_len:.1f}%)"
        )
        for i, (s, e, l) in enumerate(splices):
            logger.info(
                f"  splice {i+1}: {s/sr:.3f}s - {e/sr:.3f}s "
                f"(duration: {l/sr:.3f}s = {l} samples)"
            )

        return result

    # ------------------------------------------------------------------
    # Silence detection (adapted from MixingAttack VAD logic)
    # ------------------------------------------------------------------

    def _detect_silence(self, audio, sr, threshold_db=-35):
        """Return a binary mask: 1 = silence, 0 = speech/sound.

        Uses short-term LUFS with K-weighting, same approach as the
        MixingAttack VAD but inverted (we want silence, not speech).
        """
        audio_k = self._k_weighting(audio, sr)
        window_samples = int(0.05 * sr)  # 50ms window for fine granularity
        if window_samples < 1:
            window_samples = 1

        mean_square = uniform_filter1d(audio_k ** 2, size=window_samples, mode='constant')
        mean_square = np.maximum(mean_square, 1e-10)
        lufs = -0.691 + 10 * np.log10(mean_square)

        max_lufs = np.max(lufs)
        silence_boundary = max_lufs + threshold_db  # e.g. max - 35 dB

        silence_mask = (lufs < silence_boundary).astype(np.int32)
        return silence_mask

    def _k_weighting(self, audio, sr):
        """Simplified K-weighting: high-shelf boost + high-pass at 38 Hz."""
        # High-shelf filter (+4 dB above 1500 Hz)
        f0 = 1500.0
        G = 4.0
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

        from scipy.signal import lfilter
        b = np.array([b0/a0, b1/a0, b2/a0])
        a = np.array([1, a1/a0, a2/a0])
        audio_hs = lfilter(b, a, audio)

        # High-pass at 38 Hz
        hp_freq = 38.0
        nyquist = sr / 2
        if hp_freq < nyquist:
            sos_hp = butter(2, hp_freq / nyquist, btype='high', output='sos')
            return sosfilt(sos_hp, audio_hs).astype(np.float32)
        return audio_hs.astype(np.float32)

    def _find_segment_boundaries(self, silence_mask, sr, min_seg, max_seg):
        """Partition the audio into contiguous segments using silence as boundaries.

        Finds all silence midpoints, then builds boundary list so segments:
          - Cover the entire audio (no gaps)
          - Start/end at silence moments where possible
          - Stay between min_seg and max_seg in length
          - If no silence is available, splits at regular intervals

        Returns sorted list of boundary positions including 0 and len(audio).
        """
        n = len(silence_mask)
        min_silence_ms = 30
        min_silence_samples = int(min_silence_ms * sr / 1000)

        # Find midpoints of all silence regions
        silence_midpoints = []
        i = 0
        while i < n:
            if silence_mask[i] == 1:
                j = i
                while j < n and silence_mask[j] == 1:
                    j += 1
                if (j - i) >= min_silence_samples:
                    silence_midpoints.append((i + j) // 2)
                i = j
            else:
                i += 1

        # Build boundaries: start with 0, end with n
        boundaries = [0]

        if not silence_midpoints:
            # No silence found — split at regular intervals
            step = (min_seg + max_seg) // 2
            pos = step
            while pos < n - min_seg // 2:
                boundaries.append(pos)
                pos += step
        else:
            # Use silence midpoints as boundaries, respecting min/max segment
            last_boundary = 0
            for mid in sorted(silence_midpoints):
                dist = mid - last_boundary
                if dist < min_seg:
                    continue  # too close to previous boundary
                if dist > max_seg:
                    # Force a boundary even without silence (segment too long)
                    # Place it at max_seg from last boundary
                    forced = last_boundary + max_seg
                    boundaries.append(forced)
                    last_boundary = forced
                    # Re-check this silence midpoint
                    if mid - last_boundary >= min_seg:
                        boundaries.append(mid)
                        last_boundary = mid
                else:
                    boundaries.append(mid)
                    last_boundary = mid

            # If last segment would be too long, force extra boundaries
            while n - last_boundary > max_seg:
                forced = last_boundary + max_seg
                boundaries.append(forced)
                last_boundary = forced

        boundaries.append(n)

        # Remove duplicates and sort
        boundaries = sorted(set(boundaries))
        return boundaries

    # ------------------------------------------------------------------
    # Splicing with crossfade
    # ------------------------------------------------------------------

    def _splice_with_crossfade(self, audio_a, audio_b, start, end, fade_len):
        """Replace audio_a[start:end] with audio_b[start:end], crossfading.

        Applies a linear crossfade of ``fade_len`` samples at both the
        entry and exit boundaries so the transition is smooth and less
        audible than a hard cut.
        """
        result = audio_a.copy()
        seg_len = end - start

        # Clamp fade to half the segment (can't fade more than we have)
        fade = min(fade_len, seg_len // 2)

        if fade > 0:
            # Entry crossfade: A fades out, B fades in
            fade_in = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            fade_out = 1.0 - fade_in
            result[start:start + fade] = (
                audio_a[start:start + fade] * fade_out
                + audio_b[start:start + fade] * fade_in
            )
            # Middle: pure B
            result[start + fade:end - fade] = audio_b[start + fade:end - fade]
            # Exit crossfade: B fades out, A fades in
            result[end - fade:end] = (
                audio_b[end - fade:end] * fade_out
                + audio_a[end - fade:end] * fade_in
            )
        else:
            # No crossfade possible — hard splice
            result[start:end] = audio_b[start:end]

        return result
