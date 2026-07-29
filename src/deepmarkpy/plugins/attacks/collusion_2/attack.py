"""
Collusion 2 / Montage-Splicing Attack

Extended collusion attack that combines two differently-watermarked copies
of the same audio by splicing segments from copy B into copy A. Unlike the
original CollusionAttack (fixed-size random segments), this variant prefers
to cut at natural pauses (silences) so the splices are less audible.

Selection works in two phases against a target replacement ratio:

  Phase 1 (silence-aligned): partition the speech between detected pauses
  into candidate segments of length [min_seg, max_seg] and greedily pick
  whole ones until adding another would overshoot the target. A single
  final partial slice of an unused candidate may close the remaining gap;
  that one slice is allowed to be shorter than min_seg.

  Phase 2 (fill): only if phase 1 ends more than ``target_tolerance`` below
  the target -- e.g. the audio has few or no usable pauses -- carve the
  still-untouched audio into random-length segments in [min_seg, max_seg]
  until the target is reached. Here too, only the very last piece may be
  shorter than min_seg. These cuts don't fall on pauses, but the crossfade
  smooths them and replacing more of the signal only strengthens the attack.

We never overshoot the target: replacing slightly less than requested is
preferable to more, so a low detection accuracy can't be blamed on
accidentally replacing too much.
"""

import logging
from dataclasses import dataclass

import librosa
import numpy as np

from deepmarkpy.core.base_attack import BaseAttack

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SpliceParams:
    """Resolved attack parameters.

    Durations are pre-converted to samples so the rest of the pipeline works
    in a single unit; only the ratio/tolerance stay fractions.
    """

    target_ratio: float
    min_seg_samples: int
    max_seg_samples: int
    crossfade_samples: int
    silence_thresh_db: float
    min_silence_ms: float
    target_tolerance: float
    log_splice_stats: bool


class Collusion2Attack(BaseAttack):
    """
    Montage/splicing collusion attack -- prefers to splice at silences.

    Config parameters:
        - target_ratio (float): Target fraction (0-1) of the
            audio to replace with the second watermarked copy.
        - min_segment_sec (float): Minimum segment duration in
            seconds (only the final gap-closing slice may be shorter).
        - max_segment_sec (float): Maximum segment duration in
            seconds.
        - crossfade_ms (float): Crossfade duration at splice
            boundaries in milliseconds.
        - silence_threshold_db (float): level (dB below the peak)
            under which audio counts as silence; passed to librosa as
            top_db = abs(value).
        - min_silence_ms (float): Minimum duration of a silent
            stretch for it to be usable as a splice point.
        - target_tolerance (float): Maximum acceptable shortfall
            below target_ratio before the phase-2 fill kicks in. E.g. 0.02
            means "stay within 2 percentage points of the target".
        - log_splice_stats (bool): When true, log the per-splice
            replacement statistics (counts, durations, each splice's span).
            Defaults to false; intended for testing/inspection.
    """

    # Built-in fallback defaults, used only when a key is absent from both
    # kwargs and the plugin's config.json. Keeping them in one place avoids
    # the same default drifting across the per-parameter lookups below.
    _DEFAULTS = {
        "target_ratio": 0.3,
        "min_segment_sec": 0.5,
        "max_segment_sec": 2.0,
        "crossfade_ms": 50,
        "silence_threshold_db": -35,
        "min_silence_ms": 30,
        "target_tolerance": 0.02,
        "log_splice_stats": False,
    }

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """Perform the montage/splicing collusion attack.

        Args:
            audio (np.ndarray): The first watermarked audio (watermark A).
            **kwargs:
                - model: watermarking model instance
                - orig_audio: original (clean) audio
                - sampling_rate: sample rate in Hz
                - any collusion_2 parameter override

        Returns:
            np.ndarray: Spliced audio mixing segments from two watermarked
            copies.
        """
        model, orig_audio, sampling_rate = self._require_inputs(kwargs)
        sr = int(sampling_rate)
        params = self._resolve_params(kwargs, sr)

        second_audio = self._create_second_copy(model, orig_audio, sampling_rate)

        # Work with the shorter of the two signals so indices are valid in
        # both; embed() may return a slightly different length than input.
        min_len = min(len(audio), len(second_audio))
        if min_len == 0:
            logger.warning("Collusion2: empty audio, returning input unchanged.")
            return audio.copy()

        audio_a = audio[:min_len].copy()
        audio_b = second_audio[:min_len]

        rng = np.random.default_rng()
        candidates = self._segments_between_pauses(audio_a, sr, params)
        regions = self._select_regions(candidates, min_len, params, rng)

        # Merge adjacent/touching regions before splicing. Two neighbouring
        # regions spliced separately would crossfade B->A->B at their shared
        # border -- a needless dip back to A inside a continuous B stretch.
        # Merging first keeps the crossfade only at the true A<->B edges.
        merged = self._merge_regions(sorted(regions))

        result, splices = self._apply_splices(audio_a, audio_b, merged, params)
        if params.log_splice_stats:
            self._log_result(candidates, splices, min_len, sr)
        return result

    # ------------------------------------------------------------------
    # Input validation and parameter resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _require_inputs(kwargs):
        """Pull and validate the mandatory model/audio inputs."""
        model = kwargs.get("model")
        orig_audio = kwargs.get("orig_audio")
        sampling_rate = kwargs.get("sampling_rate")
        if model is None or orig_audio is None or sampling_rate is None:
            raise ValueError(
                "'model', 'orig_audio' and 'sampling_rate' must be provided."
            )
        return model, orig_audio, sampling_rate

    def _param(self, kwargs, name):
        """Resolve a parameter as kwargs > config.json > built-in default."""
        config = self.config or {}
        return kwargs.get(name, config.get(name, self._DEFAULTS[name]))

    def _resolve_params(self, kwargs, sr):
        """Build the :class:`_SpliceParams` for this run, in sample units."""
        return _SpliceParams(
            target_ratio=self._param(kwargs, "target_ratio"),
            min_seg_samples=int(self._param(kwargs, "min_segment_sec") * sr),
            max_seg_samples=int(self._param(kwargs, "max_segment_sec") * sr),
            crossfade_samples=int(
                self._param(kwargs, "crossfade_ms") * sr / 1000
            ),
            silence_thresh_db=self._param(kwargs, "silence_threshold_db"),
            min_silence_ms=self._param(kwargs, "min_silence_ms"),
            target_tolerance=self._param(kwargs, "target_tolerance"),
            log_splice_stats=self._param(kwargs, "log_splice_stats"),
        )

    @staticmethod
    def _create_second_copy(model, orig_audio, sampling_rate):
        """Embed a second, differently-watermarked copy of the audio."""
        second_watermark = model.generate_watermark()
        second_audio = model.embed(orig_audio, second_watermark, sampling_rate)
        if second_audio is None or len(second_audio) == 0:
            raise ValueError(
                "Collusion2 attack failed: model.embed() returned empty audio."
            )
        return second_audio

    # ------------------------------------------------------------------
    # Pause detection and candidate segments
    # ------------------------------------------------------------------

    @staticmethod
    def _pause_midpoints(audio, sr, params):
        """Return the midpoint sample of every usable pause.

        ``librosa.effects.split`` returns the non-silent intervals using an
        energy threshold ``top_db`` dB below the peak; the pauses are simply
        the gaps between (and around) those intervals. We compute pause
        centres straight from those intervals -- no per-sample mask -- which
        keeps this O(number of pauses) instead of O(number of samples). We
        don't need a true speech/non-speech classifier here, only the pause
        centres, which serve as splice points (any imprecision is hidden by
        the crossfade). Only pauses lasting at least ``min_silence_ms``
        qualify.

        ``silence_thresh_db`` is the (negative) level below the peak that
        counts as silence; librosa expects the positive magnitude ``top_db``.
        """
        top_db = abs(params.silence_thresh_db)
        # Short frames/hop so brief pauses are resolved: a frame must fit
        # inside a gap for its energy to register as silence, so keep the
        # frame well under the minimum pause we care about (min_silence_ms,
        # which defaults to 30ms). 20ms frame with a 10ms hop is fine-grained.
        frame_length = max(int(0.02 * sr), 1)
        hop_length = max(int(0.01 * sr), 1)
        min_silence = int(params.min_silence_ms * sr / 1000)

        intervals = librosa.effects.split(
            audio, top_db=top_db,
            frame_length=frame_length, hop_length=hop_length,
        )

        # Each pause is a gap [prev_end, next_start); the leading and trailing
        # silences are bounded by 0 and len(audio). Take the centre of every
        # gap long enough to count as a usable pause.
        n = len(audio)
        midpoints = []
        prev_end = 0
        for start, end in intervals:
            if start - prev_end >= min_silence:
                midpoints.append((prev_end + start) // 2)
            prev_end = end
        if n - prev_end >= min_silence:
            midpoints.append((prev_end + n) // 2)
        return midpoints

    def _segments_between_pauses(self, audio, sr, params):
        """Candidate (start, end) segments of speech lying between pauses.

        Cuts the audio at pause midpoints and keeps only the spans whose
        length already falls within [min_seg, max_seg]. A span longer than
        max_seg is a stretch of speech with no usable pause inside it (so it
        can't be split on a pause); a span shorter than min_seg is too small
        to be a segment. Both are intentionally left out of the candidate
        list -- they become the "untouched" audio that the phase-2 fill may
        carve up later. With no usable pauses the list is empty and phase 2
        handles the whole signal.
        """
        min_seg = params.min_seg_samples
        max_seg = params.max_seg_samples
        n = len(audio)

        midpoints = self._pause_midpoints(audio, sr, params)

        # Cut points partition [0, n] into spans between consecutive pauses
        # (and the head/tail bounded by a pause on one side).
        cuts = [0, *midpoints, n]
        return [
            (start, end)
            for start, end in zip(cuts, cuts[1:])
            if min_seg <= (end - start) <= max_seg
        ]

    # ------------------------------------------------------------------
    # Region selection
    # ------------------------------------------------------------------

    def _select_regions(self, candidates, min_len, params, rng):
        """Choose which audio to replace with copy B.

        See the module docstring for the two-phase strategy. Targets a
        fraction of the total DURATION (not segment count) and never
        overshoots it.
        """
        target_samples = int(min_len * params.target_ratio)
        regions = []
        replaced_total = 0

        # --- Phase 1: silence-aligned segments ---
        order = list(range(len(candidates)))
        rng.shuffle(order)
        used = set()
        for idx in order:
            s, e = candidates[idx]
            if replaced_total + (e - s) <= target_samples:
                regions.append((s, e))
                used.add(idx)
                replaced_total += e - s

        # Close the remaining gap with a single partial slice of an unused
        # candidate, keeping that cut on a silence-bounded span. This is the
        # only phase-1 slice allowed to be shorter than min_seg (an inaudible
        # tail on a pause). Skipped if the gap is too small for the crossfade.
        gap = target_samples - replaced_total
        min_partial = max(2 * params.crossfade_samples, 1)
        if gap >= min_partial:
            for idx in order:
                if idx in used:
                    continue
                s, e = candidates[idx]
                if (e - s) >= gap:
                    regions.append((s, s + gap))
                    replaced_total += gap
                    break

        # Within tolerance? Accept the small undershoot and keep every splice
        # on a natural pause -- don't touch non-silence audio.
        tolerance_samples = int(min_len * params.target_tolerance)
        if replaced_total >= target_samples - tolerance_samples:
            return regions

        # --- Phase 2: fill from untouched (non-silence) audio ---
        self._fill_to_target(regions, replaced_total, target_samples,
                              params, min_len, rng)
        return regions

    def _fill_to_target(self, regions, replaced_total, target_samples, params,
                        min_len, rng):
        """Carve untouched audio into [min,max] segments up to the target.

        Mutates ``regions`` in place. Walks the free spans (audio not yet
        selected) in random order and emits random-length segments in
        [min_seg, max_seg]. The single piece that closes the gap may be
        shorter than min_seg so the target is hit exactly; every other piece
        is at least min_seg. Never overshoots.
        """
        min_seg = params.min_seg_samples
        max_seg = params.max_seg_samples
        for fs, fe in self._free_intervals(regions, min_len, rng):
            cursor = fs
            while cursor < fe:
                need = target_samples - replaced_total
                if need <= 0:
                    return
                span_left = fe - cursor
                # Final piece: the whole remaining need fits in this span and
                # in one segment. Place it exactly (may be < min_seg) and stop.
                if need <= span_left and need <= max_seg:
                    regions.append((cursor, cursor + need))
                    return
                # Otherwise take a full random-length chunk in [min_seg,
                # max_seg], bounded by what's left in this span.
                hi = min(max_seg, span_left)
                if hi < min_seg:
                    break  # leftover of this span too short for a valid segment
                take = int(rng.integers(min_seg, hi + 1))
                regions.append((cursor, cursor + take))
                replaced_total += take
                cursor += take

    @staticmethod
    def _free_intervals(regions, min_len, rng):
        """Return the (start, end) spans of [0, min_len) not in ``regions``.

        These are the parts of the audio still showing copy A. Returned in
        random order so the fill pass spreads its forced splices around
        rather than always biasing toward the start of the file.
        """
        occupied = Collusion2Attack._merge_regions(sorted(regions))
        free = []
        cursor = 0
        for s, e in occupied:
            if s > cursor:
                free.append((cursor, s))
            cursor = max(cursor, e)
        if cursor < min_len:
            free.append((cursor, min_len))
        rng.shuffle(free)
        return free

    @staticmethod
    def _merge_regions(sorted_regions):
        """Merge overlapping or touching (start, end) regions into one.

        Expects ``sorted_regions`` sorted by start. Two regions are merged
        when the next one starts at or before the current one's end, so a
        run of adjacent segments becomes a single contiguous region.
        """
        if not sorted_regions:
            return []
        merged = [list(sorted_regions[0])]
        for s, e in sorted_regions[1:]:
            if s <= merged[-1][1]:        # touches/overlaps previous region
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return [(s, e) for s, e in merged]

    # ------------------------------------------------------------------
    # Splicing with crossfade
    # ------------------------------------------------------------------

    def _apply_splices(self, audio_a, audio_b, regions, params):
        """Splice every region from B into A, returning (result, splices)."""
        result = audio_a.copy()
        splices = []
        for s, e in regions:
            result = self._splice_with_crossfade(
                result, audio_b, s, e, params.crossfade_samples,
            )
            splices.append((s, e, e - s))
        return result, splices

    @staticmethod
    def _splice_with_crossfade(audio_a, audio_b, start, end, fade_len):
        """Replace audio_a[start:end] with audio_b[start:end], crossfading.

        Applies a linear crossfade of ``fade_len`` samples at both the entry
        and exit boundaries so the transition is smooth and less audible than
        a hard cut.
        """
        result = audio_a.copy()
        seg_len = end - start

        # Clamp fade to half the segment (can't fade more than we have).
        fade = min(fade_len, seg_len // 2)

        if fade <= 0:
            # No crossfade possible -- hard splice.
            result[start:end] = audio_b[start:end]
            return result

        fade_in = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        fade_out = 1.0 - fade_in

        # Entry crossfade: A fades out, B fades in.
        result[start:start + fade] = (
            audio_a[start:start + fade] * fade_out
            + audio_b[start:start + fade] * fade_in
        )
        # Middle: pure B.
        result[start + fade:end - fade] = audio_b[start + fade:end - fade]
        # Exit crossfade: B fades out, A fades in.
        result[end - fade:end] = (
            audio_b[end - fade:end] * fade_out
            + audio_a[end - fade:end] * fade_in
        )
        return result

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @staticmethod
    def _log_result(candidates, splices, min_len, sr):
        """Log a summary of which audio was replaced."""
        replaced_total = sum(length for _, _, length in splices)
        total_dur = min_len / sr
        replaced_dur = replaced_total / sr
        pct = 100 * replaced_total / min_len if min_len else 0.0
        logger.info(
            f"Collusion2: {len(candidates)} pause-aligned candidates, "
            f"{len(splices)} spliced, "
            f"{replaced_dur:.2f}s / {total_dur:.2f}s ({pct:.1f}%)"
        )
        for i, (s, e, length) in enumerate(splices):
            logger.info(
                f"  splice {i + 1}: {s / sr:.3f}s - {e / sr:.3f}s "
                f"(duration: {length / sr:.3f}s = {length} samples)"
            )
