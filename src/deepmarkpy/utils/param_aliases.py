from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping


# Legacy/public aliases are keyed by the existing flag/config name and point to
# the attack-local canonical parameter name. Keep this table literal: the old
# names are irregular and cannot be derived safely.
LEGACY_PARAM_ALIASES: dict[str, dict[str, str]] = {
    "bandstop_filter": {
        "order_bandstop": "order",
        "freq_range_bandstop": "freq_range",
    },
    "chorus": {
        "start_delays_chorus": "start_delays",
        "w_delays_chorus": "w_delays",
        "delay_rates_chorus": "delay_rates",
        "dry_gain_chorus": "dry_gain",
        "chorus_gains_chorus": "chorus_gains",
    },
    "codec2_vocoder": {
        "bitrate_codec2": "bitrate",
    },
    "collusion": {
        "size_collusion": "size",
    },
    "collusion_2": {
        "target_ratio_collusion2": "target_ratio",
        "min_segment_sec_collusion2": "min_segment_sec",
        "max_segment_sec_collusion2": "max_segment_sec",
        "crossfade_ms_collusion2": "crossfade_ms",
        "silence_threshold_db_collusion2": "silence_threshold_db",
        "min_silence_ms_collusion2": "min_silence_ms",
        "target_tolerance_collusion2": "target_tolerance",
        "log_splice_stats_collusion2": "log_splice_stats",
    },
    "crop_beginning": {
        "crop_percentage_beginning": "crop_percentage",
    },
    "crop_random": {
        "crop_percentage_random": "crop_percentage",
    },
    "cross_model": {
        "different_model_name_cross_model": "different_model_name",
    },
    "cut_samples": {
        "max_sequence_length_cut": "max_sequence_length",
        "num_sequences_cut": "num_sequences",
        "duration_cut": "duration",
        "max_value_difference_cut": "max_value_difference",
        "cut_max_sequence_length": "max_sequence_length",
        "cut_num_sequences": "num_sequences",
        "cut_duration": "duration",
        "cut_max_value_difference": "max_value_difference",
    },
    "descript_audio_codec": {
        "model_type_dac": "model_type",
        "target_sampling_rate_dac": "target_sampling_rate",
        "n_codebooks_dac": "n_codebooks",
    },
    "diffusion": {
        "model_name_diffusion": "model_name",
        "steps_diffusion": "steps",
        "diffusion_steps": "steps",
    },
    "echo": {
        "volume_range_echo": "volume_range",
        "duration_range_echo": "duration_range",
    },
    "encodec": {
        "model_name_encodec": "model_name",
        "bandwidth_encodec": "bandwidth",
        "target_sampling_rate_encodec": "target_sampling_rate",
    },
    "equalizer": {
        "gains_equalizer": "gains",
    },
    "flanger": {
        "start_delay_flanger": "start_delay",
        "w_delay_flanger": "w_delay",
        "delay_rate_flanger": "delay_rate",
        "gain_flanger": "gain",
    },
    "flip_samples": {
        "num_flip_samples": "num_flips",
        "duration_flip_samples": "duration",
        "flip_duration": "duration",
    },
    "gaussian_noise": {
        "snr_db_gaussian_noise": "snr_db",
    },
    "highpass_filter": {
        "cutoff_freq_highpass": "cutoff_freq",
        "order_highpass": "order",
    },
    "inverted_time_stretch": {
        "rate_inverted_time_stretch": "stretch_rate",
        "inverted_stretch_rate": "stretch_rate",
    },
    "lowpass_filter": {
        "cutoff_freq_lowpass": "cutoff_freq",
        "cutoff_freq_per_sr_lowpass": "cutoff_freq_per_sr",
        "order_lowpass": "order",
    },
    "lpc": {
        "order_lpc": "order",
        "axis_lpc": "axis",
    },
    "mixing": {
        "music_folder_mixing": "music_folder",
        "music_volume_high_mixing": "music_volume_high",
        "music_volume_low_mixing": "music_volume_low",
        "smoothing_window_mixing": "smoothing_window",
        "eq_gains_mixing": "eq_gains",
        "highpass_cutoff_mixing": "highpass_cutoff",
        "sampling_rate_mixing": "sampling_rate",
    },
    "mp3_compression": {
        "quality_mp3": "quality",
    },
    "network_transmission": {
        "bitrate_bps_netem": "bitrate_bps",
        "frame_duration_ms_netem": "frame_duration_ms",
        "delay_ms_netem": "delay_ms",
        "jitter_ms_netem": "jitter_ms",
        "packet_loss_netem": "packet_loss",
        "duplication_netem": "duplication",
        "reorder_netem": "reorder",
        "corruption_netem": "corruption",
        "fec_enabled_netem": "fec_enabled",
        "expected_loss_netem": "expected_loss",
        "ns_enabled_netem": "ns_enabled",
        "vad_enabled_netem": "vad_enabled",
        "agc_enabled_netem": "agc_enabled",
        "agc_target_lufs_netem": "agc_target_lufs",
        "playout_delay_ms_netem": "playout_delay_ms",
    },
    "neural_vocoder": {
        "model_name_neural_vocoder": "model_name",
    },
    "opus_codec": {
        "bitrate_opus_codec": "bitrate",
        "framesize_opus_codec": "framesize",
    },
    "pcm_quantization": {
        "pcm_quantization": "bit_depth",
        "pcm": "bit_depth",
    },
    "pink_noise": {
        "snr_db_pink_noise": "snr_db",
        "snr_db_pn": "snr_db",
    },
    "pitch_shift": {
        "cents_pitch_shift": "cents",
    },
    "quantization": {
        "bit_quantization": "quantization_levels",
        "quantization_bit": "quantization_levels",
    },
    "replacement": {
        "block_size_replacement": "block_size",
        "overlap_factor_replacement": "overlap_factor",
        "lower_bound_replacement": "lower_bound",
        "upper_bound_replacement": "upper_bound",
        "use_masking_replacement": "use_masking",
        "replacement_block_size": "block_size",
        "replacement_overlap_factor": "overlap_factor",
        "replacement_lower_bound": "lower_bound",
        "replacement_upper_bound": "upper_bound",
        "replacement_use_masking": "use_masking",
    },
    "replacement_2": {
        "block_size_replacement2": "block_size",
        "overlap_factor_replacement2": "overlap_factor",
        "lower_bound_replacement2": "lower_bound",
        "upper_bound_replacement2": "upper_bound",
        "k_replacement2": "k",
        "use_masking_replacement2": "use_masking",
        "search_window_sec_replacement2": "search_window_sec",
        "search_dims_replacement2": "search_dims",
        "tile_size_replacement2": "tile_size",
    },
    "replay": {
        "air_folder_replay": "air_folder",
        "air_sr_replay": "air_sampling_rate",
        "bandpass_replay": "bandpass",
        "low_freq_replay": "low_freq",
        "high_freq_replay": "high_freq",
        "filter_order_replay": "filter_order",
        "add_noise_replay": "add_noise",
        "snr_db_replay": "snr_db",
        "sampling_rate_replay": "sampling_rate",
    },
    "resampling_poly": {
        "down_factor_resampling_poly": "down_factor",
    },
    "smoothing": {
        "window_size_smoothing": "window_size",
    },
    "speech_enhancement_1": {
        "type_se1": "type",
        "noise_strength_se1": "noise_strength",
        "sampling_rate_se1": "sampling_rate",
    },
    "speech_enhancement_2": {
        "model_name_se2": "model_name",
        "model_name_speech_enh": "model_name",
        "noise_strength_se2": "noise_strength",
        "sampling_rate_se2": "sampling_rate",
    },
    "speech_tokenization": {
        "model_name_speech_tokenization": "model_name",
        "sampling_rate_st": "sampling_rate",
    },
    "stft_quantization": {
        "n_fft_stft_quantization": "n_fft",
        "hop_length_stft_quantization": "hop_length",
        "quantization_levels_stft_quantization": "quantization_levels",
    },
    "time_stretch": {
        "stretch_rate_time_stretch": "stretch_rate",
    },
    "vae": {
        "model_name_vae": "model_name",
    },
    "wavelet": {
        "wt_mode_wavelet": "wt_mode",
        "threshold_factor_wavelet": "threshold_factor",
    },
    "zero_bit_collusion": {
        "x_zero_bit_collusion": "x",
        "position_zero_bit_collusion": "position",
        "original_audio_collusion": "original_audio",
    },
    "zero_cross_inserts": {
        "pause_length_zero_cross_inserts": "pause_length",
        "min_distance_zero_cross_inserts": "min_distance",
        "zero_cross_pause_length": "pause_length",
        "zero_cross_min_distance": "min_distance",
    },
}


LEGACY_FLAG_TO_ATTACK_PARAM: dict[str, tuple[str, str]] = {
    legacy: (attack_key, canonical)
    for attack_key, aliases in LEGACY_PARAM_ALIASES.items()
    for legacy, canonical in aliases.items()
}


_CANONICAL_TO_LEGACY: dict[str, dict[str, list[str]]] = {}
for _attack_key, _aliases in LEGACY_PARAM_ALIASES.items():
    grouped: dict[str, list[str]] = defaultdict(list)
    for _legacy, _canonical in _aliases.items():
        grouped[_canonical].append(_legacy)
    _CANONICAL_TO_LEGACY[_attack_key] = dict(grouped)


def normalize_attack_config(attack_key: str, config: Mapping[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}

    aliases = LEGACY_PARAM_ALIASES.get(attack_key, {})
    normalized: dict[str, Any] = {}
    for key, value in config.items():
        if key.startswith("_"):
            normalized[key] = value
            continue
        normalized[aliases.get(key, key)] = value
    return normalized


def expand_attack_config(attack_key: str, config: Mapping[str, Any] | None) -> dict[str, Any]:
    expanded = normalize_attack_config(attack_key, config)
    for canonical, legacy_names in _CANONICAL_TO_LEGACY.get(attack_key, {}).items():
        if canonical in expanded:
            for legacy in legacy_names:
                expanded.setdefault(legacy, expanded[canonical])
    return expanded


def expand_attack_kwargs(attack_key: str, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    expanded = normalize_attack_config(attack_key, kwargs)
    for canonical, legacy_names in _CANONICAL_TO_LEGACY.get(attack_key, {}).items():
        if canonical in expanded:
            for legacy in legacy_names:
                expanded.setdefault(legacy, expanded[canonical])
    return expanded


def namespaced_dest(attack_key: str, param: str) -> str:
    return f"{attack_key}__{param}"


def new_flag_for_legacy(legacy_name: str) -> str | None:
    attack_param = LEGACY_FLAG_TO_ATTACK_PARAM.get(legacy_name)
    if attack_param is None:
        return None
    attack_key, param = attack_param
    return f"--{attack_key}.{param}"
