"""Attack group definitions for the DeepMark Benchmark.

Each group declares which quality / intelligibility metrics are
meaningful for that attack family. This acts as the single source of
truth for both the benchmark runner (which metrics to compute) and
report generators (which metrics to display).

Groups with empty metric lists skip those metrics entirely -- this
avoids reporting misleading values (e.g. PESQ for collusion attacks
that preserve audio quality but overwrite the watermark).
"""

_NISQA_METRICS = ["nisqa_mos", "nisqa_noi", "nisqa_dis", "nisqa_col", "nisqa_loud"]

ATTACK_GROUPS = {
    "process_disruption": {
        "label": "Process Disruption Attacks",
        "attacks": [
            "CrossModelAttack",
            "CollusionAttack",
            "ZeroBitCollusionAttack",
            "Collusion2Attack",
            "SameModelAttack",
        ],
        "quality_metrics": ["pesq", "psnr", "si_sdr", "mcd", "visqol"],
        "intelligibility_metrics": ["stoi", "sii", "ncm"],
        "nisqa_metrics": _NISQA_METRICS,
    },
    "audio_editing": {
        "label": "Audio Editing Attacks",
        "attacks": [
            "CutSamplesAttack",
            "CropBeginningAttack",
            "CropRandomAttack",
            "WaveletAttack",
            "LowpassFilterAttack",
            "HighpassFilterAttack",
            "BandstopFilterAttack",
            "SmoothingAttack",
            "ChorusAttack",
            "FlangerAttack",
            "EchoAttack",
            "EqualizerAttack",
            "QuantizationAttack",
            "STFTQuantizationAttack",
            "PCMQuantizationAttack",
            "Mp3CompressionAttack",
            "EncodecAttack",
            "DescriptAudioCodecAttack",
            "OpusCodecAttack",
            "Codec2VocoderAttack",
            "ResamplingPolyAttack",
            "MixingAttack",
        ],
        "quality_metrics": ["pesq", "psnr", "si_sdr", "mcd", "visqol"],
        "intelligibility_metrics": ["stoi", "sii", "ncm"],
        "nisqa_metrics": _NISQA_METRICS,
    },
    "audio_distortion": {
        "label": "Audio Distortion Attacks",
        "attacks": [
            "GaussianNoiseAttack",
            "PinkNoiseAttack",
            "SignInversionAttack",
            "LPCAttack",
        ],
        "quality_metrics": ["pesq", "psnr", "si_sdr", "visqol"],
        "intelligibility_metrics": ["stoi", "sii", "ncm"],
        "nisqa_metrics": _NISQA_METRICS,
    },
    "desynchronization": {
        "label": "Desynchronization Attacks",
        "attacks": [
            "TimeStretchAttack",
            "PitchShiftAttack",
            "InvertedTimeStretchAttack",
            "ZeroCrossInsertsAttack",
            "FlipSamplesAttack",
            "ReplacementAttack",
        ],
        "quality_metrics": ["mcd", "visqol"],
        "intelligibility_metrics": [],
        "nisqa_metrics": _NISQA_METRICS,
    },
    "ai_attacks": {
        "label": "AI Attacks",
        "attacks": [
            "SpeechEnhancement1Attack",
            "SpeechEnhancement2Attack",
            "SpeechTokenizationAttack",
            "NeuralVocoderAttack",
            "DiffusionAttack",
        ],
        "quality_metrics": ["pesq", "mcd", "visqol"],
        "intelligibility_metrics": ["stoi", "sii", "ncm"],
        "nisqa_metrics": _NISQA_METRICS,
    },
    "transmission": {
        "label": "Transmission Attacks",
        "attacks": [
            "ReplayAttack",
            "NetworkTransmissionAttack"
        ],
        "quality_metrics": ["pesq", "psnr", "si_sdr", "mcd", "visqol"],
        "intelligibility_metrics": ["stoi", "sii", "ncm"],
        "nisqa_metrics": _NISQA_METRICS,
    },
}


def get_metrics_for_attack(attack_name):
    """Return the list of metric names relevant for ``attack_name``.

    Combines quality, intelligibility, and NISQA metrics. Attacks with
    no known group fall back to the full metric set so callers do not
    silently drop unknown attacks.
    """
    group_key = get_group_for_attack(attack_name)
    if group_key is None:
        from deepmarkpy.utils.metrics import ALL_METRICS
        return list(ALL_METRICS)
    group = ATTACK_GROUPS[group_key]
    return (
        list(group["quality_metrics"])
        + list(group["intelligibility_metrics"])
        + list(group.get("nisqa_metrics", []))
    )


def get_attacks_for_groups(group_names):
    """Return a flat list of attack names for the given group name(s).

    Args:
        group_names: A single group name or list of group names

    Returns:
        List of attack class names
    """
    if isinstance(group_names, str):
        group_names = [group_names]
    attacks = []
    for name in group_names:
        if name not in ATTACK_GROUPS:
            raise ValueError(
                f"Unknown attack group '{name}'. "
                f"Available: {list(ATTACK_GROUPS.keys())}"
            )
        attacks.extend(ATTACK_GROUPS[name]["attacks"])
    return attacks


def get_group_for_attack(attack_name):
    """Return the group key for a given attack name, or None.

    Handles expanded names like Codec2VocoderAttack_700 by stripping
    the trailing _<number> suffix when no exact match is found.
    """
    for group_key, group in ATTACK_GROUPS.items():
        if attack_name in group["attacks"]:
            return group_key
    # Try stripping bitrate suffix (e.g. Codec2VocoderAttack_700 -> Codec2VocoderAttack)
    base_name = "_".join(attack_name.rsplit("_", 1)[:-1]) if "_" in attack_name else None
    if base_name:
        for group_key, group in ATTACK_GROUPS.items():
            if base_name in group["attacks"]:
                return group_key
    return None


def group_attacks(attack_names):
    """Organize a list of attack names into their groups.

    Args:
        attack_names: List of attack class names

    Returns:
        Dict of {group_key: {"label": ..., "attacks": [...]}}
        Only includes groups that have at least one matching attack.
    """
    grouped = {}
    for attack in attack_names:
        group_key = get_group_for_attack(attack)
        if group_key is None:
            group_key = "other"
        if group_key not in grouped:
            label = ATTACK_GROUPS.get(group_key, {}).get("label", "Other Attacks")
            grouped[group_key] = {"label": label, "attacks": []}
        grouped[group_key]["attacks"].append(attack)
    return grouped
