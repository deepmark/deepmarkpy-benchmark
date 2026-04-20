"""Attack group definitions for the DeepMark Benchmark."""

ATTACK_GROUPS = {
    "process_disruption": {
        "label": "Process Disruption Attacks",
        "attacks": [
            "CrossModelAttack",
            "CollusionAttack",
            "ZeroBitCollusionAttack",
            "SameModelAttack",
        ],
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
            "ResamplingPolyAttack",
            "MixingAttack",
        ],
    },
    "audio_distortion": {
        "label": "Audio Distortion Attacks",
        "attacks": [
            "GaussianNoiseAttack",
            "PinkNoiseAttack",
            "AdditiveNoiseAttack",
            "SignInversionAttack",
            "LPCAttack",
        ],
    },
    "desynchronization": {
        "label": "Desynchronization Attacks",
        "attacks": [
            "TimeStretchAttack",
            "PitchShiftAttack",
            "InvertedTimeStretch",
            "ZeroCrossInsertsAttack",
            "FlipSamplesAttack",
            "ReplacementAttack",
        ],
    },
    "ai_attacks": {
        "label": "AI Attacks",
        "attacks": [
            "SpeechEnhancement1Attack",
            "SpeechEnhancement2Attack",
            "SpeechTokenizationAttack",
            "NeuralVocoderAttack",
            "DiffusionAttack",
            "VAEAttack",
        ],
    },
    "transmission": {
        "label": "Transmission Attacks",
        "attacks": [
            "ReplayAttack",
            "NetworkTransmissionAttack",
        ],
    },
}


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
    """Return the group key for a given attack name, or None."""
    for group_key, group in ATTACK_GROUPS.items():
        if attack_name in group["attacks"]:
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
