"""Attack group definitions for the DeepMark Benchmark.

``ATTACK_GROUPS`` is the attack *selection* taxonomy: it maps a group
name to the attacks it contains, and declares which quality /
intelligibility metrics that family makes sense for. Those metric lists
are the built-in default matrix (``MetricResolver.from_attack_groups``).
The shipped benchmark and detection_reliability templates restate them by
hand, and a test fails if the two drift.

``ATTACK_SUBGROUPS`` refines one group -- ``audio_editing`` -- into four
report-only subsections whose metric lists deliberately differ from the
parent's. Subgroups are not selectable as attack groups; they exist so
the detailed report can, for example, suppress every metric for
length-changing edits. Both levels are configurable: see
``CONFIG_GROUP_KEYS``.

``get_metric_caveat`` names the values an attack makes unreadable -- a
sample-aligned metric under desynchronization, SI-SDR under
SignInversion. Reports print them marked, with the reason in a footnote.

A metric a group's lists omit is off for that group, except PESQ, ViSQOL
and STOI, which every top-level group reports.
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
            "AdditiveNoiseAttack",
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
            "VAEAttack",
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


# Report-only refinement of ``audio_editing``. These are NOT selectable as
# attack groups -- ``attacks.groups`` still takes "audio_editing" -- but each
# is a valid ``metrics.per_group`` key, and the detailed report renders
# audio_editing exclusively through these subsections.
#
# The metric lists deliberately differ from the parent group: temporal edits
# change signal length, which makes every reference-aligned metric report the
# length change rather than a quality loss, so all of them are off.
ATTACK_SUBGROUPS = {
    "frequency_filtering": {
        "parent": "audio_editing",
        "label": "Frequency Filtering",
        "attacks": [
            "LowpassFilterAttack",
            "HighpassFilterAttack",
            "BandstopFilterAttack",
            "EqualizerAttack",
        ],
        "quality_metrics": ["pesq", "mcd", "visqol"],
        "intelligibility_metrics": ["stoi", "sii", "ncm"],
        "nisqa_metrics": _NISQA_METRICS,
        "description": (
            "Frequency-domain modifications that selectively attenuate "
            "or boost spectral content."
        ),
    },
    "temporal_editing": {
        "parent": "audio_editing",
        "label": "Temporal Editing",
        "attacks": [
            "CutSamplesAttack",
            "CropBeginningAttack",
            "CropRandomAttack",
        ],
        "quality_metrics": [],
        "intelligibility_metrics": [],
        "nisqa_metrics": [],
        "description": (
            "Attacks that modify the temporal structure of the audio signal "
            "by removing or rearranging samples. Quality and intelligibility "
            "metrics are not reported for these attacks, as changes in signal "
            "length make direct metric comparison unreliable."
        ),
    },
    "audio_effects": {
        "parent": "audio_editing",
        "label": "Audio Effects",
        "attacks": [
            "WaveletAttack",
            "SmoothingAttack",
            "ChorusAttack",
            "FlangerAttack",
            "EchoAttack",
            "MixingAttack",
        ],
        "quality_metrics": ["pesq", "si_sdr", "mcd", "visqol"],
        "intelligibility_metrics": ["stoi", "sii", "ncm"],
        "nisqa_metrics": _NISQA_METRICS,
        "description": (
            "Common audio processing effects that alter signal characteristics "
            "while preserving perceptual quality."
        ),
    },
    "compression_quantization": {
        "parent": "audio_editing",
        "label": "Compression \\& Quantization",
        "attacks": [
            "QuantizationAttack",
            "STFTQuantizationAttack",
            "PCMQuantizationAttack",
            "Mp3CompressionAttack",
            "EncodecAttack",
            "DescriptAudioCodecAttack",
            "OpusCodecAttack",
            "Codec2VocoderAttack",
            "ResamplingPolyAttack",
        ],
        "quality_metrics": ["pesq", "psnr", "mcd", "visqol"],
        "intelligibility_metrics": ["stoi", "sii", "ncm"],
        "nisqa_metrics": _NISQA_METRICS,
        "description": (
            "Lossy compression and bit-depth reduction operations commonly "
            "encountered in audio distribution pipelines."
        ),
    },
}

# Attacks that belong to no declared group are collected under this key. It is
# a valid ``metrics.per_group`` key so a third-party plugin's metrics can be
# configured without editing this file.
OTHER_GROUP_KEY = "other"

# Every key ``metrics.per_group`` accepts. Groups first (so "did you mean"
# suggestions prefer the selectable names), then subgroups, then "other".
CONFIG_GROUP_KEYS = (
    tuple(ATTACK_GROUPS) + tuple(ATTACK_SUBGROUPS) + (OTHER_GROUP_KEY,)
)

# Order sections appear in every report that groups by attack family.
GROUP_ORDER = tuple(ATTACK_GROUPS)


# Metrics that compare the reference and degraded signals sample-by-sample.
# They report timing offset as if it were quality loss: on a speech-like
# signal a 20-sample (1.25 ms) shift already moves STOI from ~1.0 to 0.66 and
# a 1600-sample shift takes NCM from 0.999 to 0.188, while MCD grows without
# bound. PESQ, ViSQOL and SII align internally and are not listed. NISQA is
# reference-free and therefore immune.
_ALIGNMENT_SENSITIVE_METRICS = ("psnr", "si_sdr", "stoi", "mcd", "ncm")

# Each caveat completes the sentence "This metric ..." in a report footnote.
_DESYNC_CAVEAT = (
    "compares samples by index, so it reports the attack's timing shift "
    "rather than a quality change"
)

# Per-attack overrides for metrics that are uninformative for a specific
# attack rather than for a whole family.
_ATTACK_METRIC_CAVEATS = {
    # SI-SDR is scale-invariant and a sign flip is a scale of -1, so the
    # residual collapses and the score is pinned at the epsilon ceiling.
    "SignInversionAttack": {
        "si_sdr": "is scale-invariant, so it cannot see polarity inversion",
    },
}


def get_metric_caveat(attack_name, metric):
    """Return why ``metric`` is unreliable for ``attack_name``, or None.

    The value is reported as usual; this states what a reader should not
    conclude from it. Desynchronization attacks move the time axis, which
    the sample-aligned metrics register as damage. Expanded and version
    names (``SignInversionAttack (mild)``) carry their attack's caveats.
    """
    for candidate in _name_candidates(attack_name):
        attack_caveats = _ATTACK_METRIC_CAVEATS.get(candidate, {})
        if metric in attack_caveats:
            return attack_caveats[metric]
    if (
        get_group_for_attack(attack_name) == "desynchronization"
        and metric in _ALIGNMENT_SENSITIVE_METRICS
    ):
        return _DESYNC_CAVEAT
    return None


def group_definition(group_key):
    """Return the group or subgroup definition for ``group_key``, or None."""
    if group_key in ATTACK_GROUPS:
        return ATTACK_GROUPS[group_key]
    return ATTACK_SUBGROUPS.get(group_key)


def group_label(group_key, default=None):
    """Human-readable label for a group, subgroup, or ``other``."""
    if group_key == OTHER_GROUP_KEY:
        return default or "Other Attacks"
    definition = group_definition(group_key)
    if definition is None:
        return default or group_key
    return definition["label"]


def group_parent(group_key):
    """Return the parent group of a subgroup, or None for a top-level group.

    Used by the metric resolver: a subgroup inherits from its parent group
    before falling back to ``metrics.defaults``.
    """
    definition = ATTACK_SUBGROUPS.get(group_key)
    return definition["parent"] if definition else None


def subgroups_of(group_key):
    """Return the subgroup keys refining ``group_key``, in declaration order."""
    return [
        key for key, definition in ATTACK_SUBGROUPS.items()
        if definition["parent"] == group_key
    ]


def get_subgroup_for_attack(attack_name):
    """Return the subgroup key for ``attack_name``, or None.

    Applies the same suffix stripping as ``get_group_for_attack`` so
    expanded names (``Codec2VocoderAttack_700``) and version display names
    (``EchoAttack (mild)``) resolve to their subgroup.
    """
    for candidate in _name_candidates(attack_name):
        for key, definition in ATTACK_SUBGROUPS.items():
            if candidate in definition["attacks"]:
                return key
    return None


def _name_candidates(attack_name):
    """Yield ``attack_name`` and the base names it may have been expanded from.

    Order matters: the exact name wins over a suffix-stripped guess.
    """
    yield attack_name

    stripped = attack_name
    if " (" in attack_name and attack_name.endswith(")"):
        stripped = attack_name[:attack_name.index(" (")]
        yield stripped

    if "_" in stripped:
        yield "_".join(stripped.rsplit("_", 1)[:-1])


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
    the trailing _<number> suffix, and version display names like
    "GaussianNoiseAttack (aggressive)" by stripping the parenthetical.
    """
    for candidate in _name_candidates(attack_name):
        for group_key, group in ATTACK_GROUPS.items():
            if candidate in group["attacks"]:
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
