import json
import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from deepmarkpy.utils.attack_groups import get_metric_caveat, group_attacks, ATTACK_GROUPS
from deepmarkpy.utils.latex_helpers import (
    build_longtable,
    compile_latex,
    display_attack_name,
    make_preamble,
)
from deepmarkpy.utils.metrics import (
    ALL_METRICS,
    INTELLIGIBILITY_METRICS,
    METRIC_LABELS,
    NISQA_METRICS,
    QUALITY_METRICS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-group metric relevance (read from attack_groups.py — single source)
# ---------------------------------------------------------------------------

GROUP_QUALITY_METRICS = {
    key: group.get("quality_metrics", []) for key, group in ATTACK_GROUPS.items()
}

GROUP_INTELLIGIBILITY_METRICS = {
    key: group.get("intelligibility_metrics", []) for key, group in ATTACK_GROUPS.items()
}

GROUP_NISQA_METRICS = {
    key: group.get("nisqa_metrics", []) for key, group in ATTACK_GROUPS.items()
}

# Audio Editing is handled via sub-groups instead of top-level metrics
from deepmarkpy.utils.attack_groups import _NISQA_METRICS

AUDIO_EDITING_SUBGROUPS = {
    "frequency_filtering": {
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

GROUP_DESCRIPTIONS = {
    "process_disruption": (
        "These attacks attempt to disrupt the watermarking process itself, "
        "including cross-model interference, collusion between multiple "
        "watermarked copies, and same-model re-watermarking."
    ),
    "audio_editing": (
        "Audio editing attacks simulate common audio processing operations "
        "that may be applied to watermarked content, ranging from filtering "
        "and effects to compression and temporal modifications."
    ),
    "audio_distortion": (
        "These attacks introduce various forms of noise and signal distortion, "
        "testing the watermark's resilience to additive interference and "
        "signal corruption."
    ),
    "desynchronization": (
        "Desynchronization attacks alter the temporal alignment of the audio "
        "signal through time-scaling, pitch shifting, and sample-level "
        "manipulations."
    ),
    "ai_attacks": (
        "AI-based attacks leverage neural networks and machine learning models "
        "to process the watermarked audio, potentially removing or degrading "
        "the embedded watermark."
    ),
    "transmission": (
        "These attacks simulate real-world audio transmission scenarios, "
        "including acoustic replay through speakers/microphones and "
        "network-based audio transmission."
    ),
}

GROUP_ORDER = [
    "process_disruption",
    "audio_editing",
    "audio_distortion",
    "desynchronization",
    "ai_attacks",
    "transmission",
]


class DetailedReportGenerator:
    """Generate detailed LaTeX reports with full quality metrics analysis."""

    BASE_QUALITY_METRICS = QUALITY_METRICS
    INTELLIGIBILITY_METRICS = INTELLIGIBILITY_METRICS
    ALL_METRIC_LABELS = METRIC_LABELS

    def __init__(self, report_dir="report"):
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)
        self._has_deepmark_cls = os.path.exists(
            os.path.join(self.report_dir, "deepmark.cls")
        )

    # ------------------------------------------------------------------
    # LaTeX helpers
    # ------------------------------------------------------------------

    def _preamble(self, title, author):
        return make_preamble(title, author, self._has_deepmark_cls,
                             extra_packages=("xcolor",))

    def _format_val(self, stats):
        """Format mean value for display."""
        if stats is None or stats.get("mean") is None:
            return "N/A"
        return f"{stats['mean']:.2f}"

    def _display_name(self, attack_name):
        return display_attack_name(attack_name)

    # ------------------------------------------------------------------
    # Data aggregation
    # ------------------------------------------------------------------

    def aggregate_results(self, results, is_zero_bit=False):
        """
        Aggregate per-file results into per-attack means.

        Args:
            results: Raw benchmark results dict (per file, per attack)
            is_zero_bit: When True, accuracy is treated as a boolean
                detection flag (0/1 per file) and the aggregate is
                reported as a count string ``"n/N"`` (files detected /
                files total) instead of a percentage.

        Returns:
            dict with structure:
            {
                "quality_metrics": [...],
                "intelligibility_metrics": [...],
                "watermarked_audio_quality": {metric: {"mean": ...}},
                "watermark_intelligibility": {metric: {"mean": ...}},
                "is_zero_bit": bool,
                "attacks": {
                    attack_name: {
                        "accuracy": {"mean": ..., "count": "n/N" (zero-bit)},
                        "attacked_audio_quality_wm": {metric: {"mean": ...}},
                        "attack_intelligibility": {metric: {"mean": ...}},
                    }
                }
            }
        """
        quality_metrics = list(self.BASE_QUALITY_METRICS)
        intelligibility_metrics = list(self.INTELLIGIBILITY_METRICS)
        all_metrics = quality_metrics + intelligibility_metrics

        watermark_values = {m: [] for m in all_metrics}
        attack_data = {}

        for filepath, file_data in results.items():
            # Watermark-only quality is stored once at file level (S5)
            wm_quality = file_data.get("watermarked_audio_quality") if isinstance(file_data, dict) else None
            if wm_quality and wm_quality != "N/A":
                for m in all_metrics:
                    val = wm_quality.get(m)
                    if val is not None and val != "N/A":
                        watermark_values[m].append(val)

            attacks = file_data.get("attacks", {}) if isinstance(file_data, dict) else {}
            for attack_name, data in attacks.items():
                if attack_name not in attack_data:
                    attack_data[attack_name] = {
                        "accuracy": [],
                        "confidence": [],
                        "metrics": {m: [] for m in all_metrics},
                    }

                attack_data[attack_name]["accuracy"].append(data["accuracy"])

                if "confidence" in data:
                    attack_data[attack_name]["confidence"].append(data["confidence"])

                attacked_quality = data.get("attacked_audio_quality_wm")
                if attacked_quality and attacked_quality != "N/A":
                    for m in all_metrics:
                        val = attacked_quality.get(m)
                        if val is not None and val != "N/A":
                            attack_data[attack_name]["metrics"][m].append(val)

        def mean_std(values):
            if not values:
                return {"mean": None, "std": None}
            return {"mean": float(np.mean(values)), "std": float(np.std(values))}

        aggregated = {
            "quality_metrics": quality_metrics,
            "intelligibility_metrics": intelligibility_metrics,
            "watermarked_audio_quality": {
                m: mean_std(watermark_values[m]) for m in quality_metrics
            },
            "watermark_intelligibility": {
                m: mean_std(watermark_values[m]) for m in intelligibility_metrics
            },
            "is_zero_bit": bool(is_zero_bit),
            "attacks": {},
        }

        for attack_name, data in attack_data.items():
            if is_zero_bit:
                valid = [a for a in data["accuracy"] if a is not None]
                n_total = len(valid)
                n_detected = int(sum(1 for v in valid if v))
                accuracy_entry = {
                    "mean": (n_detected / n_total * 100.0) if n_total else None,
                    "std": None,
                    "count": f"{n_detected}/{n_total}",
                }
            else:
                accuracy_entry = mean_std(data["accuracy"])

            entry = {
                "accuracy": accuracy_entry,
                "attacked_audio_quality_wm": {
                    m: mean_std(data["metrics"][m]) for m in quality_metrics
                },
                "attack_intelligibility": {
                    m: mean_std(data["metrics"][m]) for m in intelligibility_metrics
                },
            }
            if data["confidence"]:
                entry["confidence"] = mean_std(data["confidence"])

            aggregated["attacks"][attack_name] = entry

        return aggregated

    # ------------------------------------------------------------------
    # Generic table builders
    # ------------------------------------------------------------------

    def _accuracy_table(self, aggregated, attacks, caption, label):
        """Generate accuracy longtable for a subset of attacks.

        Zero-bit models additionally report a per-attack count "n/N"
        (files where the watermark was detected over total files) next
        to the percentage, since the per-file value collapses to 0/1
        and the count makes the underlying detection ratio explicit.
        """
        available = sorted([a for a in attacks if a in aggregated["attacks"]])
        if not available:
            return ""

        is_zero_bit = aggregated.get("is_zero_bit", False)
        has_confidence = any(
            "confidence" in aggregated["attacks"][a] for a in available
        )

        rows = []
        for a in available:
            data = aggregated["attacks"][a]
            display = self._display_name(a)
            acc = self._format_val(data["accuracy"])
            if is_zero_bit:
                count = data["accuracy"].get("count", "N/A")
                if has_confidence:
                    conf = (
                        self._format_val(data["confidence"])
                        if "confidence" in data
                        else "---"
                    )
                    rows.append(f"    {display} & {acc} & {count} & {conf} \\\\")
                else:
                    rows.append(f"    {display} & {acc} & {count} \\\\")
            else:
                if has_confidence:
                    conf = (
                        self._format_val(data["confidence"])
                        if "confidence" in data
                        else "---"
                    )
                    rows.append(f"    {display} & {acc} & {conf} \\\\")
                else:
                    rows.append(f"    {display} & {acc} \\\\")

        if is_zero_bit:
            if has_confidence:
                col_spec = "lccc"
                header = "Attack & Accuracy (\\%) & Detected (files) & Confidence"
            else:
                col_spec = "lcc"
                header = "Attack & Accuracy (\\%) & Detected (files)"
        else:
            if has_confidence:
                col_spec = "lcc"
                header = "Attack & Accuracy (\\%) & Confidence"
            else:
                col_spec = "lc"
                header = "Attack & Accuracy (\\%)"

        return build_longtable(col_spec, header, rows, caption, label)

    def _metrics_table(self, aggregated, attacks, metrics, data_key,
                       caption, label, baseline_data=None):
        """Generate metrics longtable for a subset of attacks and metrics.

        Args:
            aggregated: Aggregated results
            attacks: List of attack names to include
            metrics: List of metric keys to include
            data_key: Key in attack data ('attacked_audio_quality_wm' or
                      'attack_intelligibility')
            caption: Table caption
            label: Table label
            baseline_data: Optional watermark-only baseline dict (e.g.
                ``watermarked_audio_quality``). When provided, prepends a
                "No Attack (watermark only)" row so attack rows can be
                compared to the embedding-only condition.
        """
        if not metrics:
            return ""

        available = sorted([a for a in attacks if a in aggregated["attacks"]])
        if not available:
            return ""

        headers = [self.ALL_METRIC_LABELS[m] for m in metrics]
        header_str = " & ".join(headers)
        col_spec = "l" + "c" * len(metrics)

        rows = []
        if baseline_data:
            baseline_cols = " & ".join(
                self._format_val(baseline_data.get(m, {"mean": None}))
                for m in metrics
            )
            rows.append(
                f"    No Attack (watermark only) & {baseline_cols} \\\\"
            )
            rows.append("    \\midrule")

        any_unreliable = False
        for a in available:
            data = aggregated["attacks"][a][data_key]
            display = self._display_name(a)
            cells = []
            for m in metrics:
                cell = self._format_val(data.get(m, {"mean": None}))
                if cell != "N/A" and get_metric_caveat(a, m):
                    # Reported, but this attack makes the metric unreliable.
                    cell += "\\textsuperscript{\\dag}"
                    any_unreliable = True
                cells.append(cell)
            rows.append(f"    {display} & " + " & ".join(cells) + " \\\\")

        if any_unreliable:
            caption += (
                " \\textsuperscript{\\dag}Sample-aligned metric on a "
                "time-shifting attack: the value reflects the shift, not a "
                "quality change."
            )

        return build_longtable(
            col_spec, f"Condition & {header_str}", rows, caption, label,
        )

    # ------------------------------------------------------------------
    # Report section builders
    # ------------------------------------------------------------------

    def _generate_audio_editing_section(self, aggregated, group_attack_list):
        """Generate the Audio Editing section with sub-group subsections."""
        sections = ""

        # Sub-sections
        for sub_key, sub_info in AUDIO_EDITING_SUBGROUPS.items():
            sub_attacks = []
            for base_name in sub_info["attacks"]:
                for result_name in aggregated["attacks"]:
                    if result_name == base_name or result_name.startswith(base_name + "_"):
                        sub_attacks.append(result_name)
            if not sub_attacks:
                continue

            q_metrics = sub_info.get("quality_metrics", [])
            i_metrics = sub_info.get("intelligibility_metrics", [])
            n_metrics = sub_info.get("nisqa_metrics", [])

            if not q_metrics and not i_metrics and not n_metrics:
                continue

            sections += "\\needspace{5\\baselineskip}\n"
            sections += f"\\subsection{{{sub_info['label']}}}\n\n"
            sections += f"{sub_info['description']}\n\n"

            if q_metrics:
                sections += self._metrics_table(
                    aggregated, sub_attacks, q_metrics,
                    "attacked_audio_quality_wm",
                    f"Audio quality --- {sub_info['label']}.",
                    f"tab:qual_{sub_key}",
                    baseline_data=aggregated["watermarked_audio_quality"],
                )
                sections += "\n\n"

            if i_metrics:
                sections += self._metrics_table(
                    aggregated, sub_attacks, i_metrics,
                    "attack_intelligibility",
                    f"Speech intelligibility --- {sub_info['label']}.",
                    f"tab:intell_{sub_key}",
                    baseline_data=aggregated["watermark_intelligibility"],
                )
                sections += "\n\n"

            if n_metrics:
                sections += self._metrics_table(
                    aggregated, sub_attacks, n_metrics,
                    "attacked_audio_quality_wm",
                    f"NISQA non-intrusive quality dimensions --- {sub_info['label']}.",
                    f"tab:nisqa_{sub_key}",
                    baseline_data=aggregated["watermarked_audio_quality"],
                )
                sections += "\n\n"

        return sections

    def _generate_grouped_sections(self, aggregated):
        """Generate per-group sections with relevant metrics."""
        attack_names = list(aggregated["attacks"].keys())
        grouped = group_attacks(attack_names)

        sections = ""
        for group_key in GROUP_ORDER:
            if group_key not in grouped:
                continue

            group_info = grouped[group_key]
            group_label = ATTACK_GROUPS.get(group_key, {}).get(
                "label", group_info["label"]
            )
            group_attack_list = group_info["attacks"]
            description = GROUP_DESCRIPTIONS.get(group_key, "")

            sections += "\\needspace{5\\baselineskip}\n"
            sections += f"\\section{{{group_label}}}\n\n"
            if description:
                sections += f"{description}\n\n"

            if group_key == "audio_editing":
                sections += self._generate_audio_editing_section(
                    aggregated, group_attack_list
                )
            else:
                q_metrics = GROUP_QUALITY_METRICS.get(group_key, [])
                i_metrics = GROUP_INTELLIGIBILITY_METRICS.get(group_key, [])
                n_metrics = GROUP_NISQA_METRICS.get(group_key, [])

                if q_metrics:
                    sections += self._metrics_table(
                        aggregated, group_attack_list, q_metrics,
                        "attacked_audio_quality_wm",
                        f"Audio quality --- {group_label}.",
                        f"tab:qual_{group_key}",
                        baseline_data=aggregated["watermarked_audio_quality"],
                    )
                    sections += "\n\n"

                if i_metrics:
                    sections += self._metrics_table(
                        aggregated, group_attack_list, i_metrics,
                        "attack_intelligibility",
                        f"Speech intelligibility --- {group_label}.",
                        f"tab:intell_{group_key}",
                        baseline_data=aggregated[
                            "watermark_intelligibility"
                        ],
                    )
                    sections += "\n\n"

                if n_metrics:
                    sections += self._metrics_table(
                        aggregated, group_attack_list, n_metrics,
                        "attacked_audio_quality_wm",
                        f"NISQA non-intrusive quality dimensions --- {group_label}.",
                        f"tab:nisqa_{group_key}",
                        baseline_data=aggregated["watermarked_audio_quality"],
                    )
                    sections += "\n\n"

        # Handle attacks not in any known group
        if "other" in grouped:
            other_attacks = grouped["other"]["attacks"]
            # Split quality metrics from NISQA so the audio-quality table
            # stays narrow enough to fit the page; NISQA's five dimensions
            # get their own table, matching every other group's layout.
            other_quality_metrics = [
                m for m in self.BASE_QUALITY_METRICS if m not in NISQA_METRICS
            ]

            sections += "\\needspace{5\\baselineskip}\n"
            sections += "\\section{Other Attacks}\n\n"
            sections += self._metrics_table(
                aggregated, other_attacks, other_quality_metrics,
                "attacked_audio_quality_wm",
                "Audio quality --- other attacks.",
                "tab:qual_other",
                baseline_data=aggregated["watermarked_audio_quality"],
            )
            sections += "\n\n"
            sections += self._metrics_table(
                aggregated, other_attacks, self.INTELLIGIBILITY_METRICS,
                "attack_intelligibility",
                "Speech intelligibility --- other attacks.",
                "tab:intell_other",
                baseline_data=aggregated["watermark_intelligibility"],
            )
            sections += "\n\n"
            sections += self._metrics_table(
                aggregated, other_attacks, list(NISQA_METRICS),
                "attacked_audio_quality_wm",
                "NISQA non-intrusive quality dimensions --- other attacks.",
                "tab:nisqa_other",
                baseline_data=aggregated["watermarked_audio_quality"],
            )
            sections += "\n\n"

        return sections

    # ------------------------------------------------------------------
    # Main report assembly
    # ------------------------------------------------------------------

    def generate_latex_report(self, aggregated, model_name="DeepMark",
                             crop_before_attack=None):
        """Generate complete LaTeX document.

        Args:
            aggregated: Aggregated results from aggregate_results()
            model_name: Name of the watermarking model
            crop_before_attack: If set, percentage cropped before attacks
        """
        num_attacks = len(aggregated["attacks"])
        all_attacks = list(aggregated["attacks"].keys())
        is_single = num_attacks == 1
        attack_word = "attack type" if is_single else "attack types"
        across_phrase = (
            f"a single attack type" if is_single
            else f"{num_attacks} attack types"
        )

        preamble = self._preamble(
            f"Detailed Benchmark Report: {model_name}",
            "DeepMark Benchmark System",
        )

        crop_note = ""
        if crop_before_attack is not None:
            crop_note = (
                f" \\textcolor{{red}}{{A crop of {crop_before_attack:.1f}\\% was applied to the "
                f"beginning of the watermarked audio prior to each attack. "
                f"The original (reference) audio was cropped identically, so "
                f"quality metrics compare cropped original vs.\\ cropped attacked "
                f"audio, and BER is measured by detecting the watermark from the "
                f"cropped attacked signal.}}"
            )

        abstract = (
            f"\\begin{{abstract}}\n"
            f"This report presents a detailed evaluation of the "
            f"{model_name} watermarking model across {across_phrase}. "
            f"It covers watermark detection robustness, audio quality "
            f"impact analysis, and speech intelligibility measures, comparing "
            f"the effects of watermark embedding and adversarial "
            f"{'attack' if is_single else 'attacks'} on the audio signal."
            f"{crop_note}\n"
            f"\\end{{abstract}}\n"
        )

        # Overall accuracy table + bar chart at the top
        robustness = "\\section{Watermark Detection Robustness}\n\n"
        robustness += (
            f"The following table reports the mean detection accuracy "
            f"per {attack_word}.\n\n"
        )
        robustness += self._accuracy_table(
            aggregated, all_attacks,
            f"Watermark detection robustness per {attack_word}.",
            "tab:robustness",
        )
        robustness += "\n\n"

        chart_path = os.path.join(self.report_dir, "benchmark_chart.png")
        if os.path.exists(chart_path):
            robustness += (
                "\\begin{figure}[H]\n"
                "    \\centering\n"
                "    \\includegraphics[width=\\linewidth]"
                "{benchmark_chart.png}\n"
                "    \\caption{Watermark detection accuracy by attack "
                "type.}\n"
                "    \\label{fig:accuracy_chart}\n"
                "\\end{figure}\n\n"
            )

        # Per-group metric sections (only groups with attacks are shown)
        body = self._generate_grouped_sections(aggregated)

        return (
            f"{preamble}\n\n"
            f"{abstract}\n"
            f"{robustness}"
            f"{body}"
            f"\\end{{document}}"
        )

    def generate_full_report(self, results, model_name="DeepMark",
                              is_zero_bit=False, crop_before_attack=None):
        """
        Generate complete detailed report from raw benchmark results.

        Args:
            results: Raw benchmark results dict from Benchmark.run()
            model_name: Name of the watermarking model
            is_zero_bit: When True, render accuracy as "n/N" detection
                counts (zero-bit detector returns 0/1 per file, so a
                percentage column would only ever show 0 or 100).
            crop_before_attack: If set, percentage cropped before attacks

        Returns:
            Path to the generated ``.tex`` file.
        """
        aggregated = self.aggregate_results(results, is_zero_bit=is_zero_bit)

        latex_content = self.generate_latex_report(aggregated, model_name,
                                                   crop_before_attack=crop_before_attack)
        latex_path = os.path.join(self.report_dir, "detailed_report.tex")
        with open(latex_path, "w") as f:
            f.write(latex_content)

        logger.info(f"Detailed report saved to {latex_path}")

        compile_latex(self.report_dir, "detailed_report")

        return latex_path
