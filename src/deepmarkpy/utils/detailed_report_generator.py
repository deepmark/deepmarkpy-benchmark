"""The detailed benchmark report: per-attack-family metric breakdowns.

One section per attack family; ``audio_editing`` is split into the
subsections ``ATTACK_SUBGROUPS`` declares, each configurable under
``metrics.per_group`` like a family. Columns come from the config's
``MetricResolver``, and every metric table opens with a "No Attack
(watermark only)" baseline row.
"""

import logging
import os

import numpy as np

from deepmarkpy.utils.attack_groups import (
    GROUP_ORDER,
    OTHER_GROUP_KEY,
    get_subgroup_for_attack,
    group_attacks,
    group_label,
    subgroups_of,
    ATTACK_SUBGROUPS,
)
from deepmarkpy.utils.latex_helpers import (
    MetricCaveats,
    compact_header,
    compile_latex,
    container_section,
    crop_note,
    display_attack_name,
    duration_label_tex,
    efficiency_tables,
    embedding_cost_line,
    format_emr_cell,
    format_metric_cell,
    grid_table,
    make_preamble,
    part_heading,
    slugify,
    metric_label,
    stat_header,
)
from deepmarkpy.utils.metric_resolver import (
    INTELLIGIBILITY_METRICS,
    PER_FILE_EFFICIENCY_METRICS,
    MetricResolver,
    NISQA_METRICS,
    QUALITY_METRICS,
    compute_statistics,
)

logger = logging.getLogger(__name__)


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
    OTHER_GROUP_KEY: (
        "Attacks that belong to no declared family, including any supplied by "
        "third-party plugins. They are configured under the \\texttt{other} "
        "key in \\texttt{metrics.per\\_group}."
    ),
}

# Metric families, each tabled separately.
_METRIC_SECTIONS = (
    ("qual", "Audio quality", QUALITY_METRICS),
    ("intell", "Speech intelligibility", INTELLIGIBILITY_METRICS),
    ("nisqa", "NISQA non-intrusive quality dimensions", NISQA_METRICS),
)


class DetailedReportGenerator:
    """Generate detailed LaTeX reports with full quality metrics analysis."""

    def __init__(self, report_dir="report", resolver=None):
        """
        Args:
            report_dir: where the ``.tex`` and ``.pdf`` are written.
            resolver: decides every table's metrics and statistic columns;
                defaults to the built-in ``ATTACK_GROUPS`` matrix.
        """
        self.report_dir = report_dir
        self.resolver = resolver or MetricResolver.from_attack_groups()
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

    @staticmethod
    def _display_name(attack_name):
        return display_attack_name(attack_name)

    # ------------------------------------------------------------------
    # Data aggregation
    # ------------------------------------------------------------------

    def aggregate_results(self, results, is_zero_bit=False):
        """
        Aggregate per-file results into per-attack statistics.

        Args:
            results: Raw benchmark results dict (per file, per attack)
            is_zero_bit: When True, accuracy is treated as a boolean
                detection flag (0/1 per file) and the aggregate also
                carries a count string ``"n/N"`` (files detected /
                files total).

        Returns:
            dict with ``watermarked_audio_quality`` (the no-attack
            baseline), ``is_zero_bit``, and ``attacks`` mapping each
            attack to its accuracy, metric and timing statistics.
        """
        metrics = self.resolver.all_signal_metrics()
        # Timings sit on the attack entry, not in its quality dict.
        timing_metrics = self.resolver.metrics_for_group(
            None, bucket="efficiency",
        )

        watermark_values = {m: [] for m in metrics}
        attack_data = {}

        for _, file_data in results.items():
            if not isinstance(file_data, dict):
                continue

            wm_quality = file_data.get("watermarked_audio_quality")
            if wm_quality and wm_quality != "N/A":
                for m in metrics:
                    value = wm_quality.get(m)
                    if value is not None and value != "N/A":
                        watermark_values[m].append(value)

            for attack_name, data in file_data.get("attacks", {}).items():
                if attack_name not in attack_data:
                    attack_data[attack_name] = {
                        "accuracy": [], "confidence": [],
                        "metrics": {m: [] for m in metrics},
                        "timings": {m: [] for m in timing_metrics},
                    }

                attack_data[attack_name]["accuracy"].append(data["accuracy"])
                if "confidence" in data:
                    attack_data[attack_name]["confidence"].append(data["confidence"])

                attacked_quality = data.get("attacked_audio_quality_wm")
                if attacked_quality and attacked_quality != "N/A":
                    for m in metrics:
                        value = attacked_quality.get(m)
                        if value is not None and value != "N/A":
                            attack_data[attack_name]["metrics"][m].append(value)

                for m in timing_metrics:
                    value = data.get(m)
                    if value is not None and value != "N/A":
                        attack_data[attack_name]["timings"][m].append(value)

        aggregated = {
            "watermarked_audio_quality": {
                m: _statistics(watermark_values[m], metric=m) for m in metrics
            },
            "is_zero_bit": bool(is_zero_bit),
            "attacks": {},
        }

        for attack_name, data in attack_data.items():
            entry = {
                "accuracy": _statistics(data["accuracy"], zero_bit=is_zero_bit),
                "metrics": {
                    m: _statistics(data["metrics"][m], metric=m) for m in metrics
                },
                "timings": {
                    m: _statistics(data["timings"][m], metric=m) for m in timing_metrics
                },
            }
            if data["confidence"]:
                entry["confidence"] = _statistics(data["confidence"])
            aggregated["attacks"][attack_name] = entry

        return aggregated

    # ------------------------------------------------------------------
    # Table builders
    # ------------------------------------------------------------------

    def _accuracy_table(self, aggregated, attacks, group_key, caption, label,
                        subject=None):
        """Accuracy statistics for a subset of attacks.

        Zero-bit models add a "Detected" ``n/N`` count, since their per-file
        accuracy is 0 or 100. BER with several statistics follows in a table
        of its own, captioned with ``subject`` or else ``caption``.
        """
        available = sorted(a for a in attacks if a in aggregated["attacks"])
        if not available:
            return ""

        is_zero_bit = aggregated.get("is_zero_bit", False)
        has_confidence = any(
            "confidence" in aggregated["attacks"][a] for a in available
        )
        statistics = self.resolver.statistics_for(group_key, "accuracy")

        ber_statistics = self.resolver.statistics_for(group_key, "ber")
        show_ber = not is_zero_bit and self.resolver.is_enabled(group_key, "ber")
        inline_ber = show_ber and len(ber_statistics) == 1
        show_emr = self.resolver.is_enabled(group_key, "emr")

        headers = [stat_header(s) for s in statistics]
        if is_zero_bit:
            headers.append("Detected")
        if inline_ber:
            headers.append(metric_label("ber"))
        if show_emr:
            headers.append(metric_label("emr"))
        if has_confidence:
            headers.append("Confidence")

        rows = []
        for attack in available:
            data = aggregated["attacks"][attack]
            accuracy = data["accuracy"]
            cells = [format_metric_cell("accuracy", accuracy.get(s), "--")
                     for s in statistics]
            if is_zero_bit:
                cells.append(accuracy.get("count", "N/A"))
            if inline_ber:
                cells.append(format_metric_cell(
                    "ber", _ber_statistic(accuracy, ber_statistics[0]), "--",
                ))
            if show_emr:
                cells.append(format_emr_cell(
                    accuracy.get("emr_count"), accuracy.get("n"),
                    accuracy.get("emr_rate"),
                ))
            if has_confidence:
                confidence = data.get("confidence", {}).get("mean")
                cells.append(
                    f"{confidence:.2f}" if confidence is not None else "---"
                )
            rows.append((self._display_name(attack), cells))

        tables = [grid_table("Attack", headers, rows, caption, label)]

        if show_ber and not inline_ber:
            ber_rows = [
                (self._display_name(attack), [
                    format_metric_cell("ber", _ber_statistic(
                        aggregated["attacks"][attack]["accuracy"], s), "--")
                    for s in ber_statistics
                ])
                for attack in available
            ]
            tables.append(grid_table(
                "Attack", [stat_header(s) for s in ber_statistics], ber_rows,
                f"Bit error rate --- {subject or caption}", f"{label}_ber",
            ))
        return "\n\n".join(tables)

    def _metric_table(self, aggregated, available, columns, headers, caption,
                      label):
        """A row per attack and a column per ``(metric, statistic)``.

        The no-attack baseline row comes first when any of its cells has a
        value. A metric's first statistic carries the caveat mark.
        """
        baseline = aggregated.get("watermarked_audio_quality") or {}
        baseline_cells = [
            format_metric_cell(metric, (baseline.get(metric) or {}).get(statistic))
            for metric, statistic in columns
        ]
        rows = []
        if any(cell != "N/A" for cell in baseline_cells):
            rows += [("No Attack (watermark only)", baseline_cells),
                     "    \\midrule"]

        caveats = MetricCaveats()
        for attack in available:
            attack_metrics = aggregated["attacks"][attack]["metrics"]
            first = {}
            cells = []
            for metric, statistic in columns:
                first.setdefault(metric, statistic)
                value = (attack_metrics.get(metric) or {}).get(statistic)
                if value is None:
                    cells.append("N/A")
                    continue
                cell = format_metric_cell(metric, value)
                if statistic == first[metric]:
                    cell += caveats.mark(attack, metric)
                cells.append(cell)
            rows.append((self._display_name(attack), cells))

        if caveats.any_flagged:
            caption += " " + caveats.footnote().strip()
        return grid_table("Condition", headers, rows, caption, label)

    def _embedding_cost_line(self, aggregated):
        """The per-file embedding time, stated once for the run, or ''."""
        metric = "embed_latency"
        if not self.resolver.is_enabled(None, metric):
            return ""

        # Every attack entry records the same per-file cost.
        statistics = self.resolver.statistics_for(None, metric)
        for data in aggregated["attacks"].values():
            line = embedding_cost_line(
                (data.get("timings") or {}).get(metric) or {}, statistics,
            )
            if line:
                return line
        return ""

    def _efficiency_table(self, aggregated, attacks, group_key, label_text,
                          label_key):
        """Timing tables for one section; embedding time is stated for the run."""
        available = sorted(a for a in attacks if a in aggregated["attacks"])
        metrics = [
            m for m in self.resolver.metrics_for_group(None,
                                                       bucket="efficiency")
            if m not in PER_FILE_EFFICIENCY_METRICS
            and any((aggregated["attacks"][a].get("timings") or {}).get(m)
                    for a in available)
        ]
        return efficiency_tables(
            "Attack",
            [(self._display_name(a), aggregated["attacks"][a].get("timings") or {})
             for a in available],
            metrics,
            lambda metric: self.resolver.statistics_for(group_key, metric),
            f"--- {label_text}", f"tab:efficiency_{label_key}",
        )

    def _metric_tables(self, aggregated, attacks, group_key, label_text,
                       label_key):
        """Every metric table for one group or subgroup, and its silent metrics."""
        available = sorted(a for a in attacks if a in aggregated["attacks"])
        if not available:
            return "", []

        blocks = []
        silent = []
        for section_key, section_title, family in _METRIC_SECTIONS:
            enabled = [
                m for m in self.resolver.signal_metrics_for_group(group_key)
                if m in family
            ]
            if not enabled:
                continue

            with_data = [
                m for m in enabled
                if any(
                    (aggregated["attacks"][a]["metrics"].get(m) or {}).get("mean")
                    is not None
                    for a in available
                )
            ]
            silent += [m for m in enabled if m not in with_data]

            single = []
            for metric in with_data:
                statistics = self.resolver.statistics_for(group_key, metric)
                if len(statistics) > 1:
                    blocks.append(self._metric_table(
                        aggregated, available, [(metric, s) for s in statistics],
                        [stat_header(s) for s in statistics],
                        f"{metric_label(metric)} --- {label_text}.",
                        f"tab:{section_key}_{label_key}_{metric}",
                    ))
                else:
                    single.append((metric, statistics[0]))

            if single:
                blocks.append(self._metric_table(
                    aggregated, available, single,
                    [compact_header(m, s) for m, s in single],
                    f"{section_title} --- {label_text}.",
                    f"tab:{section_key}_{label_key}",
                ))

        return "\n\n".join(blocks), silent

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _group_section(self, aggregated, group_key, attacks, label_key):
        """One group's section: accuracy, then metrics (or subsections)."""
        available = [a for a in attacks if a in aggregated["attacks"]]
        if not available:
            return ""

        label_text = group_label(group_key)
        section = "\\needspace{5\\baselineskip}\n"
        section += f"\\section{{{label_text}}}\n\n"
        description = GROUP_DESCRIPTIONS.get(group_key, "")
        if description:
            section += f"{description}\n\n"

        section += self._accuracy_table(
            aggregated, available, group_key,
            f"Watermark detection robustness --- {label_text}.",
            f"tab:robustness_{label_key}",
            subject=f"{label_text}.",
        )
        section += "\n\n"

        subgroups = subgroups_of(group_key)
        if subgroups:
            section += self._subgroup_sections(
                aggregated, subgroups, available, label_key,
            )
            # The timing tables cover the whole family, after its subsections.
            section += self._efficiency_table(
                aggregated, available, group_key, label_text, label_key,
            )
            return section

        body, silent = self._metric_tables(
            aggregated, available, group_key, label_text, label_key,
        )
        section += body
        section += _silent_note(silent)
        section += self._efficiency_table(
            aggregated, available, group_key, label_text, label_key,
        )
        return section + "\n\n"

    def _subgroup_sections(self, aggregated, subgroups, group_attack_list,
                           label_key):
        """One subsection per subgroup with attacks in the report."""
        sections = ""
        for subgroup in subgroups:
            definition = ATTACK_SUBGROUPS[subgroup]
            members = [
                attack for attack in group_attack_list
                if get_subgroup_for_attack(attack) == subgroup
            ]
            if not members:
                continue

            sections += "\\needspace{5\\baselineskip}\n"
            sections += f"\\subsection{{{definition['label']}}}\n\n"
            sections += f"{definition['description']}\n\n"

            body, silent = self._metric_tables(
                aggregated, members, subgroup, definition["label"],
                f"{label_key}_{subgroup}",
            )
            if body:
                sections += body + "\n\n"
            elif not self.resolver.signal_metrics_for_group(subgroup):
                sections += (
                    "\\noindent No quality or intelligibility metric is "
                    "enabled for this subsection in the configuration.\n\n"
                )
            sections += _silent_note(silent)
        return sections

    def _generate_body(self, aggregated, label_suffix=""):
        """Per-group sections, in the taxonomy's order.

        ``label_suffix`` keeps one duration part's ``\\label`` names apart
        from another's.
        """
        grouped = group_attacks(list(aggregated["attacks"]))
        ordered = [k for k in GROUP_ORDER if k in grouped]
        if OTHER_GROUP_KEY in grouped:
            ordered.append(OTHER_GROUP_KEY)

        suffix = f"_{label_suffix}" if label_suffix else ""
        return "".join(
            self._group_section(
                aggregated, key, grouped[key]["attacks"], f"{key}{suffix}",
            )
            for key in ordered
        )

    # ------------------------------------------------------------------
    # Main report assembly
    # ------------------------------------------------------------------

    def generate_latex_report(self, aggregated, model_name="DeepMark",
                             crop_before_attack=None, containers=None):
        """Generate complete LaTeX document.

        Args:
            aggregated: Aggregated results from aggregate_results()
            model_name: Name of the watermarking model
            crop_before_attack: If set, percentage cropped before attacks
        """
        num_attacks = len(aggregated["attacks"])
        is_single = num_attacks == 1
        across_phrase = (
            "a single attack type" if is_single else f"{num_attacks} attack types"
        )

        preamble = self._preamble(
            f"Detailed Benchmark Report: {model_name}",
            "DeepMark Benchmark System",
        )

        crop = crop_note(crop_before_attack)
        if crop:
            crop = " " + crop

        abstract = (
            f"\\begin{{abstract}}\n"
            f"This report presents a detailed evaluation of the "
            f"{model_name} watermarking model across {across_phrase}. "
            f"It covers watermark detection robustness, audio quality "
            f"impact analysis, and speech intelligibility measures, comparing "
            f"the effects of watermark embedding and adversarial "
            f"{'attack' if is_single else 'attacks'} on the audio signal."
            f"{crop}\n"
            f"\\end{{abstract}}\n"
        )

        return (
            f"{preamble}\n\n{abstract}\n"
            f"{self._embedding_cost_line(aggregated)}"
            f"{self._generate_body(aggregated)}"
            f"{container_section(containers or [])}"
            f"\\end{{document}}"
        )

    def generate_full_report(self, results, model_name="DeepMark",
                              is_zero_bit=False, crop_before_attack=None,
                              duration_partitions=None, containers=None):
        """
        Generate complete detailed report from raw benchmark results.

        Args:
            results: Raw benchmark results dict from Benchmark.run()
            model_name: Name of the watermarking model
            is_zero_bit: When True, also render accuracy as "n/N" detection
                counts.
            crop_before_attack: If set, percentage cropped before attacks
            duration_partitions: Optional list of (label, file_list) tuples
                for duration-based grouping. When provided, generates a
                section per duration group.

        Returns:
            Path to the generated ``.tex`` file.
        """
        if duration_partitions:
            latex_content = self._grouped_document(
                results, model_name, is_zero_bit, duration_partitions,
                containers, crop_before_attack=crop_before_attack,
            )
        else:
            aggregated = self.aggregate_results(results, is_zero_bit=is_zero_bit)
            latex_content = self.generate_latex_report(
                aggregated, model_name, crop_before_attack=crop_before_attack,
                containers=containers,
            )

        latex_path = os.path.join(self.report_dir, "detailed_report.tex")
        with open(latex_path, "w") as f:
            f.write(latex_content)

        logger.info(f"Detailed report saved to {latex_path}")
        compile_latex(self.report_dir, "detailed_report")
        return latex_path

    def _grouped_document(self, results, model_name, is_zero_bit,
                          duration_partitions, containers=None,
                          crop_before_attack=None):
        """Detailed report with one part per duration bin."""
        preamble = self._preamble(
            f"{model_name} --- Detailed Results (by Duration)",
            "DeepMark Benchmark",
        )

        parts = []
        for group_label_text, group_files in duration_partitions:
            group_results = {fp: results[fp] for fp in group_files if fp in results}
            if not group_results:
                continue
            aggregated = self.aggregate_results(
                group_results, is_zero_bit=is_zero_bit,
            )
            safe_label = duration_label_tex(group_label_text)
            slug = slugify(group_label_text)
            parts.append(
                part_heading(safe_label, f"{len(group_results)} files")
                + self._embedding_cost_line(aggregated)
                + self._generate_body(aggregated, label_suffix=slug)
            )

        # The crop applies to every bin, so it is stated once above them.
        crop = crop_note(crop_before_attack)
        if crop:
            crop = f"\\noindent {crop}\n\n"

        return (f"{preamble}\n\n" + crop + "\n\n".join(parts)
                + "\n\n" + container_section(containers or [])
                + "\n\n\\end{document}")


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _statistics(values, zero_bit=False, metric="accuracy"):
    """All eight statistics of ``values``, with the EMR count and rate and ``n``.

    Zero-bit accuracy adds the ``count`` of detections; multi-bit accuracy
    adds ``ber``, BER's own statistics.
    """
    if not values:
        return {}
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {}

    exact = int(np.sum(arr == 100.0))
    stats = {
        **compute_statistics(arr, metric=metric),
        "emr_count": exact,
        "emr_rate": float(exact / len(arr)),
        "n": len(arr),
    }
    if zero_bit:
        detected = int(sum(1 for v in arr if v))
        stats["count"] = f"{detected}/{len(arr)}"
    if metric == "accuracy" and not zero_bit:
        # From BER's own samples: its p10 is accuracy's p90, which no
        # accuracy statistic holds.
        stats["ber"] = compute_statistics(1.0 - arr / 100.0, metric="ber")
    return stats


def _ber_statistic(accuracy_stats, statistic):
    """One BER statistic, computed over the BER samples by ``_statistics``."""
    return (accuracy_stats.get("ber") or {}).get(statistic)


def _silent_note(metrics):
    """Footnote naming metrics that were enabled but produced nothing."""
    if not metrics:
        return ""
    names = ", ".join(metric_label(m) for m in dict.fromkeys(metrics))
    return (
        "\n{\\noindent\\footnotesize Enabled in the configuration but not "
        f"reported here, because no value was produced for any attack in this "
        f"section: {names}. This usually means the metric's service or "
        "optional package was unavailable.}\n\n"
    )
