"""The detailed benchmark report: per-attack-family metric breakdowns.

Organised by attack family, because a family is what decides which
metrics mean anything for it. ``audio_editing`` is further split into the
four subsections declared by ``ATTACK_SUBGROUPS`` -- filtering, temporal
edits, effects and compression -- each of which can enable different
metrics, and each of which is configurable under the same
``metrics.per_group`` key as a top-level group.

Every column comes from the ``MetricResolver`` the config file built:
each group's tables show the statistics that group configured, in the
order it listed them.

Every metric table carries a "No Attack (watermark only)" baseline row,
so a value is read against what embedding alone already cost.

Two figures per section carry what the tables cannot. This report
aggregates from the raw per-file results, so it is the only one that
still holds the distributions: a box plot shows the spread of per-file
accuracy behind each mean, and a bar chart puts the section's leading
quality metric against that same no-attack baseline.
"""

import logging
import os

import numpy as np

from deepmarkpy.utils import report_charts
from deepmarkpy.utils.attack_groups import (
    GROUP_ORDER,
    OTHER_GROUP_KEY,
    group_attacks,
    group_label,
    subgroups_of,
    ATTACK_SUBGROUPS,
)
from deepmarkpy.utils.latex_helpers import (
    build_longtable,
    compile_latex,
    container_section,
    display_attack_name,
    duration_label_tex,
    figure_block,
    format_emr_cell,
    format_metric_cell,
    make_preamble,
    part_heading,
    slugify,
    metric_label,
    stat_header,
)
from deepmarkpy.utils.metric_resolver import (
    INTELLIGIBILITY_METRICS,
    LOWER_IS_BETTER_METRICS,
    PER_FILE_EFFICIENCY_METRICS,
    MetricResolver,
    NISQA_METRICS,
    QUALITY_METRICS,
    worst_case_of,
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

# Metric families, each given its own table so no table carries thirteen
# columns.
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
            resolver: decides every table's metrics and statistic columns.
                Defaults to the built-in matrix declared by
                ``ATTACK_GROUPS``/``ATTACK_SUBGROUPS``.
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
                detection flag (0/1 per file) and the aggregate is
                reported as a count string ``"n/N"`` (files detected /
                files total) alongside the percentage.

        Returns:
            dict with ``watermarked_audio_quality`` (the no-attack
            baseline), ``is_zero_bit``, and ``attacks`` mapping each
            attack to its accuracy statistics and per-metric statistics.
        """
        metrics = self.resolver.all_signal_metrics()
        # Timings are not signal metrics, so all_signal_metrics leaves them
        # out by design. They are collected alongside, from the attack
        # entry itself rather than from its quality dict.
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
                # Kept alongside the statistics because a box plot needs the
                # distribution itself, and this report is the only one that
                # still holds the per-file values.
                "accuracy_values": [
                    float(v) for v in data["accuracy"] if v is not None
                ],
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

        Zero-bit models additionally report a per-attack count "n/N"
        (files where the watermark was detected over total files), since
        the per-file value collapses to 0/1 and the count makes the
        underlying detection ratio explicit.
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
        inline_ber = (
            show_ber
            and len(ber_statistics) == 1
        )
        show_emr = self.resolver.is_enabled(group_key, "emr")

        headers = ["Attack"] + [stat_header(s) for s in statistics]
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
            cells = [self._display_name(attack)]
            for statistic in statistics:
                value = accuracy.get(statistic)
                cells.append(
                    "--" if value is None
                    else format_metric_cell("accuracy", value)
                )
            if is_zero_bit:
                cells.append(accuracy.get("count", "N/A"))
            if inline_ber:
                value = _ber_statistic(accuracy, ber_statistics[0])
                cells.append(
                    "--" if value is None else format_metric_cell("ber", value)
                )
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
            rows.append("    " + " & ".join(cells) + " \\\\")

        tables = [build_longtable(
            "l" + "c" * (len(headers) - 1), " & ".join(headers), rows,
            caption, label,
        )]

        if show_ber and not inline_ber:
            # Its own subject, not the accuracy caption with a prefix, which
            # read "Bit error rate --- Watermark detection robustness --- X".
            tables.append(self._ber_table(
                aggregated, available, ber_statistics,
                f"Bit error rate --- {subject or caption}", f"{label}_ber",
            ))
        return "\n\n".join(tables)

    def _ber_table(self, aggregated, available, statistics, caption, label):
        """BER with two or more configured statistics."""
        headers = ["Attack"] + [stat_header(s) for s in statistics]
        rows = []
        for attack in available:
            accuracy = aggregated["attacks"][attack]["accuracy"]
            cells = [self._display_name(attack)]
            for statistic in statistics:
                value = _ber_statistic(accuracy, statistic)
                cells.append(
                    "--" if value is None else format_metric_cell("ber", value)
                )
            rows.append("    " + " & ".join(cells) + " \\\\")
        return build_longtable(
            "l" + "c" * len(statistics), " & ".join(headers), rows,
            caption, label,
        )

    def _metric_table(self, aggregated, attacks, metric, group_key,
                      caption, label):
        """One metric, one column per configured statistic, baseline first."""
        statistics = self.resolver.statistics_for(group_key, metric)
        if not statistics:
            return ""

        available = sorted(a for a in attacks if a in aggregated["attacks"])
        if not available:
            return ""

        headers = ["Condition"] + [stat_header(s) for s in statistics]
        rows = []

        baseline = (aggregated.get("watermarked_audio_quality") or {}).get(metric)
        if baseline and baseline.get("mean") is not None:
            cells = ["No Attack (watermark only)"] + [
                "N/A" if baseline.get(s) is None
                else format_metric_cell(metric, baseline[s])
                for s in statistics
            ]
            rows.append("    " + " & ".join(cells) + " \\\\")
            rows.append("    \\midrule")

        for attack in available:
            data = aggregated["attacks"][attack]["metrics"].get(metric) or {}
            cells = [self._display_name(attack)]
            for statistic in statistics:
                value = data.get(statistic)
                if value is None:
                    cells.append("N/A")
                    continue
                cells.append(format_metric_cell(metric, value))
            rows.append("    " + " & ".join(cells) + " \\\\")

        return build_longtable(
            "l" + "c" * len(statistics), " & ".join(headers), rows,
            caption, label,
        )

    def _compact_metric_table(self, aggregated, attacks, metrics, group_key,
                              caption, label):
        """Metrics reduced to one statistic each, one column per metric."""
        available = sorted(a for a in attacks if a in aggregated["attacks"])
        if not available or not metrics:
            return ""

        headers = ["Condition"]
        for metric in metrics:
            statistic = self.resolver.statistics_for(group_key, metric)[0]
            header = metric_label(metric)
            if statistic != "mean":
                header += f" [{stat_header(statistic)}]"
            headers.append(header)

        rows = []

        baseline = aggregated.get("watermarked_audio_quality") or {}
        baseline_cells = []
        for metric in metrics:
            statistic = self.resolver.statistics_for(group_key, metric)[0]
            value = (baseline.get(metric) or {}).get(statistic)
            baseline_cells.append(
                "N/A" if value is None else format_metric_cell(metric, value)
            )
        if any(cell != "N/A" for cell in baseline_cells):
            rows.append(
                "    No Attack (watermark only) & "
                + " & ".join(baseline_cells) + " \\\\"
            )
            rows.append("    \\midrule")

        for attack in available:
            attack_metrics = aggregated["attacks"][attack]["metrics"]
            cells = [self._display_name(attack)]
            for metric in metrics:
                statistic = self.resolver.statistics_for(group_key, metric)[0]
                value = (attack_metrics.get(metric) or {}).get(statistic)
                if value is None:
                    cells.append("N/A")
                    continue
                cells.append(format_metric_cell(metric, value))
            rows.append("    " + " & ".join(cells) + " \\\\")

        return build_longtable(
            "l" + "c" * len(metrics), " & ".join(headers), rows, caption, label,
        )

    def _embedding_cost_line(self, aggregated):
        """Embedding time, stated once rather than per attack.

        Measured once per file and independent of which attack follows,
        so it belongs in a sentence about the run, not in a column of a
        table whose rows are attacks.
        """
        metric = "embed_latency"
        if not self.resolver.is_enabled(None, metric):
            return ""

        # The same per-file cost is recorded on every attack entry, so any
        # of them carries it.
        for data in aggregated["attacks"].values():
            timings = (data.get("timings") or {}).get(metric) or {}
            parts = []
            for statistic in self.resolver.statistics_for(None, metric):
                value = timings.get(statistic)
                if value is not None:
                    parts.append(
                        f"{float(value):.4f}\\,s ({stat_header(statistic).lower()})"
                    )
            if parts:
                return (
                    f"\\noindent\\textbf{{Embedding cost per file:}} "
                    f"{', '.join(parts)}\n"
                    "\\\\{\\footnotesize Measured once per file, before any "
                    "attack, so it does not vary by attack. Like every timing "
                    "it depends on this machine and does not reproduce across "
                    "runs.}\n\n"
                )
        return ""

    def _efficiency_table(self, aggregated, attacks, group_key, label_text,
                          label_key):
        """Processing time for one section, in its own tables.

        Below the quality tables and never a column beside them: those
        describe the watermarking method and reproduce from a seed, this
        describes the machine that ran it and does not. Split the same way
        the quality tables are, so several statistics do not become one
        very wide table.
        """
        available = sorted(a for a in attacks if a in aggregated["attacks"])
        if not available:
            return ""

        metrics = [
            m for m in self.resolver.metrics_for_group(None,
                                                       bucket="efficiency")
            if any((aggregated["attacks"][a].get("timings") or {}).get(m)
                   for a in available)
            # Embedding does not depend on the attack, so it is not a
            # column in a table whose rows are attacks.
            if m not in PER_FILE_EFFICIENCY_METRICS
        ]
        if not metrics:
            return ""

        note = (
            " These depend on the machine and on whether the plugin ran "
            "natively or over HTTP, so they do not reproduce across runs the "
            "way the measurements above do."
        )

        tables = []
        compact = []
        for metric in metrics:
            statistics = self.resolver.statistics_for(group_key, metric)
            if len(statistics) > 1:
                tables.append(self._timing_table(
                    aggregated, available, [(metric, s) for s in statistics],
                    [stat_header(s) for s in statistics],
                    f"{metric_label(metric)} --- {label_text}.{note}",
                    f"tab:efficiency_{label_key}_{metric}",
                ))
            elif statistics:
                compact.append((metric, statistics[0]))

        if compact:
            tables.append(self._timing_table(
                aggregated, available, compact,
                [metric_label(m) for m, _ in compact],
                f"Processing time --- {label_text}.{note}",
                f"tab:efficiency_{label_key}",
            ))

        return "\n\n".join(t for t in tables if t)

    def _timing_table(self, aggregated, available, columns, headers,
                      caption, label):
        """One timing table: a row per attack, a column per (metric, statistic)."""
        rows = []
        for attack in available:
            timings = aggregated["attacks"][attack].get("timings") or {}
            cells = [self._display_name(attack)]
            for metric, statistic in columns:
                value = (timings.get(metric) or {}).get(statistic)
                cells.append("--" if value is None else f"{float(value):.4f}")
            rows.append("    " + " & ".join(cells) + " \\\\")

        return build_longtable(
            "l" + "c" * len(columns), " & ".join(["Attack"] + headers),
            rows, caption, label,
        )

    def _metric_tables(self, aggregated, attacks, group_key, label_text,
                       label_key, figure_for=None):
        """Every metric table for one group or subgroup.

        ``figure_for`` is ``(metric, latex)``: the block is emitted right
        after that metric's table, because a figure of one metric read
        after a table of another is a figure the reader has to re-anchor.
        A metric shown only as a column of the compact table gets its
        figure after that table instead.
        """
        available = sorted(a for a in attacks if a in aggregated["attacks"])
        if not available:
            return "", []

        figure_metric, figure = figure_for or (None, "")
        figure_placed = False

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

            multi = [
                m for m in with_data
                if len(self.resolver.statistics_for(group_key, m)) > 1
            ]
            single = [m for m in with_data if m not in multi]

            for metric in multi:
                table = self._metric_table(
                    aggregated, available, metric, group_key,
                    f"{metric_label(metric)} --- {label_text}.",
                    f"tab:{section_key}_{label_key}_{metric}",
                )
                if table:
                    if metric == figure_metric:
                        table += "\n\n" + figure
                        figure_placed = True
                    blocks.append(table)

            if single and figure and not figure_placed \
                    and figure_metric in single:
                figure_placed = True
                blocks.append(self._compact_metric_table(
                    aggregated, available, single, group_key,
                    f"{section_title} --- {label_text}.",
                    f"tab:{section_key}_{label_key}",
                ) + "\n\n" + figure)
            elif single:
                blocks.append(self._compact_metric_table(
                    aggregated, available, single, group_key,
                    f"{section_title} --- {label_text}.",
                    f"tab:{section_key}_{label_key}",
                ))

        body = "\n\n".join(blocks)
        if figure and not figure_placed:
            body += "\n\n" + figure
        return body, silent

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------


    def _distribution_figure(self, aggregated, attacks, label_text, label_key):
        """How each attack's files split by outcome.

        This report aggregates from the raw per-file results, so it is the
        only one that can say what is behind a mean: every file degraded a
        little, or half of them destroyed. Counting files into outcome
        bands answers that for a graded score and keeps working for a
        zero-bit detector, whose per-file score is only ever 0 or 100.
        """
        is_zero_bit = bool(aggregated.get("is_zero_bit"))
        distributions = {
            self._display_name(attack): aggregated["attacks"][attack].get(
                "accuracy_values", []
            )
            for attack in attacks if attack in aggregated["attacks"]
        }
        filename = f"accuracy_spread_{label_key}.png"
        drawn = report_charts.per_file_outcome_bars(
            distributions, os.path.join(self.report_dir, filename),
            title=f"Per-file outcome --- {label_text}",
            chance_floor=0.0 if is_zero_bit else 50.0,
            is_zero_bit=is_zero_bit,
        )
        if not drawn:
            return ""
        reading = (
            "each file either yielded a detection or did not"
            if is_zero_bit else
            "a file is bit-exact, still above the random-guess floor, or at "
            "or below it"
        )
        return figure_block(
            filename,
            f"Share of files by detection outcome for {label_text.lower()}, "
            f"worst first --- {reading}. Numbers inside the bars are file "
            f"counts. Two attacks with the same mean can split very "
            f"differently here.",
            f"fig:spread_{label_key}",
        )

    # The remaining figures are the benchmark report's, drawn from this
    # report's own aggregate so a section shows the same three views of its
    # attacks: how they rank, how a ladder degrades, and what the watermark
    # cost in audio quality.

    def _accuracy_of(self, aggregated, attack, group_key):
        """An attack's headline accuracy, in the statistic its group configured."""
        statistics = self.resolver.statistics_for(group_key, "accuracy")
        statistic = statistics[0] if statistics else "mean"
        value = (aggregated["attacks"][attack].get("accuracy") or {}).get(statistic)
        return None if value is None else float(value)

    def _chance_floor(self, aggregated):
        return 0.0 if aggregated.get("is_zero_bit") else 50.0

    def _ranking_figure(self, aggregated, attacks, group_key, label_text,
                        label_key):
        """The accuracy table above, ranked worst-first and coloured by tier."""
        available = [a for a in attacks if a in aggregated["attacks"]]
        if len(available) < 3:
            # With one or two bars the table above already reads as a
            # ranking, and the figure only repeats it.
            return ""

        values = {}
        for attack in available:
            score = self._accuracy_of(aggregated, attack, group_key)
            if score is not None:
                values[self._display_name(attack)] = score
        if len(values) < 3:
            return ""

        statistics = self.resolver.statistics_for(group_key, "accuracy")
        statistic = statistics[0] if statistics else "mean"
        filename = f"ranking_{label_key}.png"
        drawn = report_charts.accuracy_ranking(
            values, os.path.join(self.report_dir, filename),
            statistic_label=stat_header(statistic),
            chance_floor=self._chance_floor(aggregated),
            title=f"{label_text} ranked by accuracy "
                  f"({stat_header(statistic)})",
        )
        if not drawn:
            return ""
        return figure_block(
            filename,
            f"Attacks in {label_text.lower()} ranked by detection accuracy "
            f"({stat_header(statistic).lower()}), worst first. Bar colour is "
            f"the robustness tier; the dashed line is what a failed detection "
            f"already scores.",
            f"fig:ranking_{label_key}",
        )

    def _strength_figure(self, aggregated, attacks, group_key, label_key):
        """Accuracy across the versions of this section's ladder attacks."""
        available = [a for a in attacks if a in aggregated["attacks"]]
        series = report_charts.version_series(
            available,
            lambda name: self._accuracy_of(aggregated, name, group_key),
        )
        if not series:
            return ""

        statistics = self.resolver.statistics_for(group_key, "accuracy")
        statistic = statistics[0] if statistics else "mean"
        filename = f"strength_{label_key}.png"
        drawn = report_charts.attack_strength_curves(
            series, os.path.join(self.report_dir, filename),
            statistic_label=stat_header(statistic),
            chance_floor=self._chance_floor(aggregated),
        )
        if not drawn:
            return ""
        return figure_block(
            filename,
            "Detection accuracy across the configured versions of the same "
            "attack, in the order the configuration declares them. The point "
            "where a curve crosses the chance line is the strength at which "
            "the watermark stops surviving.",
            f"fig:strength_{label_key}",
        )

    def _scatter_figure(self, aggregated, attacks, group_key, label_key):
        """This section's accuracy against the audio quality it cost.

        Which quality metric this is comes from the configuration, not from
        a constant: the first of a perceptual-first preference order that
        the group enabled and that produced values.

        Returns ``(metric, latex)`` so the caller can place the figure
        under the table for *that* metric.
        """
        preference = ("visqol", "pesq", "nisqa_mos", "stoi", "mcd",
                      "si_sdr", "psnr")
        enabled = self.resolver.signal_metrics_for_group(group_key)
        available = [a for a in attacks if a in aggregated["attacks"]]

        for metric in preference:
            if metric not in enabled:
                continue
            statistics = self.resolver.statistics_for(group_key, metric)
            if not statistics:
                continue
            statistic = statistics[0]

            points = []
            for attack in available:
                quality = (
                    aggregated["attacks"][attack]["metrics"].get(metric) or {}
                ).get(statistic)
                points.append((
                    self._display_name(attack), quality,
                    self._accuracy_of(aggregated, attack, group_key),
                ))
            if not any(q is not None for _, q, _ in points):
                continue


            higher_is_better = metric not in LOWER_IS_BETTER_METRICS
            filename = f"scatter_{metric}_{label_key}.png"
            drawn = report_charts.robustness_quality_scatter(
                points, os.path.join(self.report_dir, filename),
                metric_label=report_charts.direction_hint(
                    metric_label(metric), higher_is_better,
                ),
                higher_is_better=higher_is_better,
                chance_floor=self._chance_floor(aggregated),
            )
            if not drawn:
                return None, ""
            return metric, figure_block(
                filename,
                f"Detection accuracy against {metric_label(metric)} of the "
                f"attacked audio, for the attacks tabled above. An attack in "
                f"the shaded corner removed the watermark while leaving the "
                f"recording usable, which is the case that matters; one in "
                f"the opposite corner paid for it with the audio.",
                f"fig:scatter_{metric}_{label_key}",
            )
        return None, ""

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
        # The three views of the accuracy table above, then the per-file
        # split behind it.
        section += self._ranking_figure(
            aggregated, available, group_key, label_text, label_key,
        )
        section += self._strength_figure(
            aggregated, available, group_key, label_key,
        )
        section += self._distribution_figure(
            aggregated, available, label_text, label_key,
        )

        subgroups = subgroups_of(group_key)
        if subgroups:
            section += self._subgroup_sections(
                aggregated, subgroups, available, label_key,
            )
            # One table for the family, after its subsections: timing is a
            # property of the attack, not of which metrics a subsection
            # happens to report.
            section += self._efficiency_table(
                aggregated, available, group_key, label_text, label_key,
            )
            return section

        body, silent = self._metric_tables(
            aggregated, available, group_key, label_text, label_key,
            figure_for=self._scatter_figure(
                aggregated, available, group_key, label_key,
            ),
        )
        section += body
        section += _silent_note(silent)
        section += self._efficiency_table(
            aggregated, available, group_key, label_text, label_key,
        )
        return section + "\n\n"

    def _subgroup_sections(self, aggregated, subgroups, group_attack_list,
                           label_key):
        """Subsections for a group that declares subgroups."""
        assigned = set()
        sections = ""

        for subgroup in subgroups:
            definition = ATTACK_SUBGROUPS[subgroup]
            members = [
                attack for attack in group_attack_list
                if _matches_any(attack, definition["attacks"])
            ]
            if not members:
                continue
            assigned.update(members)

            sections += "\\needspace{5\\baselineskip}\n"
            sections += f"\\subsection{{{definition['label']}}}\n\n"
            sections += f"{definition['description']}\n\n"

            body, silent = self._metric_tables(
                aggregated, members, subgroup, definition["label"],
                f"{label_key}_{subgroup}",
                figure_for=self._scatter_figure(
                    aggregated, members, subgroup, f"{label_key}_{subgroup}",
                ),
            )
            if body:
                sections += body + "\n\n"
            elif not self.resolver.signal_metrics_for_group(subgroup):
                sections += (
                    "\\noindent No quality or intelligibility metric is "
                    "enabled for this subsection in the configuration.\n\n"
                )
            sections += _silent_note(silent)

        # An attack in the group but in none of its subgroups would
        # otherwise vanish from the report entirely.
        unassigned = [a for a in group_attack_list if a not in assigned]
        if unassigned:
            sections += "\\needspace{5\\baselineskip}\n"
            sections += "\\subsection{Other Editing Attacks}\n\n"
            body, silent = self._metric_tables(
                aggregated, unassigned, "audio_editing",
                "Other Editing Attacks", f"{label_key}_unassigned",
            )
            sections += body + "\n\n" + _silent_note(silent)

        return sections

    def _generate_body(self, aggregated, label_suffix=""):
        """Per-group sections, in the taxonomy's order.

        ``label_suffix`` distinguishes one duration part from another: the
        parts repeat the same groups, so without it every part's figures
        would be written to the same filenames and only the last would
        survive.
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

        crop_note = self._crop_note(crop_before_attack)
        if crop_note:
            crop_note = " " + crop_note

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

        return (
            f"{preamble}\n\n{abstract}\n"
            f"{self._embedding_cost_line(aggregated)}"
            f"{self._generate_body(aggregated)}"
            f"{container_section(containers or [])}"
            f"\\end{{document}}"
        )

    @staticmethod
    def _crop_note(crop_before_attack):
        """The caveat that every attack ran on cropped audio.

        Shared by the flat and the duration-grouped document, because a
        reader who does not see it takes the numbers for the whole signal
        whichever shape the report has.
        """
        if crop_before_attack is None:
            return ""
        return (
            f"\\textcolor{{red}}{{A crop of {crop_before_attack:.1f}\\% was applied to the "
            f"beginning of the watermarked audio prior to each attack. "
            f"The original (reference) audio was cropped identically, so "
            f"quality metrics compare cropped original vs.\\ cropped attacked "
            f"audio, and BER is measured by detecting the watermark from the "
            f"cropped attacked signal.}}"
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
                counts (a zero-bit detector returns 0/1 per file, so the
                percentage column alone only ever shows 0 or 100).
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
        preamble = make_preamble(
            f"{model_name} --- Detailed Results (by Duration)",
            "DeepMark Benchmark",
            self._has_deepmark_cls,
            extra_packages=("xcolor",),
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

        # The crop applies to every bin, so it is stated once above
        # them rather than repeated in each part.
        crop_note = self._crop_note(crop_before_attack)
        if crop_note:
            crop_note = f"\\noindent {crop_note}\n\n"

        return (f"{preamble}\n\n" + crop_note + "\n\n".join(parts)
                + "\n\n" + container_section(containers or [])
                + "\n\n\\end{document}")


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _statistics(values, zero_bit=False, metric="accuracy"):
    """All eight statistics for a list of values, plus EMR and BER inputs.

    Computed in full because this aggregate is internal and never written
    to disk; which of them a table shows is the resolver's decision. That
    keeps aggregation independent of which group an attack lands in.
    """
    if not values:
        return {}
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {}

    exact = int(np.sum(arr == 100.0))
    stats = {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "worst_case": worst_case_of(arr, metric),
        "emr_count": exact,
        "emr_rate": float(exact / len(arr)),
        "n": len(arr),
    }
    if zero_bit:
        detected = int(sum(1 for v in arr if v))
        stats["count"] = f"{detected}/{len(arr)}"
    if metric == "accuracy" and not zero_bit:
        # From the BER samples themselves, exactly as ``compute_metrics``
        # does, and not by inverting accuracy's summary: BER's 10th
        # percentile is accuracy's 90th, which is not among the eight
        # statistics at all, so a derived one printed a number that
        # disagreed with the same run's basic report.
        ber = 1.0 - arr / 100.0
        stats["ber"] = {
            "mean": float(np.mean(ber)),
            "std": float(np.std(ber, ddof=1)) if len(ber) > 1 else 0.0,
            "median": float(np.median(ber)),
            "p5": float(np.percentile(ber, 5)),
            "p10": float(np.percentile(ber, 10)),
            "p95": float(np.percentile(ber, 95)),
            "p99": float(np.percentile(ber, 99)),
            "worst_case": worst_case_of(ber, "ber"),
        }
    return stats


def _ber_statistic(accuracy_stats, statistic):
    """One BER statistic, computed over the BER samples by ``_statistics``."""
    return (accuracy_stats.get("ber") or {}).get(statistic)


def _matches_any(attack_name, base_names):
    """Whether ``attack_name`` is one of ``base_names`` or an expansion of one.

    Covers ``Codec2VocoderAttack_700`` and ``EchoAttack (mild)``.
    """
    for base in base_names:
        if attack_name == base or attack_name.startswith(f"{base}_") \
                or attack_name.startswith(f"{base} ("):
            return True
    return False


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
