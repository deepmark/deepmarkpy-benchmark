"""The basic benchmark report: accuracy per attack, plus a chart.

Every table's columns come from the ``MetricResolver`` the config file
built. Nothing here decides which metric or which statistic to show, so
adding ``std`` to a metric in the config makes a ``Std`` column appear,
and removing a metric removes its table. There is no fallback column set
to override the config.

Attacks are grouped into sections by attack family, because two families
can enable different metrics and a single flat table would be mostly
holes.

Within a section:

* one accuracy table -- a column per configured accuracy statistic, plus
  BER and EMR when enabled and reduced to a single number;
* one table per metric configured with two or more statistics;
* one compact table collecting every remaining metric, one column each.

That last rule keeps the common case (one statistic per metric) to a
single readable table without ever dropping something the config asked
for.
"""

import json
import os
import logging
from typing import Dict, Mapping, Union

from deepmarkpy.utils import report_charts
from deepmarkpy.utils.attack_groups import (
    GROUP_ORDER,
    OTHER_GROUP_KEY,
    group_attacks,
    group_label,
)
from deepmarkpy.utils.latex_helpers import (
    build_longtable,
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
    EFFICIENCY_METRICS,
    LOWER_IS_BETTER_METRICS,
    PER_FILE_EFFICIENCY_METRICS,
    MetricResolver,
)

logger = logging.getLogger(__name__)

StatsValue = Union[float, Mapping[str, float]]


class BenchmarkReportGenerator:
    """Generate LaTeX reports for benchmark results with visualizations."""

    def __init__(self, report_dir: str = "report", resolver: MetricResolver = None,
                 is_zero_bit: bool = False):
        """
        Args:
            report_dir: where the ``.tex``, ``.pdf`` and chart are written.
            resolver: decides every table's metrics and statistic columns.
                Defaults to the built-in matrix declared by
                ``ATTACK_GROUPS`` -- the same one ``--init`` ships -- so
                the generator stays usable as a library.
            is_zero_bit: whether the model reports detection rather than
                bit agreement. It only moves the reference line the charts
                draw: a failed multi-bit detection lands at chance (50\\%),
                a failed zero-bit detection lands at 0.
        """
        self.report_dir = report_dir
        self.resolver = resolver or MetricResolver.from_attack_groups()
        self.is_zero_bit = bool(is_zero_bit)
        self.ensure_report_dir()
        self._has_deepmark_cls = os.path.exists(os.path.join(self.report_dir, "deepmark.cls"))

    def ensure_report_dir(self):
        """Ensure the report directory exists."""
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)

    def _preamble(self, title, author):
        """Generate LaTeX preamble with deepmark class fallback."""
        return make_preamble(title, author, self._has_deepmark_cls,
                             extra_packages=("xcolor",))

    # ------------------------------------------------------------------
    # Stats access
    # ------------------------------------------------------------------

    @property
    def _accuracy_statistic(self) -> str:
        """The accuracy statistic the chart and the headline mean use.

        The first one the config lists for accuracy, so a config that
        drops ``mean`` still produces a chart rather than an empty one.
        """
        return self._accuracy_statistic_for(None)

    def _accuracy_statistic_for(self, attack_name=None) -> str:
        """The accuracy statistic to read for one attack.

        A per-group override means two attacks in the same report can
        carry different accuracy statistics, and ``compute_mean_accuracy``
        writes each attack under the statistic *its own group* asked for.
        Reading the report-wide default off an attack whose group dropped
        it finds nothing, so the group decides the key here exactly as it
        does in ``_metric_value``.
        """
        group_key = (self.resolver.group_for_attack(attack_name)
                     if attack_name else None)
        configured = self.resolver.statistics_for(group_key, "accuracy")
        return configured[0] if configured else "mean"

    def _accuracy_label_for(self, stats) -> str:
        """Axis label for a chart drawn over ``stats``.

        Names the statistic only when every attack in the figure shares
        one. Groups configured differently have no single honest label,
        and printing one group's would mislabel the rest.
        """
        statistics = {
            self._accuracy_statistic_for(name)
            for name in (stats or {})
        }
        if len(statistics) == 1:
            return stat_header(statistics.pop())
        if not statistics:
            return stat_header(self._accuracy_statistic)
        return "per-group statistic"

    def _accuracy_of(self, value: StatsValue, attack_name=None) -> float:
        """Headline accuracy for an attack, in its group's statistic.

        Stats entries are either a bare float (a caller passing accuracy
        directly) or the per-attack dict ``compute_mean_accuracy`` writes.
        The fallbacks matter: an entry that carries accuracy under some
        other statistic must not be read as zero, because a zero here is
        indistinguishable from a watermark that did not survive.
        """
        if not isinstance(value, Mapping):
            return float(value or 0.0)

        key = f"accuracy_{self._accuracy_statistic_for(attack_name)}"
        for candidate in (key, "accuracy_mean"):
            found = value.get(candidate)
            if found is not None:
                return float(found)
        for name, found in value.items():
            if (name.startswith("accuracy_") and name != "accuracy_n"
                    and found is not None):
                return float(found)
        return 0.0

    @staticmethod
    def _stat_of(value: StatsValue, key: str, default=None):
        """Return an arbitrary key from a per-attack stats dict, or ``default``."""
        if isinstance(value, Mapping):
            return value.get(key, default)
        return default

    # ------------------------------------------------------------------
    # Chart
    # ------------------------------------------------------------------

    @property
    def _chance_floor(self) -> float:
        """Accuracy a failed detection lands on, for the charts' reference line."""
        return 0.0 if self.is_zero_bit else 50.0

    def create_gradient_bar_chart(self, stats: Dict[str, StatsValue],
                                  output_path: str) -> bool:
        """Attacks ranked worst-first by the configured accuracy statistic.

        Ranked, not alphabetical: the question a reader brings to this
        figure is which attack does the most damage, and an alphabetical
        axis answers a different one.
        """
        values = {
            display_attack_name(name): self._accuracy_of(value, name)
            for name, value in stats.items()
        }
        return report_charts.accuracy_ranking(
            values, output_path,
            statistic_label=self._accuracy_label_for(stats),
            chance_floor=self._chance_floor,
        )

    def _quality_metric_for_charts(self, stats, group_key=None):
        """The quality metric to plot accuracy against, or ``None``.

        Preference runs from the most perceptual measure to the least, but
        only metrics this group enabled are considered, and only one that
        actually produced values is chosen -- so the figure always has a
        table of its own in the same section to sit under.
        """
        preference = ("visqol", "pesq", "nisqa_mos", "stoi", "mcd",
                      "si_sdr", "psnr")
        # The section's own resolution, ``None`` included -- never
        # ``all_signal_metrics()``, which is the union across every group
        # and can name a metric this section's tables leave out.
        enabled = self.resolver.signal_metrics_for_group(group_key)
        for metric in preference:
            if metric not in enabled:
                continue
            if any(self._metric_value(value, name, metric) is not None
                   for name, value in stats.items()):
                return metric
        return None

    def _metric_value(self, entry, attack_name, metric):
        """One metric value for an attack, in that group's first statistic."""
        group_key = self.resolver.group_for_attack(attack_name)
        statistics = self.resolver.statistics_for(group_key, metric)
        if not statistics:
            return None
        return self._stat_of(entry, f"{metric}_{statistics[0]}")


    # Each figure sits directly under the table whose numbers it draws --
    # the strength curves under the accuracy table, the scatter under the
    # quality table -- rather than collected at the end of the document,
    # where the reader has to carry a section's numbers to them.

    def _strength_figure(self, stats, name_key):
        """Accuracy across the versions of this section's ladder attacks."""
        series = report_charts.version_series(
            list(stats), lambda name: self._accuracy_of(stats[name], name),
        )
        if not series:
            return ""

        filename = f"attack_strength_{name_key}.png"
        drawn = report_charts.attack_strength_curves(
            series, os.path.join(self.report_dir, filename),
            statistic_label=self._accuracy_label_for(stats),
            chance_floor=self._chance_floor,
        )
        if not drawn:
            return ""
        return "\n\n" + figure_block(
            filename,
            "Detection accuracy across the configured versions of the same "
            "attack, in the order the configuration declares them. The point "
            "where a curve crosses the chance line is the strength at which "
            "the watermark stops surviving.",
            f"fig:attack_strength_{name_key}",
        )

    def _quality_scatter_figure(self, stats, name_key, group_key=None):
        """This section's accuracy against the audio quality it cost.

        Returns ``(metric, latex)`` so the caller can place the figure
        directly under the table for *that* metric, rather than after
        whichever quality table happens to come last.
        """
        metric = self._quality_metric_for_charts(stats, group_key)
        if not metric:
            return None, ""

        higher_is_better = metric not in LOWER_IS_BETTER_METRICS
        points = [
            (display_attack_name(name),
             self._metric_value(value, name, metric),
             self._accuracy_of(value, name))
            for name, value in stats.items()
        ]
        filename = f"robustness_quality_{name_key}.png"
        drawn = report_charts.robustness_quality_scatter(
            points, os.path.join(self.report_dir, filename),
            metric_label=report_charts.direction_hint(
                metric_label(metric), higher_is_better,
            ),
            higher_is_better=higher_is_better,
            chance_floor=self._chance_floor,
        )
        if not drawn:
            return None, ""
        return metric, "\n\n" + figure_block(
            filename,
            f"Detection accuracy against {metric_label(metric)} of the "
            f"attacked audio, for the attacks tabled above. An attack in the "
            f"shaded corner removed the watermark while leaving the recording "
            f"usable, which is the case that matters; one in the opposite "
            f"corner paid for it with the audio.",
            f"fig:robustness_quality_{name_key}",
        )

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def generate_latex_table(self, stats: Dict[str, StatsValue],
                             group_key: str = None,
                             label_suffix: str = "") -> str:
        """Build every table for one attack group.

        Args:
            stats: ``{attack_name: per-attack stats dict}`` for the attacks
                in this group.
            group_key: which group's configuration to apply. ``None`` uses
                ``metrics.defaults`` alone, which is what a caller with an
                ungrouped stats dict gets.
            label_suffix: appended to every ``\\label`` so the same group
                can appear once per duration bin without clashing.
        """
        sorted_attacks = sorted(stats.items())
        if not sorted_attacks:
            return ""

        caption_word = (
            "the attack type" if len(sorted_attacks) == 1
            else "different attack types"
        )
        suffix = f"_{label_suffix}" if label_suffix else ""
        group_name = group_key or "all"

        name_key = f"{group_name}{suffix}"

        # The strength curves read the accuracy column, so they follow the
        # accuracy table; the scatter reads a quality column, so it follows
        # the quality tables at the end. Neither is collected at the foot of
        # the document, where the reader would have to carry the section's
        # numbers to it.
        tables = [
            self._accuracy_table(
                sorted_attacks, group_key, caption_word,
                f"tab:benchmark_accuracy_{group_name}{suffix}",
            ) + self._strength_figure(stats, name_key)
        ]

        # The scatter plots one quality metric, so it belongs under that
        # metric's own table -- not after the last quality table in the
        # section, which is a different metric entirely.
        scatter_metric, scatter = self._quality_scatter_figure(
            stats, name_key, group_key,
        )
        scatter_placed = False

        enabled = self.resolver.metrics_for_group(group_key)
        compact = []
        silent = []
        for metric in enabled:
            if metric in ("accuracy", "ber", "emr"):
                continue
            # Timings have their own tables below, built by
            # _efficiency_table; without this they would be built twice.
            if metric in EFFICIENCY_METRICS:
                continue
            if not self._has_data(sorted_attacks, metric):
                # Configured but nothing came back -- usually a metric whose
                # service or optional package is missing. A table of N/A
                # rows says the same thing in far more space, and dropping
                # it without a word would hide that the config asked for it.
                silent.append(metric)
                continue
            statistics = self.resolver.statistics_for(group_key, metric)
            if len(statistics) > 1:
                table = self._metric_table(
                    sorted_attacks, metric, statistics, caption_word,
                    f"tab:benchmark_{metric}_{group_name}{suffix}",
                )
                if metric == scatter_metric:
                    table += scatter
                    scatter_placed = True
                tables.append(table)
            elif statistics:
                compact.append((metric, statistics[0]))

        # BER is a metric like any other, but with a single statistic it
        # reads better beside the accuracy it is derived from, so it is
        # already in the accuracy table; with several it gets its own.
        if "ber" in enabled and not self.is_zero_bit:
            ber_statistics = self.resolver.statistics_for(group_key, "ber")
            if len(ber_statistics) > 1:
                tables.append(self._metric_table(
                    sorted_attacks, "ber", ber_statistics,
                    caption_word, f"tab:benchmark_ber_{group_name}{suffix}",
                ))

        if compact:
            table = self._compact_metric_table(
                sorted_attacks, compact, caption_word,
                f"tab:benchmark_metrics_{group_name}{suffix}",
            )
            # A single-statistic metric has no table of its own; the compact
            # one carries its column, so the figure follows that.
            if scatter and not scatter_placed:
                table += scatter
                scatter_placed = True
            tables.append(table)

        efficiency = self._efficiency_table(
            sorted_attacks, group_key, caption_word,
            f"tab:benchmark_efficiency_{group_name}{suffix}",
        )
        if efficiency:
            tables.append(efficiency)

        body = "\n\n".join(t for t in tables if t)
        if scatter and not scatter_placed:
            body += scatter
        return body + self._footnotes(sorted_attacks, silent)

    def _efficiency_table(self, sorted_attacks, group_key, caption_word, label):
        """Timings, in their own tables below the quality ones.

        Never a column beside PESQ or accuracy: those reproduce from a
        seed and these do not. Split the same way the quality tables are
        -- a metric with two or more statistics gets its own table, and
        the single-statistic ones share one -- because three metrics times
        four statistics is thirteen columns on one page.
        """
        metrics = [
            m for m in self.resolver.metrics_for_group(group_key,
                                                       bucket="efficiency")
            if self._has_data(sorted_attacks, m)
            # Embedding does not depend on the attack, so it is not a
            # column in a table whose rows are attacks.
            if m not in PER_FILE_EFFICIENCY_METRICS
        ]
        if not metrics:
            return ""

        note = (
            f" These depend on the machine and on whether the plugin ran "
            f"natively or over HTTP, so they do not reproduce across runs "
            f"the way the measurements above do."
        )

        tables = []
        compact = []
        for metric in metrics:
            statistics = self.resolver.statistics_for(group_key, metric)
            if len(statistics) > 1:
                tables.append(self._timing_table(
                    sorted_attacks, [(metric, s) for s in statistics],
                    [stat_header(s) for s in statistics],
                    f"{metric_label(metric)} for {caption_word}.{note}",
                    f"{label}_{metric}",
                ))
            elif statistics:
                compact.append((metric, statistics[0]))

        if compact:
            tables.append(self._timing_table(
                sorted_attacks, compact,
                [metric_label(m) for m, _ in compact],
                f"Processing time for {caption_word}.{note}", label,
            ))

        return "\n\n".join(t for t in tables if t)

    def _timing_table(self, sorted_attacks, columns, headers, caption, label):
        """One timing table: a row per attack, a column per (metric, statistic)."""
        rows = []
        for attack_name, value in sorted_attacks:
            cells = [display_attack_name(attack_name)]
            for metric, statistic in columns:
                raw = self._stat_of(value, f"{metric}_{statistic}")
                cells.append("--" if raw is None else f"{float(raw):.4f}")
            rows.append("    " + " & ".join(cells) + " \\\\")

        return build_longtable(
            "l" + "c" * len(columns), " & ".join(["Attack Type"] + headers),
            rows, caption, label,
        )

    @staticmethod
    def _has_data(sorted_attacks, metric):
        """Whether any attack produced any value for ``metric``.

        Looks for any ``<metric>_<statistic>`` key rather than the
        configured ones, so a stats file written under a different config
        is still recognised as carrying data.
        """
        prefix = f"{metric}_"
        return any(
            key.startswith(prefix) and key != f"{metric}_n" and value is not None
            for _, entry in sorted_attacks
            if isinstance(entry, Mapping)
            for key, value in entry.items()
        )

    def _accuracy_table(self, sorted_attacks, group_key, caption_word, label):
        """Accuracy statistics, plus single-valued BER and EMR."""
        statistics = self.resolver.statistics_for(group_key, "accuracy")
        headers = ["Attack Type"] + [stat_header(s) for s in statistics]

        ber_statistics = self.resolver.statistics_for(group_key, "ber")
        inline_ber = (
            not self.is_zero_bit
            and self.resolver.is_enabled(group_key, "ber")
            and len(ber_statistics) == 1
        )
        if inline_ber:
            headers.append(metric_label("ber"))

        show_emr = self.resolver.is_enabled(group_key, "emr")
        if show_emr:
            headers.append(metric_label("emr"))

        rows = []
        for attack_name, value in sorted_attacks:
            cells = [display_attack_name(attack_name)]
            n_files = self._stat_of(value, "accuracy_n")
            failures = self._stat_of(value, "detection_failures", 0) or 0

            for statistic in statistics:
                raw = self._stat_of(value, f"accuracy_{statistic}")
                if raw is None and not isinstance(value, Mapping):
                    raw = float(value) if statistic == "mean" else None
                cell = "--" if raw is None else format_metric_cell("accuracy", raw)
                # Mark only the headline column: repeating the count on
                # every percentile would say the same thing eight times.
                if (statistic == statistics[0] and failures and raw is not None
                        and n_files is not None):
                    cell += f"\\textsuperscript{{({int(n_files) - int(failures)})}}"
                cells.append(cell)

            if inline_ber:
                raw = self._stat_of(value, f"ber_{ber_statistics[0]}")
                cells.append("--" if raw is None else format_metric_cell("ber", raw))

            if show_emr:
                cells.append(format_emr_cell(
                    self._stat_of(value, "emr_count"),
                    n_files,
                    self._stat_of(value, "emr_rate"),
                ))

            rows.append("    " + " & ".join(cells) + " \\\\")

        return build_longtable(
            "l" + "c" * (len(headers) - 1),
            " & ".join(headers),
            rows,
            f"Watermark detection accuracy for {caption_word}.",
            label,
        )

    def _metric_table(self, sorted_attacks, metric, statistics,
                      caption_word, label):
        """One metric, one column per configured statistic."""
        headers = ["Attack Type"] + [stat_header(s) for s in statistics]

        rows = []
        for attack_name, value in sorted_attacks:
            cells = [display_attack_name(attack_name)]
            n_files = self._stat_of(value, "accuracy_n")
            metric_n = self._stat_of(value, f"{metric}_n")
            for statistic in statistics:
                raw = self._stat_of(value, f"{metric}_{statistic}")
                if raw is None:
                    cells.append("N/A")
                    continue
                cell = format_metric_cell(metric, raw)
                # Say once per row, not once per column, that this metric
                # scored fewer files than the accuracy beside it.
                if (statistic == statistics[0] and metric_n is not None
                        and n_files is not None and metric_n < n_files):
                    cell += f" ($n$={int(metric_n)})"
                cells.append(cell)
            rows.append("    " + " & ".join(cells) + " \\\\")

        return build_longtable(
            "l" + "c" * (len(headers) - 1),
            " & ".join(headers),
            rows,
            f"{metric_label(metric)} statistics for {caption_word}.",
            label,
        )

    def _compact_metric_table(self, sorted_attacks, metric_statistics,
                              caption_word, label):
        """Metrics reduced to a single statistic, one column each."""
        headers = ["Attack Type"]
        for metric, statistic in metric_statistics:
            header = metric_label(metric)
            # Say which statistic this is unless it is the mean, which is
            # what an unlabelled quality column has always meant.
            if statistic != "mean":
                header += f" [{stat_header(statistic)}]"
            headers.append(header)

        rows = []
        for attack_name, value in sorted_attacks:
            cells = [display_attack_name(attack_name)]
            n_files = self._stat_of(value, "accuracy_n")
            for metric, statistic in metric_statistics:
                raw = self._stat_of(value, f"{metric}_{statistic}")
                if raw is None:
                    cells.append("N/A")
                    continue
                cell = format_metric_cell(metric, raw)
                metric_n = self._stat_of(value, f"{metric}_n")
                if metric_n is not None and n_files is not None and metric_n < n_files:
                    cell += f" ($n$={int(metric_n)})"
                cells.append(cell)
            rows.append("    " + " & ".join(cells) + " \\\\")

        return build_longtable(
            "l" + "c" * (len(headers) - 1),
            " & ".join(headers),
            rows,
            f"Audio quality and intelligibility for {caption_word}.",
            label,
        )

    def _footnotes(self, sorted_attacks, silent_metrics=()):
        """Coverage notes for the tables just built."""
        any_failures = any(
            self._stat_of(value, "detection_failures", 0)
            for _, value in sorted_attacks
        )
        notes = ""
        if any_failures:
            notes += (
                "\n\n{\\noindent\\footnotesize A superscript count marks values "
                "that include files where the detector returned no usable "
                "watermark; those files are scored at the random-guess floor "
                "(50\\%), not measured.}\n"
            )
        if silent_metrics:
            names = ", ".join(metric_label(m) for m in silent_metrics)
            notes += (
                "\n\n{\\noindent\\footnotesize Enabled in the configuration but "
                f"not reported here, because no value was produced for any "
                f"attack in this section: {names}. This usually means the "
                "metric's service or optional package was unavailable; see "
                "\\texttt{run\\_metadata.json}.}\n"
            )
        return notes

    # ------------------------------------------------------------------
    # Document assembly
    # ------------------------------------------------------------------

    def _grouped_sections(self, stats, label_suffix=""):
        """One section per attack family present in ``stats``."""
        grouped = group_attacks(list(stats))
        ordered = [k for k in GROUP_ORDER if k in grouped]
        if OTHER_GROUP_KEY in grouped:
            ordered.append(OTHER_GROUP_KEY)

        sections = []
        for key in ordered:
            subset = {
                name: stats[name] for name in grouped[key]["attacks"]
                if name in stats
            }
            table = self.generate_latex_table(subset, key, label_suffix)
            if not table:
                continue
            sections.append(
                "\\needspace{5\\baselineskip}\n"
                f"\\section{{{group_label(key, grouped[key]['label'])}}}\n\n"
                f"\\noindent Results computed over "
                f"{self._n_files_phrase(subset)}.\n\n{table}"
            )
        return sections

    @staticmethod
    def _n_files_phrase(stats):
        counts = {
            value.get("accuracy_n") for value in stats.values()
            if isinstance(value, Mapping) and value.get("accuracy_n") is not None
        }
        if len(counts) == 1:
            n = counts.pop()
            return f"{int(n)} audio {'file' if n == 1 else 'files'}"
        return f"{len(stats)} attack(s)"

    def calculate_mean_accuracy(self, stats: Dict[str, StatsValue]) -> float:
        """Mean of the per-attack values, weighting every attack equally.

        This averages over the attacks actually run, not over any defined
        attack distribution, so it moves when the attack selection changes.
        Report it alongside that count (see ``generate_latex_report``).
        """
        if not stats:
            return 0.0
        accuracies = [self._accuracy_of(v, name) for name, v in stats.items()]
        return sum(accuracies) / len(accuracies)

    def generate_latex_report(self, stats: Dict[str, StatsValue],
                              model_name: str = "DeepMark",
                              chart_filename: str = "benchmark_chart.png",
                              crop_before_attack: float = None,
                              containers=None) -> str:
        """
        Generate complete LaTeX report content.

        Args:
            stats: Dictionary with attack names as keys and per-attack stats
            model_name: Name of the watermarking model
            chart_filename: Filename of the generated chart
            crop_before_attack: If set, percentage cropped before attacks

        Returns:
            Complete LaTeX document as string
        """
        mean_accuracy = self.calculate_mean_accuracy(stats)
        attack_count = len(stats)

        preamble = self._preamble(
            f"Benchmark Report: {model_name}",
            "DeepMark Benchmark System",
        )

        attack_word = "attack type" if attack_count == 1 else "different attack types"
        coverage_phrase = (
            "a single attack type" if attack_count == 1
            else f"{attack_count} {attack_word}"
        )

        crop_note = self._crop_note(crop_before_attack)
        if crop_note:
            crop_note = " " + crop_note

        statistic = self._accuracy_label_for(stats)
        chart_block = figure_block(
            chart_filename,
            f"Attacks ranked by watermark detection accuracy ({statistic}), "
            f"worst first. Bar colour is the robustness tier; the dashed line "
            f"is the accuracy a failed detection already scores, so a bar "
            f"reaching it carries no information.",
            "fig:benchmark_chart",
        )

        summary = (
            "\\section{Summary}\n\n"
            f"\\noindent\\textbf{{Accuracy ({statistic}) across the "
            f"{attack_count} attacks run:}} {mean_accuracy:.2f}\\%\n"
            "\\\\{\\footnotesize Unweighted mean of the per-attack values; it "
            "depends on which attacks were selected.}\n\n"
            + self._key_findings(stats)
            + self._embedding_cost_line(stats)
            + chart_block
            + "\n" + self._performance_breakdown(stats)
        )

        sections = "\n\n".join(self._grouped_sections(stats))
        sections += "\n\n" + container_section(containers or [])

        return (
            f"{preamble}\n\n"
            "% -------------------- Abstract --------------------\n"
            "\\begin{abstract}\n"
            f"This report presents the benchmark results for the {model_name} "
            f"watermarking model across various attack scenarios. The evaluation "
            f"covers {coverage_phrase}, measuring the robustness of watermark "
            f"detection under adversarial conditions using the DeepMark benchmark "
            f"framework.{crop_note}\n"
            "\\end{abstract}\n\n"
            f"{summary}\n\n"
            f"{sections}\n\n"
            "\\end{document}"
        )

    @staticmethod
    def _crop_note(crop_before_attack) -> str:
        """The caveat that every attack ran on cropped audio.

        A reader who does not see it takes the numbers for the whole
        signal, so it belongs in every shape of the report rather than
        only in the one that happens to carry an abstract.
        """
        if crop_before_attack is None:
            return ""
        return (
            f"\\textcolor{{red}}{{A crop of {crop_before_attack:.1f}\\% was applied to the beginning of "
            f"the watermarked audio prior to each attack. The original (reference) "
            f"audio was cropped identically, so quality metrics compare cropped "
            f"original vs.\\ cropped attacked audio, and BER is measured by detecting "
            f"the watermark from the cropped attacked signal.}}"
        )

    def _embedding_cost_line(self, stats):
        """Embedding time, stated once for the run rather than per attack.

        It is measured once per file and does not depend on which attack
        follows, so it belongs in a sentence about the run, not in a
        column of a table whose rows are attacks.
        """
        metric = "embed_latency"
        if not self.resolver.is_enabled(None, metric):
            return ""

        # Identical on every attack entry, because both are run-level costs
        # copied onto each of them; any row carries the same values.
        row = next(iter(stats.values()), None)
        if not isinstance(row, Mapping):
            return ""

        parts = []
        for statistic in self.resolver.statistics_for(None, metric):
            value = row.get(f"{metric}_{statistic}")
            if value is not None:
                parts.append(
                    f"{float(value):.4f}\\,s ({stat_header(statistic).lower()})"
                )
        if not parts:
            return ""

        return (
            f"\\noindent\\textbf{{Embedding cost per file:}} "
            f"{', '.join(parts)}\n"
            "\\\\{\\footnotesize Measured once per file, before any attack, so it "
            "does not vary by attack. Like every timing it depends on this "
            "machine and does not reproduce across runs.}\n\n"
        )

    def _key_findings(self, stats):
        """Name the attacks a reader would otherwise have to find by hand.

        The tier list below counts attacks per band; this says which ones.
        Everything here is read off the same statistics the tables print,
        so the prose cannot drift from the numbers under it.
        """
        accuracy_by_attack = {
            display_attack_name(name): self._accuracy_of(value, name)
            for name, value in stats.items()
        }
        if len(accuracy_by_attack) < 2:
            return ""

        ordered = sorted(accuracy_by_attack.items(), key=lambda kv: kv[1])
        floor = self._chance_floor
        statistic = self._accuracy_label_for(stats).lower()

        worst = ordered[:3]
        best = ordered[-1]
        at_floor = [name for name, score in ordered if score <= floor + 1.0]

        items = [
            "  \\item \\textbf{Most damaging:} "
            + ", ".join(f"{name} ({score:.1f}\\%)" for name, score in worst)
            + f" --- {statistic} accuracy.",
            f"  \\item \\textbf{{Least damaging:}} {best[0]} "
            f"({best[1]:.1f}\\%).",
        ]
        if at_floor:
            word = "attack leaves" if len(at_floor) == 1 else "attacks leave"
            items.append(
                f"  \\item \\textbf{{At or below the chance floor "
                f"({floor:.0f}\\%):}} {len(at_floor)} {word} the detector no "
                f"better than guessing --- {', '.join(at_floor)}."
            )
        else:
            items.append(
                f"  \\item \\textbf{{Chance floor ({floor:.0f}\\%):}} no "
                f"attack drove detection down to it."
            )

        spread = ordered[-1][1] - ordered[0][1]
        items.append(
            f"  \\item \\textbf{{Spread:}} {spread:.1f} percentage points "
            f"between the best and worst attack, so the headline mean above "
            f"stands for a wide range rather than a typical case."
        )

        return (
            "\\noindent\\textbf{Key findings}\n\n\\begin{itemize}\n"
            + "\n".join(items) + "\n\\end{itemize}\n\n"
        )

    def _performance_breakdown(self, stats):
        """The four robustness tiers, as an itemised list."""
        accuracy_by_attack = {
            name: self._accuracy_of(value, name)
            for name, value in stats.items()
        }
        tiers = [
            ("Excellent Performance ($\\geq$95\\%)",
             [a for a in accuracy_by_attack.values() if a >= 95]),
            ("Good Performance (85-95\\%)",
             [a for a in accuracy_by_attack.values() if 85 <= a < 95]),
            ("Fair Performance (70-85\\%)",
             [a for a in accuracy_by_attack.values() if 70 <= a < 85]),
            ("Poor Performance ($<$70\\%)",
             [a for a in accuracy_by_attack.values() if a < 70]),
        ]

        items = ""
        for label, members in tiers:
            if not members:
                continue
            word = "attack" if len(members) == 1 else "attacks"
            items += f"  \\item \\textbf{{{label}:}} {len(members)} {word}\n"

        if not items:
            return ""
        return (
            "\\noindent The watermarking model demonstrates the following "
            "levels of robustness:\n\n\\begin{itemize}\n" + items + "\\end{itemize}\n"
        )

    def _is_grouped_stats(self, stats):
        """Detect if stats dict is duration-grouped format."""
        if not stats:
            return False
        first_val = next(iter(stats.values()))
        return isinstance(first_val, dict) and "stats" in first_val and "n_files" in first_val

    def generate_full_report(self, stats_file: str = "benchmark_stats.json",
                           model_name: str = "DeepMark",
                           crop_before_attack: float = None,
                           containers=None):
        """
        Generate complete benchmark report with chart and LaTeX document.

        Supports both flat stats (single group) and duration-grouped stats
        (per-group sections in the report).
        """
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)

            if self._is_grouped_stats(stats):
                return self._generate_grouped_report(
                    stats, model_name, crop_before_attack, containers,
                )

            logger.info(f"Loaded benchmark statistics for {len(stats)} attacks")

            chart_path = os.path.join(self.report_dir, "benchmark_chart.png")
            self.create_gradient_bar_chart(stats, chart_path)

            latex_content = self.generate_latex_report(
                stats, model_name, "benchmark_chart.png",
                crop_before_attack=crop_before_attack, containers=containers,
            )

            latex_path = os.path.join(self.report_dir, "benchmark_report.tex")
            with open(latex_path, 'w') as f:
                f.write(latex_content)

            logger.info(f"LaTeX report saved to {latex_path}")
            self._compile(latex_path)
            return latex_path, chart_path

        except FileNotFoundError:
            logger.error(f"Benchmark statistics file not found: {stats_file}")
            raise
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise

    def _compile(self, latex_path):
        from deepmarkpy.utils.latex_helpers import compile_latex

        compile_latex(self.report_dir,
                      os.path.splitext(os.path.basename(latex_path))[0])

    def _generate_grouped_report(self, grouped_stats, model_name,
                                 crop_before_attack, containers=None):
        """Generate report with per-duration-group sections."""
        preamble = self._preamble(
            f"{model_name} Benchmark Results (by Duration)",
            "DeepMark Benchmark",
        )

        parts = []
        first_group_stats = None

        for group_label_text, group_data in grouped_stats.items():
            group_stats = group_data["stats"]
            n_files = group_data["n_files"]

            if first_group_stats is None:
                first_group_stats = group_stats

            safe_label = duration_label_tex(group_label_text)
            # \label names must be plain ASCII under pdflatex; ≥ is not.
            suffix = slugify(group_label_text)
            sections = self._grouped_sections(group_stats, label_suffix=suffix)

            # Each bin gets its own ranking chart. Drawing only the first
            # one left every other part of the report without a figure and
            # captioned as if it described that part's numbers.
            filename = f"benchmark_chart_{slugify(group_label_text)}.png"
            figure = ""
            if self.create_gradient_bar_chart(
                group_stats, os.path.join(self.report_dir, filename),
            ):
                figure = figure_block(
                    filename,
                    f"Attacks ranked by detection accuracy "
                    f"({self._accuracy_label_for(group_stats)}) for "
                    f"{safe_label}, worst first.",
                    f"fig:benchmark_chart_{slugify(group_label_text)}",
                ) + "\n"

            parts.append(
                part_heading(safe_label, f"{n_files} files")
                + self._embedding_cost_line(group_stats)
                + figure
                + self._key_findings(group_stats)
                + "\n\n".join(sections)
            )

        trend = self._duration_trend_figure(grouped_stats)

        # The crop applies to every bin, so it is stated once above them
        # rather than repeated in each part.
        crop_note = self._crop_note(crop_before_attack)
        if crop_note:
            crop_note = f"\\noindent {crop_note}\n\n"

        latex_content = (
            f"{preamble}\n\n" + crop_note + trend + "\n\n".join(parts)
            + "\n\n" + container_section(containers or [])
            + "\n\n\\end{document}"
        )

        latex_path = os.path.join(self.report_dir, "benchmark_report.tex")
        with open(latex_path, 'w') as f:
            f.write(latex_content)
        logger.info(f"LaTeX report saved to {latex_path}")

        # The unsuffixed chart names the report as a whole, so it shows the
        # combined bin when the configuration asked for one.
        chart_path = os.path.join(self.report_dir, "benchmark_chart.png")
        overall = grouped_stats.get("Overall", {}).get("stats") or first_group_stats
        if overall:
            self.create_gradient_bar_chart(overall, chart_path)

        self._compile(latex_path)
        return latex_path, chart_path

    def _duration_trend_figure(self, grouped_stats):
        """Every attack's accuracy across the duration bins, in one figure.

        The parts below are self-contained, so nothing else in the report
        lets the bins be compared; without this, a reader asking whether
        robustness depends on clip length has to page between sections and
        hold six numbers in their head.
        """
        bins = [(label, data["stats"]) for label, data in grouped_stats.items()
                if label != "Overall"]
        if len(bins) < 2:
            return ""

        series = {}
        for label, group_stats in bins:
            for attack, value in group_stats.items():
                series.setdefault(display_attack_name(attack), []).append(
                    (label, self._accuracy_of(value, attack))
                )

        filename = "duration_trend.png"
        drawn = report_charts.duration_trend(
            series, os.path.join(self.report_dir, filename),
            statistic_label=self._accuracy_label_for(
                {a: v for _, s in bins for a, v in s.items()},
            ),
            chance_floor=self._chance_floor,
        )
        if not drawn:
            return ""
        return figure_block(
            filename,
            "Detection accuracy per attack across the duration groups. A "
            "line that slopes means the attack's effect depends on how long "
            "the file is; a flat one means the per-group sections below "
            "repeat the same result.",
            "fig:duration_trend",
        ) + "\n\n"


def generate_benchmark_report(stats_file: str = "benchmark_stats.json",
                            model_name: str = "DeepMark",
                            report_dir: str = "report",
                            resolver: MetricResolver = None,
                            crop_before_attack: float = None,
                            is_zero_bit: bool = False,
                            containers=None):
    """
    Convenience function to generate a complete benchmark report.

    Args:
        stats_file: Path to the benchmark statistics JSON file
        model_name: Name of the watermarking model
        report_dir: Directory to save the report files
        resolver: metric/statistic configuration for every table
        crop_before_attack: If set, percentage cropped before attacks
        is_zero_bit: model reports detection rather than bit agreement,
            which moves the chance-floor line the charts draw to 0

    Returns:
        Tuple of (latex_path, chart_path)
    """
    generator = BenchmarkReportGenerator(report_dir, resolver=resolver,
                                         is_zero_bit=is_zero_bit)
    return generator.generate_full_report(
        stats_file, model_name, crop_before_attack=crop_before_attack,
        containers=containers,
    )
