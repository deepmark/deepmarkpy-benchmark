"""Resolves which metrics and statistics apply to an attack group.

Every report generator asks this module rather than carrying its own
constants, so a table's columns are exactly what the config file asked
for. The rules, in full:

**Enablement**, per ``(group, metric)`` pair -- first hit wins:

1. ``metrics.per_group.<group>.<metric>.enabled``
2. ``metrics.per_group.<parent>.<metric>.enabled`` (subgroups only)
3. ``metrics.defaults.<metric>.enabled``
4. disabled

Inheritance is per *metric*, not per block: a group section listing only
``pesq`` inherits every other metric from ``defaults`` untouched.

**Statistics**, per ``(group, metric)`` pair -- first list that exists
wins, and report columns appear in *that list's* order:

1. ``metrics.per_group.<group>.<metric>.statistics``
2. ``metrics.per_group.<parent>.<metric>.statistics`` (subgroups only)
3. ``metrics.defaults.<metric>.statistics``
4. the top-level ``statistics`` list
5. all eight statistics

Two metrics are special. ``accuracy`` is always enabled -- it is the
measurement the benchmark exists to make. ``emr`` is a count and a rate,
so it has no statistics.

When ``calculate_quality_metrics`` is false, the enablement rules above
are bypassed for the signal metrics: only accuracy and the always-on
trio (PESQ/ViSQOL/STOI) are computed, uniformly across every group.
``ber`` and ``emr`` are derived from accuracy at no extra cost, so they
keep honouring their configured flags. Statistics resolve normally in
both cases.
"""

import logging

from deepmarkpy.utils.attack_groups import (
    ATTACK_GROUPS,
    ATTACK_SUBGROUPS,
    OTHER_GROUP_KEY,
    get_group_for_attack,
    get_subgroup_for_attack,
    group_parent,
)

logger = logging.getLogger(__name__)

ALL_STATISTICS = (
    "mean", "std", "median", "p5", "p10", "p95", "p99", "worst_case",
)

# Metrics derived from the accuracy array rather than from a signal
# comparison. They cost nothing extra and exist in every group.
ROBUSTNESS_METRICS = ("accuracy", "ber", "emr")

# Signal metrics, split the way reports lay them out: NISQA's five
# dimensions get their own table so the audio-quality table stays narrow
# enough for the page.
QUALITY_METRICS = ("pesq", "psnr", "si_sdr", "mcd", "visqol")
INTELLIGIBILITY_METRICS = ("stoi", "sii", "ncm")
NISQA_METRICS = (
    "nisqa_mos", "nisqa_noi", "nisqa_dis", "nisqa_col", "nisqa_loud",
)

# What the run cost in time and resources. Measured during the run like
# the robustness metrics -- not by comparing two signals -- but unlike
# every other family these do not reproduce: they describe the machine,
# not the watermarking method. Kept in their own family so a report can
# say so, and so memory and anything later joins here rather than being
# mistaken for a property of the model.
EFFICIENCY_METRICS = (
    "embed_latency", "detect_latency", "attack_latency", "container_footprint",
)

METRIC_BUCKETS = {
    "robustness": ROBUSTNESS_METRICS,
    "quality": QUALITY_METRICS,
    "intelligibility": INTELLIGIBILITY_METRICS,
    "nisqa": NISQA_METRICS,
    "efficiency": EFFICIENCY_METRICS,
}

# Canonical display order. Reports iterate this so two tables built from
# different code paths order their columns identically.
CANONICAL_METRIC_ORDER = (
    ROBUSTNESS_METRICS + QUALITY_METRICS + INTELLIGIBILITY_METRICS
    + NISQA_METRICS + EFFICIENCY_METRICS
)

# Metrics computed by ``metrics.compute_metrics`` -- i.e. everything except
# the accuracy-derived ones.
SIGNAL_METRICS = frozenset(
    QUALITY_METRICS + INTELLIGIBILITY_METRICS + NISQA_METRICS
)

# Embedding happens once per file, whatever attacks follow, so its time
# does not vary by attack. Listing it in a table whose rows are attacks
# repeats one number down the column and implies a dependence that is not
# there, so the reports state it once per run instead.
PER_FILE_EFFICIENCY_METRICS = frozenset({"embed_latency"})

# Measured once for the whole run, not per file and not per attack: it is
# the memory the model's container holds, which does not move with the
# audio. One number, so no statistic applies to it either.
PER_MODEL_EFFICIENCY_METRICS = frozenset({"container_footprint"})

# Efficiency metrics are seconds, and every one of them is better small.
LOWER_IS_BETTER_EFFICIENCY = frozenset(EFFICIENCY_METRICS)

# Metrics a lower value is better on. Every other metric here improves as
# it rises, so ``worst_case_of`` reads these at their maximum.
LOWER_IS_BETTER_METRICS = frozenset({"mcd", "ber"}) | LOWER_IS_BETTER_EFFICIENCY

# Enabled regardless of the per-group matrix when quality metrics are off.
ALWAYS_ON_METRICS = ("pesq", "visqol", "stoi")

# accuracy is the point of the benchmark; it cannot be switched off.
MANDATORY_METRICS = frozenset({"accuracy"})

# emr reports "n of N files recovered exactly" -- a count and a rate, not a
# distribution, so no statistic applies to it.
STATISTICS_EXEMPT_METRICS = frozenset({"emr"}) | PER_MODEL_EFFICIENCY_METRICS


def worst_case_of(values, metric):
    """The worst value in ``values`` for ``metric``.

    "Worst" is the end of the range the metric calls bad, which is not
    always the smallest number: accuracy and PESQ are worst at their
    minimum, but latency, MCD and BER are worst at their maximum. Taking
    the minimum for all of them reports the *best* case of every
    lower-is-better metric under the label "worst case".
    """
    import numpy as np

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return None
    return float(np.max(array) if metric in LOWER_IS_BETTER_METRICS
                 else np.min(array))


class MetricResolver:
    """Answers metric/statistic questions for one benchmark mode's config."""

    def __init__(
        self,
        defaults=None,
        per_group=None,
        statistics=None,
        calculate_quality_metrics=True,
        efficiency=None,
    ):
        """
        Args:
            defaults: ``{metric: {"enabled": bool, "statistics": [...]}}``
                from ``metrics.defaults``. Keys may omit either field.
            per_group: ``{group: {metric: {...}}}`` from
                ``metrics.per_group``.
            statistics: the top-level ``statistics`` list, or None when the
                config omits it (then all eight apply).
            calculate_quality_metrics: see the module docstring.
            efficiency: the ``efficiency`` config section --
                ``{"enabled": bool, "metrics": {name: {...}}}``. Absent or
                disabled, the whole family is off no matter what its
                metrics say, because the timings cost a measurement that a
                run may not want taken.
        """
        self.defaults = dict(defaults or {})
        self.per_group = {k: dict(v) for k, v in (per_group or {}).items()}
        self.statistics = list(statistics) if statistics else None
        self.calculate_quality_metrics = bool(calculate_quality_metrics)

        efficiency = dict(efficiency or {})
        self.efficiency_enabled = bool(efficiency.get("enabled", False))
        self.efficiency_metrics = dict(efficiency.get("metrics") or {})

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_attack_groups(cls, calculate_quality_metrics=True):
        """Build the resolver the shipped config templates encode.

        Reproduces the metric matrix declared by ``ATTACK_GROUPS`` and
        ``ATTACK_SUBGROUPS``, with PESQ, ViSQOL and STOI added to every
        top-level group, so a run with no ``metrics`` block in its config
        behaves the same as one using an unedited ``--init`` file. This is
        also what report generators fall back to when constructed without a
        config, which keeps them usable as a library.
        """
        defaults = {m: {"enabled": True} for m in ROBUSTNESS_METRICS}
        # Everything on by default; the per-group sections below carry the
        # exclusions, which is also how the shipped templates are written --
        # a group that excludes nothing then needs no section at all.
        for metric in SIGNAL_METRICS:
            defaults[metric] = {"enabled": True}

        per_group = {}
        for key in list(ATTACK_GROUPS) + list(ATTACK_SUBGROUPS):
            enabled = cls._declared_metrics(key)
            if key in ATTACK_GROUPS:
                # PESQ, ViSQOL and STOI are computed for every attack when
                # quality metrics are off, so turning them on must not take
                # the trio away.
                enabled |= frozenset(ALWAYS_ON_METRICS)
            per_group[key] = {
                metric: {"enabled": metric in enabled}
                for metric in SIGNAL_METRICS
            }
        # Attacks outside every declared group have no metric opinion
        # attached to them, so they get the full signal set rather than
        # silently none.
        per_group[OTHER_GROUP_KEY] = {
            metric: {"enabled": True} for metric in SIGNAL_METRICS
        }

        return cls(
            defaults=defaults,
            per_group=per_group,
            statistics=None,
            calculate_quality_metrics=calculate_quality_metrics,
        )

    @staticmethod
    def _declared_metrics(group_key):
        """The metric set ``group_key`` declares in the taxonomy."""
        definition = ATTACK_GROUPS.get(group_key) or ATTACK_SUBGROUPS[group_key]
        return frozenset(
            list(definition.get("quality_metrics", []))
            + list(definition.get("intelligibility_metrics", []))
            + list(definition.get("nisqa_metrics", []))
        )

    # ------------------------------------------------------------------
    # Enablement
    # ------------------------------------------------------------------

    def _lookup_chain(self, group_key):
        """Config sections to consult for ``group_key``, most specific first."""
        chain = []
        if group_key is not None:
            chain.append(group_key)
            parent = group_parent(group_key)
            if parent is not None:
                chain.append(parent)
        return chain

    def is_enabled(self, group_key, metric):
        """Whether ``metric`` is on for ``group_key``."""
        if metric in EFFICIENCY_METRICS:
            # Its own section, not the metrics block: a timing is a
            # measurement of the machine, and a run asks for it or does
            # not, independently of which quality metrics it wants.
            if not self.efficiency_enabled:
                return False
            entry = self.efficiency_metrics.get(metric)
            if entry is not None and "enabled" in entry:
                return bool(entry["enabled"])
            return True

        if metric in MANDATORY_METRICS:
            return True

        if not self.calculate_quality_metrics and metric in SIGNAL_METRICS:
            return metric in ALWAYS_ON_METRICS

        for section in self._lookup_chain(group_key):
            entry = self.per_group.get(section, {}).get(metric)
            if entry is not None and "enabled" in entry:
                return bool(entry["enabled"])

        entry = self.defaults.get(metric)
        if entry is not None and "enabled" in entry:
            return bool(entry["enabled"])

        return False

    def metrics_for_group(self, group_key, bucket=None):
        """Enabled metrics for ``group_key`` in canonical display order.

        Args:
            group_key: group, subgroup, or ``"other"``. ``None`` resolves
                against ``metrics.defaults`` alone.
            bucket: restrict to one of ``METRIC_BUCKETS`` (``"robustness"``,
                ``"quality"``, ``"intelligibility"``, ``"nisqa"``). None
                returns every bucket.
        """
        candidates = (
            METRIC_BUCKETS[bucket] if bucket else CANONICAL_METRIC_ORDER
        )
        return [
            metric for metric in candidates
            if self.is_enabled(group_key, metric)
        ]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _efficiency_statistics(self, metric):
        """Statistics for an efficiency metric, from its own section."""
        entry = self.efficiency_metrics.get(metric) or {}
        configured = entry.get("statistics")
        return list(configured) if configured else None

    def statistics_for(self, group_key, metric):
        """Statistics for ``(group_key, metric)``, in report-column order."""
        if metric in STATISTICS_EXEMPT_METRICS:
            return []

        if metric in EFFICIENCY_METRICS:
            configured = self._efficiency_statistics(metric)
            if configured is not None:
                return configured
            # Falls through to the top-level list, so a config that sets
            # statistics once gets them here too.

        for section in self._lookup_chain(group_key):
            entry = self.per_group.get(section, {}).get(metric)
            if entry is not None and entry.get("statistics") is not None:
                return list(entry["statistics"])

        entry = self.defaults.get(metric)
        if entry is not None and entry.get("statistics") is not None:
            return list(entry["statistics"])

        if self.statistics is not None:
            return list(self.statistics)

        return list(ALL_STATISTICS)

    def signal_metrics_for_group(self, group_key):
        """Enabled metrics for a group that ``compute_metrics`` can produce.

        Excludes accuracy, BER and EMR, which come from the accuracy array
        rather than from comparing two signals.
        """
        return [
            metric for metric in self.metrics_for_group(group_key)
            if metric in SIGNAL_METRICS
        ]

    def statistics_by_metric(self, group_key):
        """``{metric: [statistics]}`` for every metric enabled in a group."""
        return {
            metric: self.statistics_for(group_key, metric)
            for metric in self.metrics_for_group(group_key)
        }

    # ------------------------------------------------------------------
    # Attack-oriented views (used by the run loop)
    # ------------------------------------------------------------------

    def group_for_attack(self, attack_name):
        """The reporting group an attack belongs to, or ``"other"``."""
        return get_group_for_attack(attack_name) or OTHER_GROUP_KEY

    def metrics_for_attack(self, attack_name):
        """Signal metrics that must be computed for ``attack_name``.

        An attack is rendered under its group in the basic and
        detection-reliability reports and under its subgroup in the
        detailed report, and those two sections can enable different
        metrics. Computing the union means neither report has a hole.
        """
        group_key = self.group_for_attack(attack_name)
        needed = set(self.metrics_for_group(group_key)) & SIGNAL_METRICS

        subgroup = get_subgroup_for_attack(attack_name)
        if subgroup is not None:
            needed |= set(self.metrics_for_group(subgroup)) & SIGNAL_METRICS

        return [m for m in CANONICAL_METRIC_ORDER if m in needed]

    def all_signal_metrics(self):
        """Every signal metric enabled anywhere -- the watermark-only baseline set.

        The "no attack (watermark only)" row appears in every group's table,
        so it has to carry any metric that any group might display.
        """
        needed = set()
        for group_key in list(self.per_group) + [None]:
            needed |= set(self.metrics_for_group(group_key)) & SIGNAL_METRICS
        return [m for m in CANONICAL_METRIC_ORDER if m in needed]

    def any_signal_metric_enabled(self):
        """Whether any group asks for a metric beyond the accuracy-derived ones."""
        return bool(self.all_signal_metrics())
