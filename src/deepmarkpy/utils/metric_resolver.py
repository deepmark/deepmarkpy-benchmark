"""Which metrics and statistics each attack group reports, as the config sets them.

Each ``metrics`` setting resolves per ``(group, metric)``, most specific
first: ``metrics.per_group.<group>``, then the parent group's section for a
subgroup, then ``metrics.defaults``. A metric nothing enables is off;
statistics nothing configures fall back to the top-level ``statistics``
list, then to all eight, and columns follow the list's order. ``accuracy``
is always on and ``emr`` has no statistics. Efficiency metrics are set in
the ``efficiency`` section, and with ``calculate_quality_metrics`` off the
only signal metrics on are PESQ, ViSQOL and STOI.
"""

from deepmarkpy.utils.attack_groups import (
    ATTACK_GROUPS,
    ATTACK_SUBGROUPS,
    OTHER_GROUP_KEY,
    get_group_for_attack,
    get_subgroup_for_attack,
    group_parent,
)

ALL_STATISTICS = (
    "mean", "std", "median", "p5", "p10", "p95", "p99", "worst_case",
)

# Derived from the accuracy array; every group has them.
ROBUSTNESS_METRICS = ("accuracy", "ber", "emr")

# Signal metrics, by the table the reports put them in.
QUALITY_METRICS = ("pesq", "psnr", "si_sdr", "mcd", "visqol")
INTELLIGIBILITY_METRICS = ("stoi", "sii", "ncm")
NISQA_METRICS = (
    "nisqa_mos", "nisqa_noi", "nisqa_dis", "nisqa_col", "nisqa_loud",
)

# What the run cost in time and memory. They describe the machine, not the
# watermarking method, and do not reproduce across runs.
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

# Canonical display order of every table's metrics.
CANONICAL_METRIC_ORDER = (
    ROBUSTNESS_METRICS + QUALITY_METRICS + INTELLIGIBILITY_METRICS
    + NISQA_METRICS + EFFICIENCY_METRICS
)

# Metrics ``metrics.compute_metrics`` computes by comparing two signals.
SIGNAL_METRICS = frozenset(
    QUALITY_METRICS + INTELLIGIBILITY_METRICS + NISQA_METRICS
)

# Measured once per file, so the reports state it once rather than per attack.
PER_FILE_EFFICIENCY_METRICS = frozenset({"embed_latency"})

# Measured once per run, for the model's container; no statistic applies.
PER_MODEL_EFFICIENCY_METRICS = frozenset({"container_footprint"})

# Metrics that are better low; ``worst_case_of`` reads them at their maximum.
LOWER_IS_BETTER_METRICS = frozenset({"mcd", "ber"}) | frozenset(EFFICIENCY_METRICS)

# Enabled regardless of the per-group matrix when quality metrics are off.
ALWAYS_ON_METRICS = ("pesq", "visqol", "stoi")

# accuracy is the point of the benchmark; it cannot be switched off.
MANDATORY_METRICS = frozenset({"accuracy"})

# No statistic applies: emr is a count and a rate, container memory one number.
STATISTICS_EXEMPT_METRICS = frozenset({"emr"}) | PER_MODEL_EFFICIENCY_METRICS


def worst_case_of(values, metric):
    """The worst value in ``values`` for ``metric``, or None when empty.

    The maximum for a lower-is-better metric, the minimum otherwise.
    """
    import numpy as np

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return None
    return float(np.max(array) if metric in LOWER_IS_BETTER_METRICS
                 else np.min(array))


def compute_statistics(values, statistics=None, metric="accuracy"):
    """``statistics`` of ``values`` (all eight when None), in canonical order.

    None when there are no values; ``metric`` decides which end is the worst
    case. The values keep their own dtype.
    """
    import numpy as np

    if len(values) == 0:
        return None
    wanted = set(ALL_STATISTICS if statistics is None else statistics)
    array = np.array(values)
    available = {
        "mean": lambda: float(np.mean(array)),
        "std": lambda: float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "median": lambda: float(np.median(array)),
        "p5": lambda: float(np.percentile(array, 5)),
        "p10": lambda: float(np.percentile(array, 10)),
        "p95": lambda: float(np.percentile(array, 95)),
        "p99": lambda: float(np.percentile(array, 99)),
        "worst_case": lambda: worst_case_of(array, metric),
    }
    return {
        name: compute() for name, compute in available.items()
        if name in wanted
    }


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
            calculate_quality_metrics: False leaves PESQ, ViSQOL and STOI
                as the only signal metrics on.
            efficiency: the ``efficiency`` config section --
                ``{"enabled": bool, "metrics": {name: {...}}}``. Absent or
                disabled, every efficiency metric is off.
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
        """The resolver the shipped config templates encode.

        The ``ATTACK_GROUPS`` and ``ATTACK_SUBGROUPS`` metric matrix, with
        PESQ, ViSQOL and STOI on in every top-level group. It applies when a
        config has no ``metrics`` block and when a generator gets no resolver.
        """
        defaults = {m: {"enabled": True} for m in ROBUSTNESS_METRICS}
        # Everything on by default; the per-group sections carry the exclusions.
        for metric in SIGNAL_METRICS:
            defaults[metric] = {"enabled": True}

        per_group = {}
        for key in list(ATTACK_GROUPS) + list(ATTACK_SUBGROUPS):
            enabled = cls._declared_metrics(key)
            if key in ATTACK_GROUPS:
                enabled |= frozenset(ALWAYS_ON_METRICS)
            per_group[key] = {
                metric: {"enabled": metric in enabled}
                for metric in SIGNAL_METRICS
            }
        # Attacks outside every declared group get every signal metric.
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
            # Set in the efficiency section, not the metrics block.
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
                ``"quality"``, ``"intelligibility"``, ``"nisqa"``,
                ``"efficiency"``). None returns every bucket.
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

    def statistics_for(self, group_key, metric):
        """Statistics for ``(group_key, metric)``, in report-column order."""
        if metric in STATISTICS_EXEMPT_METRICS:
            return []

        if metric in EFFICIENCY_METRICS:
            entry = self.efficiency_metrics.get(metric) or {}
            if entry.get("statistics"):
                return list(entry["statistics"])
            # Unset there, it resolves like any other metric.

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

    # ------------------------------------------------------------------
    # Attack-oriented views (used by the run loop)
    # ------------------------------------------------------------------

    def group_for_attack(self, attack_name):
        """The reporting group an attack belongs to, or ``"other"``."""
        return get_group_for_attack(attack_name) or OTHER_GROUP_KEY

    def metrics_for_attack(self, attack_name):
        """Signal metrics to compute for ``attack_name``.

        The union of its group's and its subgroup's, since the detailed
        report shows an attack under its subgroup and the others under its
        group.
        """
        group_key = self.group_for_attack(attack_name)
        needed = set(self.metrics_for_group(group_key)) & SIGNAL_METRICS

        subgroup = get_subgroup_for_attack(attack_name)
        if subgroup is not None:
            needed |= set(self.metrics_for_group(subgroup)) & SIGNAL_METRICS

        return [m for m in CANONICAL_METRIC_ORDER if m in needed]

    def all_signal_metrics(self):
        """Every signal metric any group enables: the no-attack baseline's set."""
        needed = set()
        for group_key in list(self.per_group) + [None]:
            needed |= set(self.metrics_for_group(group_key)) & SIGNAL_METRICS
        return [m for m in CANONICAL_METRIC_ORDER if m in needed]

    def any_signal_metric_enabled(self):
        """Whether any group asks for a metric beyond the accuracy-derived ones."""
        return bool(self.all_signal_metrics())
