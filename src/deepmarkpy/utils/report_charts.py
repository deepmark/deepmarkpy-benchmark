"""Figures for the DeepMark reports.

Every generator draws through this module, so a colour, a tier boundary
and a reference line mean the same thing in all five reports.

Two rules hold everywhere:

* **Nothing is drawn from a constant.** Which metric a figure plots, and
  which statistic it reads, is decided by the caller from the
  ``MetricResolver`` the config built -- the same source the tables use.
* **A figure never breaks a report.** Each function returns ``True`` when
  it wrote a file and ``False`` when the data it needs was missing or the
  draw failed; the caller then omits the figure instead of pointing
  ``\\includegraphics`` at a file that does not exist.

Categorical charts are sorted worst-first rather than alphabetically,
because the reader is looking for the worst case. The headline ranking
uses vertical bars; the charts that stack or pair values per attack are
horizontal, where long attack names fit on one line.
"""

import functools
import logging

import matplotlib
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

BRAND = "#469CA9"
INK = "#2c3e50"
LABEL = "#333333"
MUTED = "#777777"
GRID = "#dddddd"
PANEL = "#fafafa"

# The four robustness tiers the reports already speak in, given colours
# once so a bar, a scatter point and a heat cell agree on what "poor"
# looks like. Ordered best-first; the threshold is the lower bound.
TIERS = (
    (95.0, "Excellent (≥95%)", "#3C9D6E"),
    (85.0, "Good (85–95%)", BRAND),
    (70.0, "Fair (70–85%)", "#E0A030"),
    (0.0, "Poor (<70%)", "#C0392B"),
)

# Distinct model colours, reused by every figure that separates models.
MODEL_COLORS = (
    "#039FAC", "#E74C3C", "#2ECC71", "#F39C12",
    "#9B59B6", "#1ABC9C", "#E67E22", "#3498DB",
)


# Private-use stand-ins for characters a later rule would otherwise eat:
# an escaped "$" must survive the math-mode "$" removal, and a literal
# backslash must not start another escape. Restored at the very end.
_DOLLAR, _BACKSLASH = "", ""

_LATEX_TO_TEXT = (
    ("\\textbackslash{}", _BACKSLASH),
    ("\\textasciitilde{}", "~"),
    ("\\textasciicircum{}", "^"),
    ("\\textsuperscript{0}", "⁰"),
    ("---", "—"),
    ("--", "–"),
    ("\\%", "%"),
    ("\\_", "_"),
    ("\\&", "&"),
    ("\\#", "#"),
    ("\\{", "{"),
    ("\\}", "}"),
    ("\\$", _DOLLAR),
    ("$\\sim$", "~"),
    ("$\\geq$", "≥"),
    ("$<$", "<"),
    ("$>$", ">"),
    ("$", ""),
    (_DOLLAR, "$"),
    (_BACKSLASH, "\\"),
)


def plain(text):
    """Render a LaTeX fragment as the plain text a figure can draw.

    Callers hand these charts the same labels the tables use --
    ``ViSQOL (1--5)``, ``Codec2Vocoder\\_700`` -- because a figure that
    named things differently from the table beside it would be a second
    vocabulary to learn. The conversion happens here rather than at every
    call site.
    """
    if not isinstance(text, str):
        return text
    for source, target in _LATEX_TO_TEXT:
        text = text.replace(source, target)
    return text


def direction_hint(label, higher_is_better):
    """Append the reading direction to a metric label when it is not the usual one.

    Every metric here improves as it rises except MCD, and a bar chart
    gives no clue which way round it is: a taller MCD bar is a worse
    result. The tables carry the metric's range in its label, so the
    figures say the rest.
    """
    return label if higher_is_better else f"{label} — lower is better"


def tier_color(accuracy):
    """Colour for an accuracy value, by robustness tier."""
    for threshold, _, color in TIERS:
        if accuracy >= threshold:
            return color
    return TIERS[-1][2]


def _chart(fn):
    """Make a chart function total: it returns False instead of raising.

    A report that loses a figure is still a report; one that raises part
    way through writing its ``.tex`` is not.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return bool(fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001 - a figure is never fatal
            logger.warning("Chart '%s' skipped: %s", fn.__name__, exc)
            plt.close("all")
            return False
    return wrapper


def _style(ax, xlabel=None, ylabel=None, title=None, grid_axis="x"):
    ax.set_facecolor(PANEL)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cccccc")
    if xlabel:
        ax.set_xlabel(plain(xlabel), fontsize=12, fontweight="bold", color=LABEL)
    if ylabel:
        ax.set_ylabel(plain(ylabel), fontsize=12, fontweight="bold", color=LABEL)
    if title:
        ax.set_title(plain(title), fontsize=14, fontweight="bold", pad=18, color=INK)
    if grid_axis:
        ax.grid(axis=grid_axis, alpha=0.45, linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#555555", labelsize=10)


def _save(fig, output_path):
    fig.savefig(output_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    logger.info("Chart saved to %s", output_path)
    return True


def _tier_legend(ax, accuracies, chance_floor=None, inside=False):
    """Legend listing only the tiers actually present.

    ``inside`` puts it in the axes' top-left corner, which the ascending
    sort guarantees is empty; otherwise it goes under the axes, clear of
    the bars and of a rotated tick label.
    """
    handles = [
        matplotlib.patches.Patch(facecolor=color, edgecolor="white", label=label)
        for _, label, color in TIERS
        if any(tier_color(a) == color for a in accuracies)
    ]
    if chance_floor is not None:
        handles.append(matplotlib.lines.Line2D(
            [], [], color="#C0392B", linestyle="--", linewidth=1.2,
            label=f"chance floor ({chance_floor:.0f}%)",
        ))
    if len(handles) < 2:
        return
    if inside:
        ax.legend(handles=handles, loc="upper left", frameon=True,
                  framealpha=0.92, edgecolor="#dddddd", fontsize=9.5)
        return
    ax.legend(handles=handles, loc="upper center", frameon=False, fontsize=9.5,
              bbox_to_anchor=(0.5, _below_axes(ax.get_figure())),
              ncol=min(len(handles), 5))


def _bar_height(n, per_row=0.30, floor=2.3):
    """Figure height that grows with the number of categories."""
    return max(floor, per_row * n + 1.8)


def _below_axes(fig, inches=0.62):
    """``bbox_to_anchor`` y that sits a fixed distance under the axes.

    A legend placed at a fixed *fraction* below the axes lands on the x
    label of a short chart and far away from a tall one, because these
    figures size themselves by how many bars they hold. Converting a real
    distance into that fraction keeps the gap the same on both.
    """
    axes_height = max(fig.get_figheight() * 0.72, 0.75)
    return -(inches / axes_height)


# ---------------------------------------------------------------------------
# Basic report
# ---------------------------------------------------------------------------

@_chart
def accuracy_ranking(values, output_path, statistic_label="Mean",
                     chance_floor=None, title=None):
    """Attacks as vertical bars, ranked worst-first, coloured by tier.

    Args:
        values: ``{attack display name: accuracy in percent}``.
        statistic_label: which statistic the values are, for the axis.
        chance_floor: accuracy a failed detection lands on (50 for a
            multi-bit model, 0 for zero-bit). Drawn as a reference line so
            "low" can be read against "no better than guessing".
        title: overrides the default, for a report that draws one of these
            per section and would otherwise repeat the same heading.

    Vertical bars, which is the shape this chart has always had, but
    ordered by severity rather than alphabetically and with the bar colour
    carrying the robustness tier, so the worst cases are the first thing
    read and the tier boundaries need no lookup.
    """
    if not values:
        return False

    ordered = sorted(values.items(), key=lambda kv: kv[1])
    names = [plain(name) for name, _ in ordered]
    scores = [score for _, score in ordered]

    # Rotated attack names add real height under the axes, so the plot area
    # is kept wide and shallow to stop the figure squaring up and taking a
    # third of the page.
    width = max(10.0, 0.7 * len(names) + 3.5)
    fig, ax = plt.subplots(figsize=(min(width, 20.0), 4.6))
    fig.patch.set_facecolor("white")

    positions = np.arange(len(names))
    ax.bar(positions, scores, color=[tier_color(s) for s in scores],
           alpha=0.9, edgecolor="white", linewidth=1.2, width=0.72)

    for x, score in zip(positions, scores):
        ax.text(x, score + 1.5, f"{score:.1f}", ha="center", va="bottom",
                fontsize=9, color="#555555")

    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9,
                       color="#555555")
    ax.set_xlim(-0.7, len(names) - 0.3)
    ax.set_ylim(0, 108)

    if chance_floor is not None:
        ax.axhline(chance_floor, color="#C0392B", linestyle="--",
                   linewidth=1.2, alpha=0.8)

    _style(ax, xlabel="Attack type",
           ylabel=f"Detection accuracy ({statistic_label}, %)",
           title=title or
           f"Attacks ranked by detection accuracy ({statistic_label})",
           grid_axis="y")
    _tier_legend(ax, scores, chance_floor, inside=True)
    return _save(fig, output_path)


@_chart
def robustness_quality_scatter(points, output_path, metric_label,
                               higher_is_better=True, chance_floor=None):
    """Watermark survival against how much of the audio survived.

    Args:
        points: ``[(attack display name, quality value, accuracy %)]``.
        metric_label: axis label for the quality metric.
        higher_is_better: direction of the quality metric, which decides
            which side of the plot means "audio preserved".
        chance_floor: horizontal reference line for a failed detection.

    The quadrant worth looking at is the one where the audio is preserved
    and the watermark is not: an attack there removes the mark at no
    perceptual cost, which is the only kind an adversary can actually
    use. The opposite corner destroys the recording to do it.
    """
    usable = [(n, q, a) for n, q, a in points if q is not None and a is not None]
    if len(usable) < 3:
        return False

    names = [plain(n) for n, _, _ in usable]
    quality = np.array([q for _, q, _ in usable], dtype=float)
    accuracy = np.array([a for _, _, a in usable], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5.6))
    fig.patch.set_facecolor("white")

    q_split = float(np.median(quality))
    a_split = chance_floor if chance_floor is not None else float(np.median(accuracy))

    # Shade the quadrant an attacker would aim for.
    preserved_right = higher_is_better
    x_lo, x_hi = quality.min(), quality.max()
    pad = (x_hi - x_lo) * 0.12 or 1.0
    span = (q_split, x_hi + pad) if preserved_right else (x_lo - pad, q_split)
    ax.fill_between(span, -5, a_split, color="#C0392B", alpha=0.07, zorder=0)

    ax.axvline(q_split, color=MUTED, linestyle=":", linewidth=1.0)
    ax.axhline(a_split, color="#C0392B", linestyle="--", linewidth=1.2,
               alpha=0.8)

    ax.scatter(quality, accuracy, s=110,
               c=[tier_color(a) for a in accuracy],
               edgecolor="white", linewidth=1.2, zorder=3)

    # Attacks of similar severity land close together, so labels are
    # placed round the four corners in turn rather than all to one side.
    corners = ((9, 6, "left"), (9, -13, "left"),
               (-9, 6, "right"), (-9, -13, "right"))
    for index, (name, x, y) in enumerate(zip(names, quality, accuracy)):
        dx, dy, align = corners[index % len(corners)]
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=(dx, dy), fontsize=8, color="#555555", ha=align)

    ax.set_xlim(x_lo - pad, x_hi + pad)
    ax.set_ylim(-5, 108)
    ax.text(
        0.99 if preserved_right else 0.01,
        0.02,
        "audio preserved,\nwatermark lost",
        transform=ax.transAxes, fontsize=9.5, color="#C0392B",
        fontweight="bold", va="bottom",
        ha="right" if preserved_right else "left",
    )
    _style(ax, xlabel=metric_label, ylabel="Detection accuracy (%)",
           title="Watermark survival against audio quality", grid_axis="both")
    return _save(fig, output_path)


@_chart
def attack_strength_curves(series, output_path, statistic_label="Mean",
                           chance_floor=None):
    """Accuracy across the configured versions of the same attack.

    Args:
        series: ``{attack base name: [(version label, accuracy %)]}``, the
            versions in the order the configuration declared them.

    Five rows of a table say the same thing, but only a curve shows where
    the watermark stops surviving.
    """
    return _line_series(
        series, output_path,
        xlabel="Configured attack version (weakest to strongest)",
        ylabel=f"Detection accuracy ({statistic_label}, %)",
        title="Accuracy across attack strengths",
        chance_floor=chance_floor,
    )


@_chart
def duration_trend(series, output_path, statistic_label="Mean",
                   chance_floor=None):
    """Accuracy of each attack across the duration bins.

    Args:
        series: ``{attack display name: [(bin label, accuracy %)]}`` in bin
            order.

    With duration groups the report is one self-contained part per bin;
    this is the only place the bins are put side by side, which is where
    a length dependency becomes visible.
    """
    return _line_series(
        series, output_path,
        xlabel="File duration group",
        ylabel=f"Detection accuracy ({statistic_label}, %)",
        title="Accuracy by file duration",
        chance_floor=chance_floor,
    )


def _line_series(series, output_path, xlabel, ylabel, title,
                 chance_floor=None):
    """One line per key over a shared categorical x axis."""
    usable = {k: v for k, v in series.items() if len(v) >= 2}
    if not usable:
        return False

    # Every label any series uses, in first-seen order; each point is
    # drawn at its own label.
    tick_labels = []
    for entries in usable.values():
        for label, _ in entries:
            if label not in tick_labels:
                tick_labels.append(label)
    position = {label: index for index, label in enumerate(tick_labels)}

    width = max(8.0, 1.5 * len(tick_labels) + 3.0)
    fig, ax = plt.subplots(figsize=(min(width, 13), 4.6))
    fig.patch.set_facecolor("white")

    annotate = len(usable) <= 4
    for index, (base, entries) in enumerate(usable.items()):
        xs = [position[label] for label, _ in entries]
        ys = [value for _, value in entries]
        color = MODEL_COLORS[index % len(MODEL_COLORS)]
        ax.plot(xs, ys, marker="o", markersize=7, linewidth=2.2,
                color=color, label=plain(base), alpha=0.9)
        if annotate:
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                            xytext=(0, 9), fontsize=8.5, color=color,
                            ha="center")

    ax.set_xticks(np.arange(len(tick_labels)))
    ax.set_xticklabels([plain(label) for label in tick_labels],
                       fontsize=10, color="#555555")
    ax.set_ylim(-5, 112)
    handles = None
    if chance_floor is not None:
        ax.axhline(chance_floor, color="#C0392B", linestyle="--",
                   linewidth=1.2, alpha=0.8)
        handles, _ = ax.get_legend_handles_labels()
        handles.append(matplotlib.lines.Line2D(
            [], [], color="#C0392B", linestyle="--", linewidth=1.2,
            label=f"chance floor ({chance_floor:.0f}%)",
        ))
    ax.legend(handles=handles, frameon=False, fontsize=10,
              loc="upper center", ncol=min(len(usable) + 1, 4),
              bbox_to_anchor=(0.5, _below_axes(fig, inches=0.75)))
    _style(ax, xlabel=xlabel, ylabel=ylabel, title=title, grid_axis="y")
    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# Detailed report
# ---------------------------------------------------------------------------

@_chart
def per_file_outcome_bars(distributions, output_path, title,
                          chance_floor=50.0, is_zero_bit=False):
    """How the files split by outcome under each attack, worst first.

    Args:
        distributions: ``{attack display name: [per-file accuracy, ...]}``.
        chance_floor: accuracy a failed detection already scores.
        is_zero_bit: the model reports detection, not bit agreement.

    A mean hides whether every file degraded a little or half of them were
    destroyed, and this report is the only one holding the per-file values
    to answer that. A box plot cannot: a zero-bit detector scores each file
    0 or 100, so its quartiles collapse onto the two ends and the picture
    is empty. Counting the files into outcome bands says the same thing for
    a graded score and stays readable for a binary one.

    Returns ``False`` when every row sits wholly in one and the same band:
    the bars are a composition, so they always reach 100%, and identical
    full-width bars say nothing the table above does not.
    """
    usable = {k: [v for v in vals if v is not None]
              for k, vals in distributions.items()}
    usable = {k: v for k, v in usable.items() if v}
    if not usable:
        return False

    if is_zero_bit:
        # Nothing lands between the two ends, so a third band would be a
        # segment that is always empty.
        bands = (("Detected", "#3C9D6E"), ("Not detected", "#C0392B"))

        def classify(value):
            return 0 if value > chance_floor else 1
    else:
        bands = (("Bit-exact (100%)", "#3C9D6E"),
                 ("Above chance", BRAND),
                 ("At or below chance", "#C0392B"))

        def classify(value):
            if value >= 100.0:
                return 0
            return 1 if value > chance_floor else 2

    shares = {}
    for name, values in usable.items():
        counts = [0] * len(bands)
        for value in values:
            counts[classify(value)] += 1
        shares[name] = (counts, len(values))

    # Every bar spans the full width, because it is a composition: the
    # bands always add up to all the files. That is fine when the bands
    # differ, and actively misleading when they do not -- a row of solid
    # full-width bars beside an accuracy table reads as "every model
    # scored 100%". When every row sits wholly in one and the same band
    # there is nothing here the table does not already say, so the figure
    # declines to be drawn.
    occupied = {
        tuple(index for index, count in enumerate(counts) if count)
        for counts, _ in shares.values()
    }
    if len(occupied) == 1 and len(next(iter(occupied))) == 1:
        logger.info(
            "Outcome chart skipped for '%s': every file falls in the same "
            "band, so the bars would be identical and full width.", title,
        )
        return False

    # Worst first: the most files in the failing band, ties broken by the
    # fewest in the best band.
    ordered = sorted(
        shares.items(),
        key=lambda kv: (-kv[1][0][-1] / kv[1][1], kv[1][0][0] / kv[1][1]),
    )
    names = [plain(name) for name, _ in ordered]
    total_files = ordered[0][1][1]

    fig, ax = plt.subplots(figsize=(10, _bar_height(len(names), per_row=0.42)))
    fig.patch.set_facecolor("white")

    positions = np.arange(len(names))
    left = np.zeros(len(names))
    for index, (label, color) in enumerate(bands):
        widths = np.array([
            100.0 * counts[index] / n for _, (counts, n) in ordered
        ])
        ax.barh(positions, widths, left=left, height=0.66, color=color,
                alpha=0.9, edgecolor="white", linewidth=1.0, label=label)
        for y, (width, (_, (counts, _n))) in enumerate(zip(widths, ordered)):
            if width >= 9.0:
                ax.text(left[y] + width / 2, y, str(counts[index]),
                        ha="center", va="center", fontsize=9, color="white",
                        fontweight="bold")
        left += widths

    ax.set_yticks(positions)
    ax.set_yticklabels(names, fontsize=10, color="#555555")
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, _below_axes(fig)),
              ncol=len(bands), frameon=False, fontsize=9.5)

    counts_differ = len({n for _, (_c, n) in ordered}) > 1
    suffix = "" if counts_differ else f" (n={total_files} files)"
    _style(ax, xlabel="Share of files (%)", title=f"{title}{suffix}")
    return _save(fig, output_path)


@_chart
def quality_against_baseline(values, output_path, metric_label, baseline=None,
                             higher_is_better=True, title=None):
    """Each attack's audio quality, read against embedding alone.

    Args:
        values: ``{attack display name: metric value}``.
        baseline: the same metric on the watermarked, unattacked signal.

    Without the baseline a quality number cannot be attributed: part of
    the loss was already paid at embedding time.
    """
    usable = {k: v for k, v in values.items() if v is not None}
    if not usable:
        return False

    ordered = sorted(usable.items(), key=lambda kv: kv[1],
                     reverse=not higher_is_better)
    names = [plain(name) for name, _ in ordered]
    scores = [value for _, value in ordered]

    fig, ax = plt.subplots(figsize=(10, _bar_height(len(names))))
    fig.patch.set_facecolor("white")

    positions = np.arange(len(names))
    ax.barh(positions, scores, color=BRAND, alpha=0.85,
            edgecolor="white", linewidth=1.0, height=0.72)
    for y, score in zip(positions, scores):
        ax.text(score, y, f" {score:.2f}", va="center", ha="left",
                fontsize=9, color="#555555")

    ax.set_yticks(positions)
    ax.set_yticklabels(names, fontsize=10, color="#555555")
    ax.invert_yaxis()

    if baseline is not None:
        ax.axvline(baseline, color="#C0392B", linestyle="--", linewidth=1.4,
                   label=f"no attack ({baseline:.2f})")
        # In a legend rather than floating beside the line: with one or two
        # bars there is no room inside the axes for it to land.
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, _below_axes(fig)),
                  frameon=False, fontsize=9.5)

    _style(ax, xlabel=metric_label,
           title=title or f"{metric_label} after each attack")
    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# no_attacks report
# ---------------------------------------------------------------------------

def metric_axis_range(metric, values):
    """The y range a bar chart of ``metric`` should use.

    A metric with a defined range gets that range, so the bar height is the
    share of the scale the model actually achieved: PESQ 4.1 fills most of
    1--4.66 and PESQ 2.0 does not. Letting matplotlib fit the axis to the
    data instead makes every bar reach the top whatever the value, which
    reads as a good result even when it is a poor one.

    A dB figure has no ceiling, so it scales to the data -- anchored at
    zero when the values are positive, so lengths stay proportional.
    """
    from deepmarkpy.utils.metrics import METRIC_RANGES

    # Accuracy is a percentage and belongs on 0--100 whatever the values
    # are. It is not in METRIC_RANGES because that sits beside the signal
    # metrics' labels, and accuracy has none there.
    if metric == "accuracy":
        return (0.0, 100.0)
    if metric in METRIC_RANGES:
        return METRIC_RANGES[metric]

    usable = [v for v in values if v is not None]
    if not usable:
        return None
    low, high = min(usable), max(usable)
    span = (high - low) or abs(high) or 1.0
    return (min(0.0, low - 0.1 * span), high + 0.15 * span)


@_chart
def metric_by_model(values, output_path, metric, metric_label,
                    higher_is_better=True, reference=None):
    """One metric, one bar per model, on that metric's own scale.

    Args:
        values: ``{model: value}``.
        metric: the metric key, which decides the axis range.
        metric_label: the label the tables use, for the axis.
        reference: ``(value, label)`` drawn as a dashed line -- the
            chance floor, for accuracy, so a bar can be read against what
            a failed detection already scores.
    """
    usable = {k: v for k, v in values.items() if v is not None}
    if not usable:
        return False

    names = list(usable)
    scores = [usable[name] for name in names]

    fig, ax = plt.subplots(figsize=(max(4.0, 1.3 * len(names) + 2.2), 3.6))
    fig.patch.set_facecolor("white")

    positions = np.arange(len(names))
    colors = [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(names))]
    ax.bar(positions, scores, color=colors, alpha=0.9, edgecolor="white",
           linewidth=1.2, width=min(0.6, 0.25 * len(names)))

    for x, score in zip(positions, scores):
        ax.text(x, score, f" {score:.2f}", ha="center", va="bottom",
                fontsize=10, color="#555555")

    ax.set_xticks(positions)
    ax.set_xticklabels([plain(n) for n in names], fontsize=10, color="#555555")

    bounds = metric_axis_range(metric, scores)
    if bounds:
        low, high = bounds
        ax.set_ylim(low, high)
        # Say which end is good, since a fixed range means a short bar is a
        # real result rather than a cropped one.
        best = high if higher_is_better else low
        ax.axhline(best, color="#3C9D6E", linestyle=":", linewidth=1.2,
                   alpha=0.8)
        ax.text(len(names) - 0.5, best, " best", va="center", ha="left",
                fontsize=9, color="#3C9D6E", clip_on=False)

    if reference is not None:
        value, label = reference
        ax.axhline(value, color="#C0392B", linestyle="--", linewidth=1.2,
                   alpha=0.8)
        ax.text(len(names) - 0.5, value, f" {label}", va="center", ha="left",
                fontsize=9, color="#C0392B", clip_on=False)

    _style(ax, ylabel=metric_label,
           title=f"{metric_label} by model", grid_axis="y")
    return _save(fig, output_path)


@_chart
def accuracy_by_model(series, output_path, models, statistic_labels,
                      chance_floor=None, title="Detection accuracy by model"):
    """Accuracy per model, one bar per configured statistic.

    Args:
        series: ``[(statistic label, [value per model])]``.
        models: the x labels.
        chance_floor: what a failed detection already scores.

    One grouped figure rather than one figure per statistic: the config
    may ask for two statistics or seven, and seven separate charts would
    bury the tables. Grouping them also puts a model's mean next to its
    worst case, which is the comparison being made.
    """
    usable = [(label, values) for label, values in series
              if any(v is not None for v in values)]
    if not usable or not models:
        return False

    fig, ax = plt.subplots(
        figsize=(max(4.5, 1.1 * len(models) * len(usable) + 2.5), 3.8)
    )
    fig.patch.set_facecolor("white")

    positions = np.arange(len(models))
    span = 0.8
    width = span / len(usable)
    for index, (label, values) in enumerate(usable):
        offset = -span / 2 + width * (index + 0.5)
        heights = [0.0 if v is None else float(v) for v in values]
        ax.bar(positions + offset, heights, width=width * 0.9,
               color=MODEL_COLORS[index % len(MODEL_COLORS)], alpha=0.9,
               edgecolor="white", linewidth=1.0, label=plain(label))
        for x, value in zip(positions + offset, values):
            if value is not None:
                ax.text(x, value, f"{value:.1f}", ha="center", va="bottom",
                        fontsize=8, color="#555555", rotation=90)

    ax.set_xticks(positions)
    ax.set_xticklabels([plain(m) for m in models], fontsize=10,
                       color="#555555")
    ax.set_ylim(0, 112)

    handles = None
    if chance_floor is not None:
        ax.axhline(chance_floor, color="#C0392B", linestyle="--",
                   linewidth=1.2, alpha=0.8)
        handles, _ = ax.get_legend_handles_labels()
        handles.append(matplotlib.lines.Line2D(
            [], [], color="#C0392B", linestyle="--", linewidth=1.2,
            label=f"chance floor ({chance_floor:.0f}%)",
        ))
    ax.legend(handles=handles, frameon=False, fontsize=9.5,
              loc="upper center", ncol=min(len(usable) + 1, 4),
              bbox_to_anchor=(0.5, _below_axes(fig)))

    _style(ax, ylabel="Accuracy (%)", title=title, grid_axis="y")
    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# detection_reliability report
# ---------------------------------------------------------------------------

@_chart
def false_positive_negative_bars(rows, output_path, baseline_fp=None,
                                 baseline_fn=None):
    """Both error rates per attack, worst false-negative first.

    Args:
        rows: ``[(attack display name, FP rate %, FN rate %)]``.
        baseline_fp / baseline_fn: the no-attack rates, drawn as reference
            lines -- an attack only matters insofar as it moves the
            detector off the operating point it already had.

    Returns ``False`` when every rate is zero. A group in which the
    detector never erred produces a row of empty axes that says less than
    the table's column of zeros, and the report is better off without it.
    """
    usable = [(n, fp, fn) for n, fp, fn in rows
              if fp is not None or fn is not None]
    if not usable:
        return False

    ordered = sorted(usable, key=lambda r: (r[2] or 0.0), reverse=True)
    names = [plain(n) for n, _, _ in ordered]
    fps = [fp or 0.0 for _, fp, _ in ordered]
    fns = [fn or 0.0 for _, _, fn in ordered]

    if not any(fps) and not any(fns):
        logger.info(
            "Error-rate chart skipped for %s: the detector made no false "
            "positives and no false negatives, so every bar is zero.",
            output_path,
        )
        return False

    width = max(9.0, 0.7 * len(names) + 3.5)
    fig, ax = plt.subplots(figsize=(min(width, 20.0), 4.6))
    fig.patch.set_facecolor("white")

    positions = np.arange(len(names))
    bar = 0.36
    ax.bar(positions - bar / 2, fns, width=bar, color="#C0392B",
           alpha=0.9, edgecolor="white", linewidth=1.0,
           label="False negative (mark missed)")
    ax.bar(positions + bar / 2, fps, width=bar, color="#E0A030",
           alpha=0.9, edgecolor="white", linewidth=1.0,
           label="False positive (mark claimed)")

    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9,
                       color="#555555")
    ax.set_xlim(-0.7, len(names) - 0.3)
    ax.set_ylim(0, max(105.0, max(fps + fns) * 1.15))

    if baseline_fn is not None:
        ax.axhline(baseline_fn, color="#C0392B", linestyle="--", linewidth=1.2,
                   alpha=0.7)
    if baseline_fp is not None:
        ax.axhline(baseline_fp, color="#E0A030", linestyle="--", linewidth=1.2,
                   alpha=0.7)

    ax.legend(frameon=False, fontsize=9.5, loc="upper center",
              bbox_to_anchor=(0.5, _below_axes(fig)), ncol=2)
    _style(ax, ylabel="Rate (% of attempts)",
           title="Detector error rates under attack", grid_axis="y")
    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# Comparative report
# ---------------------------------------------------------------------------

@_chart
def accuracy_heatmap(attacks, models, matrix, output_path,
                     statistic_label="Mean"):
    """Attack x model accuracy grid.

    Args:
        attacks: row labels, already display-formatted.
        models: column labels.
        matrix: ``[[value or None per model] per attack]``.

    A radar with thirty spokes needs a legend to be read at all; a grid
    of the same numbers is read directly, and it keeps working as attacks
    are added.
    """
    if len(models) < 2 or len(attacks) < 2:
        return False

    data = np.array(
        [[np.nan if v is None else float(v) for v in row] for row in matrix],
        dtype=float,
    )
    if np.all(np.isnan(data)):
        return False

    fig, ax = plt.subplots(
        figsize=(max(5.0, 1.55 * len(models) + 3.4),
                 max(3.6, 0.38 * len(attacks) + 2.0))
    )
    fig.patch.set_facecolor("white")

    # Grey for a model/attack pair that was never run, so an absent
    # measurement cannot be read as a low score.
    cmap = matplotlib.colormaps["RdYlGn"].with_extremes(bad="#eeeeee")
    image = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=0, vmax=100,
                      aspect="auto")

    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([plain(m) for m in models], fontsize=10, color="#555555")
    ax.set_yticks(np.arange(len(attacks)))
    ax.set_yticklabels([plain(a) for a in attacks], fontsize=9.5, color="#555555")
    ax.set_xticks(np.arange(len(models) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(attacks) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)

    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]
            if np.isnan(value):
                ax.text(col, row, "n/a", ha="center", va="center",
                        fontsize=8, color="#999999")
                continue
            ax.text(col, row, f"{value:.1f}", ha="center", va="center",
                    fontsize=8.5,
                    color="white" if value < 35 or value > 92 else "#333333")

    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, shrink=0.85)
    bar.set_label(plain(f"Detection accuracy ({statistic_label}, %)"),
                  fontsize=10, color=LABEL)
    bar.ax.tick_params(labelsize=9, colors="#555555")

    ax.set_title(plain(f"Detection accuracy by attack and model "
                       f"({statistic_label})"),
                 fontsize=14, fontweight="bold", pad=16, color=INK)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    return _save(fig, output_path)


# ---------------------------------------------------------------------------
# Shared helpers for callers
# ---------------------------------------------------------------------------

def split_version(attack_name):
    """Split ``"GaussianNoiseAttack (mild)"`` into base and version."""
    if attack_name.endswith(")") and " (" in attack_name:
        index = attack_name.index(" (")
        return attack_name[:index], attack_name[index + 2:-1]
    return attack_name, None


def version_series(ordered_names, value_of):
    """Group multi-version attacks into ladders, in configuration order.

    Args:
        ordered_names: attack keys in the order the run produced them,
            which is the order the configuration declared the versions in.
        value_of: ``name -> value or None``.

    Returns:
        ``{base name: [(version label, value)]}`` for the bases that have
        two or more versions with a value. Attacks that were not expanded
        into versions produce nothing, so a run without a ladder simply
        has no such figure.
    """
    series = {}
    for name in ordered_names:
        base, version = split_version(name)
        if version is None:
            continue
        value = value_of(name)
        if value is None:
            continue
        series.setdefault(base, []).append((version, float(value)))

    from deepmarkpy.utils.latex_helpers import display_attack_name
    return {
        display_attack_name(base): entries
        for base, entries in series.items() if len(entries) >= 2
    }
