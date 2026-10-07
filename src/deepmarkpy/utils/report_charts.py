"""The basic report's figures: the accuracy ranking and the attack-strength curves.

The caller picks the statistic a figure reads. Each chart returns ``True``
when it wrote its file and ``False`` when the data was missing or the draw
failed, so the caller can leave the figure out.
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
GRID = "#dddddd"
PANEL = "#fafafa"

# The four robustness tiers the report lists, as the ranking's bar
# colours. Ordered best-first; the threshold is the lower bound.
TIERS = (
    (95.0, "Excellent (≥95%)", "#3C9D6E"),
    (85.0, "Good (85–95%)", BRAND),
    (70.0, "Fair (70–85%)", "#E0A030"),
    (0.0, "Poor (<70%)", "#C0392B"),
)

# Distinct colours, one per strength curve.
MODEL_COLORS = (
    "#039FAC", "#E74C3C", "#2ECC71", "#F39C12",
    "#9B59B6", "#1ABC9C", "#E67E22", "#3498DB",
)


# Private-use stand-ins that keep an escaped "$" and a literal backslash
# out of the later rules; restored last.
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
    """Render a LaTeX table label, e.g. ``ViSQOL (1--5)``, as plain text to draw."""
    if not isinstance(text, str):
        return text
    for source, target in _LATEX_TO_TEXT:
        text = text.replace(source, target)
    return text


def tier_color(accuracy):
    """Colour for an accuracy value, by robustness tier."""
    for threshold, _, color in TIERS:
        if accuracy >= threshold:
            return color
    return TIERS[-1][2]


def _chart(fn):
    """Make a chart function return False, with a warning, instead of raising."""
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


def _tier_legend(ax, accuracies, chance_floor=None):
    """Legend in the axes' empty top-left corner, listing the tiers present."""
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
    if len(handles) >= 2:
        ax.legend(handles=handles, loc="upper left", frameon=True,
                  framealpha=0.92, edgecolor="#dddddd", fontsize=9.5)


def _below_axes(fig, inches):
    """``bbox_to_anchor`` y for a legend ``inches`` under the axes.

    Converted from inches, so the gap is the same on short and tall charts.
    """
    axes_height = max(fig.get_figheight() * 0.72, 0.75)
    return -(inches / axes_height)


# ---------------------------------------------------------------------------
# Basic report
# ---------------------------------------------------------------------------

@_chart
def accuracy_ranking(values, output_path, statistic_label="Mean",
                     chance_floor=None):
    """Attacks as vertical bars, ranked worst-first, coloured by tier.

    Args:
        values: ``{attack display name: accuracy in percent}``.
        statistic_label: which statistic the values are, for the axis.
        chance_floor: accuracy a failed detection lands on (50 for a
            multi-bit model, 0 for zero-bit), drawn as a dashed line.
    """
    if not values:
        return False

    ordered = sorted(values.items(), key=lambda kv: kv[1])
    names = [plain(name) for name, _ in ordered]
    scores = [score for _, score in ordered]

    # Wide and shallow: the rotated attack names add height under the axes.
    width = max(10.0, 0.7 * len(names) + 3.5)
    fig, ax = plt.subplots(figsize=(min(width, 20.0), 4.6))

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
           title=f"Attacks ranked by detection accuracy ({statistic_label})",
           grid_axis="y")
    _tier_legend(ax, scores, chance_floor)
    return _save(fig, output_path)


@_chart
def attack_strength_curves(series, output_path, statistic_label="Mean",
                           chance_floor=None):
    """Accuracy across the configured versions of the same attack.

    Args:
        series: ``{attack base name: [(version label, accuracy %)]}``, the
            versions in the order the configuration declared them.
    """
    return _line_series(
        series, output_path,
        xlabel="Configured attack version (weakest to strongest)",
        ylabel=f"Detection accuracy ({statistic_label}, %)",
        title="Accuracy across attack strengths",
        chance_floor=chance_floor,
    )


def _line_series(series, output_path, xlabel, ylabel, title,
                 chance_floor=None):
    """One line per key over a shared categorical x axis."""
    usable = {k: v for k, v in series.items() if len(v) >= 2}
    if not usable:
        return False

    # Every label any series uses; a new label goes right after the previous
    # label of its own series, so each series reads in its own order.
    tick_labels = []
    for entries in usable.values():
        after = -1
        for label, _ in entries:
            if label in tick_labels:
                after = tick_labels.index(label)
            else:
                after += 1
                tick_labels.insert(after, label)
    position = {label: index for index, label in enumerate(tick_labels)}

    width = max(8.0, 1.5 * len(tick_labels) + 3.0)
    fig, ax = plt.subplots(figsize=(min(width, 13), 4.6))

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
        two or more versions with a value.
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
