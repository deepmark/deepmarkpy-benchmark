"""Shared LaTeX helpers for DeepMark report generators.

Centralizes preamble generation, attack-name formatting, longtable
scaffolding and pdflatex compilation so the three report generators
(basic, detailed, comparative) share a single implementation.
"""

import logging
import os
import shutil
import subprocess
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


_DEFAULT_ARTICLE_PACKAGES = (
    "booktabs",
    "graphicx",
    "amsmath",
    "cleveref",
    "float",
    "longtable",
    "needspace",
)

_DEFAULT_DEEPMARK_PACKAGES = (
    "float",
    "longtable",
    "needspace",
)


def make_preamble(
    title: str,
    author: str,
    has_deepmark_cls: bool,
    extra_packages: Iterable[str] = (),
) -> str:
    """Build a LaTeX preamble, using the ``deepmark`` class when available.

    Args:
        title: Document title.
        author: Document author.
        has_deepmark_cls: ``True`` if ``deepmark.cls`` is present in the
            report dir (enables the branded class). ``False`` falls back
            to a plain ``article``.
        extra_packages: Extra ``\\usepackage{...}`` names to append.
    """
    if has_deepmark_cls:
        packages = list(_DEFAULT_DEEPMARK_PACKAGES) + list(extra_packages)
        package_block = "\n".join(f"\\usepackage{{{p}}}" for p in packages)
        return (
            f"\\documentclass{{deepmark}}\n"
            f"{package_block}\n\n"
            f"\\title{{{title}}}\n"
            f"\\author{{{author}}}\n\n"
            f"\\begin{{document}}\n"
            f"\\thispagestyle{{firststyle}}\n"
            f"\\maketitle"
        )

    packages = list(_DEFAULT_ARTICLE_PACKAGES) + list(extra_packages)
    package_block = "\n".join(f"\\usepackage{{{p}}}" for p in packages)
    return (
        f"\\documentclass{{article}}\n"
        f"\\usepackage[margin=2.5cm]{{geometry}}\n"
        f"{package_block}\n\n"
        f"\\title{{{title}}}\n"
        f"\\author{{{author}}}\n"
        f"\\date{{\\today}}\n\n"
        f"\\begin{{document}}\n"
        f"\\maketitle"
    )


_LATEX_SPECIAL = {
    "\\": "\\textbackslash{}",
    "&": "\\&",
    "%": "\\%",
    "$": "\\$",
    "#": "\\#",
    "_": "\\_",
    "{": "\\{",
    "}": "\\}",
    "~": "\\textasciitilde{}",
    "^": "\\textasciicircum{}",
}


def latex_escape(text: str) -> str:
    """``text`` with every LaTeX special character made literal.

    One pass over the characters, so a replacement is never escaped again.
    ``report_charts.plain`` reverses it for the figures.
    """
    return "".join(_LATEX_SPECIAL.get(c, c) for c in text)


def display_attack_name(attack_name: str, split_camel_case: bool = False) -> str:
    """Render an attack class name as human-readable text.

    Drops the trailing ``Attack`` suffix from the base class name.
    A version suffix like ``(aggressive)`` is preserved when present —
    ``expand_attacks`` only includes one when the attack's config is
    multi-version, so the display layer trusts the key as-is.

    When ``split_camel_case`` is ``True`` each interior uppercase
    boundary in the base name is expanded to a space.

    Known acronyms (e.g. ``LPC``) are kept as a single token rather than
    split letter-by-letter.
    """
    # Separate version suffix: "GaussianNoiseAttack (aggressive)" → base + suffix
    version_suffix = ""
    base = attack_name
    if " (" in attack_name and attack_name.endswith(")"):
        idx = attack_name.index(" (")
        base = attack_name[:idx]
        version_suffix = attack_name[idx:]

    # Strip trailing "Attack" from the class name only
    if base.endswith("Attack"):
        base = base[:-6]
    base = latex_escape(base)
    # A version name is the config author's own text -- "very_aggressive"
    # is a natural one -- and every table prints this, so it is escaped
    # like the base rather than trusted.
    version_suffix = latex_escape(version_suffix)

    if not split_camel_case:
        return base + version_suffix

    ACRONYMS = ("LPC",)
    for acronym in ACRONYMS:
        if base == acronym:
            return acronym + version_suffix

    formatted = "".join(
        " " + c if c.isupper() and i > 0 else c
        for i, c in enumerate(base)
    ).strip()
    return formatted + version_suffix


# Column headers for the eight statistics. Every generator renders a
# statistic column through this, so "worst_case" reads the same everywhere.
STAT_HEADERS = {
    "mean": "Mean",
    "std": "Std",
    "median": "Median",
    "p5": "P5",
    "p10": "P10",
    "p95": "P95",
    "p99": "P99",
    "worst_case": "Worst Case",
}

# Metrics reported as a percentage rather than a bare number.
_PERCENT_METRICS = frozenset({"accuracy"})

# Metrics stored as a 0-1 fraction but read as a percentage.
_FRACTION_METRICS = frozenset({"ber"})

# Metrics whose useful resolution is below 0.01.
_FINE_GRAINED_METRICS = frozenset({"stoi", "sii", "ncm"})


def stat_header(statistic: str) -> str:
    """Column header for a statistic name."""
    return STAT_HEADERS.get(statistic, statistic.replace("_", " ").title())


def metric_label(metric: str) -> str:
    """Human-readable label for a metric, with its unit or range."""
    from deepmarkpy.utils.metrics import METRIC_LABELS

    extra = {
        "accuracy": "Accuracy (\\%)",
        "ber": "BER (\\%)",
        "emr": "EMR",
    }
    if metric in extra:
        return extra[metric]
    return METRIC_LABELS.get(metric, metric.upper().replace("_", " "))


def format_metric_cell(metric: str, value) -> str:
    """Render one metric value, in the unit that metric is read in.

    Returns ``"N/A"`` for a missing value, which is deliberately distinct
    from the ``"--"`` a report prints when a statistic was never computed.
    """
    if value is None:
        return "N/A"
    value = float(value)
    if metric in _FRACTION_METRICS:
        return f"{value * 100:.2f}\\%"
    if metric in _PERCENT_METRICS:
        return f"{value:.2f}\\%"
    if metric in _FINE_GRAINED_METRICS:
        return f"{value:.4f}"
    return f"{value:.2f}"


def format_emr_cell(count, total, rate) -> str:
    """Render the exact-match rate as ``n/N (r%)``.

    EMR is a count first: "3 of 20 files came back bit-perfect" is the
    fact, and the percentage is the derived reading of it.
    """
    if rate is None:
        return "N/A"
    if count is not None and total:
        return f"{int(count)}/{int(total)} ({float(rate) * 100:.1f}\\%)"
    return f"{float(rate) * 100:.1f}\\%"


class MetricCaveats:
    """Marks metric cells an attack makes unreliable, and explains why.

    ``get_metric_caveat`` returns a different reason per case, so a single
    hardcoded footnote describes only one of them: a table containing both a
    desynchronization row and SignInversion's SI-SDR would explain the timing
    shift twice and the scale-invariance not at all. Collecting the reasons
    while a table is built gives each its own marker and its own sentence.

    Every report generator that prints per-attack quality metrics uses this,
    so a caveat added to ``attack_groups`` reaches all of them.
    """

    _MARKERS = ("\\dag", "\\ddag", "\\S", "\\P")

    def __init__(self):
        self._reasons = []

    def mark(self, attack_name, metric):
        """Return the superscript for this cell, or '' when it needs none."""
        from deepmarkpy.utils.attack_groups import get_metric_caveat

        reason = get_metric_caveat(attack_name, metric)
        if not reason:
            return ""
        if reason not in self._reasons:
            self._reasons.append(reason)
        return f"\\textsuperscript{{{self._marker(reason)}}}"

    def _marker(self, reason):
        return self._MARKERS[self._reasons.index(reason) % len(self._MARKERS)]

    @property
    def any_flagged(self):
        return bool(self._reasons)

    def footnote(self):
        """One sentence per distinct reason, or '' when nothing was marked."""
        if not self._reasons:
            return ""
        sentences = " ".join(
            f"\\textsuperscript{{{self._marker(r)}}}This metric {r}."
            for r in self._reasons
        )
        return (
            "\n\n{\\noindent\\footnotesize " + sentences
            + " Values are shown for completeness; do not read them as "
            "quality scores.}\n"
        )


# Figures are set narrower than the text block. At full width a chart with
# a handful of bars towers over the table it belongs to, and the section
# reads as a picture with a footnote rather than a table with a picture.
FIGURE_WIDTH = "0.72\\linewidth"


def container_section(rows, label: str = "tab:containers") -> str:
    """A section listing the memory each running service holds.

    Its own section rather than a column anywhere: this is a snapshot of
    the deployment at one moment, not a measurement of the watermarking
    method, and it covers services -- models, dockerized attacks, the
    metric services -- rather than attacks or files.
    """
    if not rows:
        return ""

    body = []
    for kind, name, container, used, limit in rows:
        share = f"{100.0 * used / limit:.0f}\\%" if limit else "--"
        body.append(
            f"    {kind} & {name.replace('_', chr(92) + '_')} & "
            f"{used:.0f} & {limit:.0f} & {share} \\\\"
            if limit else
            f"    {kind} & {name.replace('_', chr(92) + '_')} & "
            f"{used:.0f} & -- & -- \\\\"
        )

    table = build_longtable(
        "llccc",
        "Service & Name & Memory (MiB) & Limit (MiB) & Of limit",
        body,
        "Resident memory of each running container this run used, read "
        "once while the services were warm. This is the whole container -- "
        "weights, Python runtime and web server -- not the size of a model, "
        "and the services load their weights at start, so a container can "
        "never be measured empty. Services that run natively or were not "
        "running are absent rather than reported as zero.",
        label,
    )
    return (
        "\\needspace{5\\baselineskip}\n"
        "\\section{Container Memory}\n\n" + table + "\n\n"
    )


def slugify(label: str) -> str:
    """Filename- and label-safe form of a duration-group label.

    The comparison is spelled out rather than stripped as punctuation.
    Dropping it collapses ``"< 5.0s"`` and ``"> 5.0s"`` onto the same
    slug, so a run with one boundary writes both bins' figures to the
    same filenames and emits the same ``\\label`` twice.
    """
    text = (label.replace("<", " lt ").replace("≥", " ge ")
            .replace(">", " gt "))
    return "".join(
        c if c.isalnum() else "_" for c in text
    ).strip("_").replace("__", "_") or "group"


def duration_label_tex(label: str) -> str:
    """A duration-group label as LaTeX text.

    The comparison signs are typeset in math mode; ``≥`` in particular
    has no text-mode glyph under pdflatex's default input encoding, so a
    raw one stops the compile.
    """
    return (label.replace("<", "$<$").replace(">", "$>$")
            .replace("≥", "$\\geq$"))


def part_heading(label: str, subtitle: str = "") -> str:
    """A ``\\part`` whose sections start again at 1.

    Each duration part is a self-contained report over its own files, so
    its sections are its first, second, third -- not the seventh, eighth
    and ninth of a document the reader is not reading straight through.
    ``\\part`` does not reset the section counter on its own.
    """
    heading = f"{label} ({subtitle})" if subtitle else label
    return (
        f"\\part{{{heading}}}\n"
        "\\setcounter{section}{0}\n\n"
    )


def figure_block(filename: str, caption: str, label: str) -> str:
    """A centred ``figure`` environment, sized relative to the text block.

    Every generator builds its figures through this, so one change of
    ``FIGURE_WIDTH`` resizes every figure.
    """
    return (
        "\\begin{figure}[H]\n"
        "    \\centering\n"
        f"    \\includegraphics[width={FIGURE_WIDTH}]{{{filename}}}\n"
        f"    \\caption{{{caption}}}\n"
        f"    \\label{{{label}}}\n"
        "\\end{figure}\n"
    )


def build_longtable(
    col_spec: str,
    header: str,
    rows: Iterable[str],
    caption: str,
    label: str,
) -> str:
    """Return a full ``longtable`` environment with the shared header/footer.

    Args:
        col_spec: Column specification, e.g. ``"lc"`` or ``"l" + "c"*n``.
        header: Header row cells, already joined with ``&`` and without
            trailing ``\\\\``.
        rows: Iterable of pre-formatted row strings (each ending in
            ``\\\\``). Caller is responsible for LaTeX escaping.
        caption: Table caption text.
        label: Table label (passed as-is to ``\\label{...}``).
    """
    body = "\n".join(rows)
    # Wide tables get smaller font and tighter column spacing so they
    # fit within page margins.
    n_cols = col_spec.count("c") + col_spec.count("l") + col_spec.count("r")
    if n_cols >= 6:
        size_prefix = "{\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
        size_suffix = "\n}"
    elif n_cols >= 5:
        size_prefix = "{\\small\\setlength{\\tabcolsep}{4pt}\n"
        size_suffix = "\n}"
    else:
        size_prefix = ""
        size_suffix = ""

    return (
        f"{size_prefix}"
        f"\\begin{{longtable}}{{{col_spec}}}\n"
        f"    \\caption{{{caption}}}\n"
        f"    \\label{{{label}}} \\\\\n"
        f"    \\toprule\n"
        f"    {header} \\\\\n"
        f"    \\midrule\n"
        f"    \\endfirsthead\n"
        f"    \\toprule\n"
        f"    {header} \\\\\n"
        f"    \\midrule\n"
        f"    \\endhead\n"
        f"    \\bottomrule\n"
        f"    \\endlastfoot\n"
        f"{body}\n"
        f"\\end{{longtable}}"
        f"{size_suffix}"
    )


def compile_latex(report_dir: str, tex_basename: str) -> Optional[str]:
    """Run ``pdflatex`` twice to resolve cross-references, then clean up.

    Args:
        report_dir: Directory containing ``<tex_basename>.tex``.
        tex_basename: File stem without extension (e.g. ``"benchmark_report"``).

    Returns:
        The path to the generated PDF if compilation succeeded, otherwise
        ``None``. Errors are logged but do not raise.
    """
    if not shutil.which("pdflatex"):
        return None

    tex_name = f"{tex_basename}.tex"
    cmd = ["pdflatex", "-interaction=nonstopmode", tex_name]
    try:
        # Two passes so cleveref/longtable references settle.
        subprocess.run(cmd, cwd=report_dir, capture_output=True, timeout=60)
        subprocess.run(cmd, cwd=report_dir, capture_output=True, timeout=60)
    except Exception as e:
        logger.warning(f"PDF compilation failed: {e}")
        return None

    pdf_path = os.path.join(report_dir, f"{tex_basename}.pdf")
    if not os.path.exists(pdf_path):
        return None

    logger.info(f"PDF report generated: {pdf_path}")
    for ext in (".aux", ".log", ".out"):
        aux = os.path.join(report_dir, f"{tex_basename}{ext}")
        if os.path.exists(aux):
            os.remove(aux)
    return pdf_path
