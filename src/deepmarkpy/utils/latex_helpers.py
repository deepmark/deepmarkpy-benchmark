"""LaTeX building blocks shared by the five report generators.

Preamble, attack-name and metric formatting, tables, caveat marks, figure
blocks and pdflatex compilation.
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
    """``text`` with every LaTeX special character escaped, in one pass.

    ``report_charts.plain`` reverses it for the figures.
    """
    return "".join(_LATEX_SPECIAL.get(c, c) for c in text)


def display_attack_name(attack_name: str, split_camel_case: bool = False) -> str:
    """Render an attack class name as human-readable text.

    Drops the trailing ``Attack`` suffix from the base class name and keeps
    a version suffix such as ``(aggressive)``; both parts are LaTeX-escaped.
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


# Metrics whose useful resolution is below 0.01.
_FINE_GRAINED_METRICS = frozenset({"stoi", "sii", "ncm"})


def stat_header(statistic: str) -> str:
    """Column header for a statistic name, e.g. ``Worst Case``."""
    return statistic.replace("_", " ").title()


def metric_label(metric: str) -> str:
    """Human-readable label for a metric, with its unit or range."""
    from deepmarkpy.utils.metrics import METRIC_LABELS

    extra = {
        "accuracy": "Accuracy (\\%)",
        "ber": "BER (\\%)",
    }
    if metric in extra:
        return extra[metric]
    return METRIC_LABELS.get(metric, metric.upper().replace("_", " "))


def compact_header(metric: str, statistic: str) -> str:
    """A metric's single-column header; names the statistic unless it is the mean."""
    if statistic == "mean":
        return metric_label(metric)
    return f"{metric_label(metric)} [{stat_header(statistic)}]"


def format_metric_cell(metric: str, value, missing: str = "N/A") -> str:
    """Render one metric value in the unit that metric is read in, or ``missing``."""
    if value is None:
        return missing
    value = float(value)
    # BER is stored as a 0-1 fraction.
    if metric == "ber":
        return f"{value * 100:.2f}\\%"
    if metric == "accuracy":
        return f"{value:.2f}\\%"
    if metric in _FINE_GRAINED_METRICS:
        return f"{value:.4f}"
    return f"{value:.2f}"


def format_emr_cell(count, total, rate) -> str:
    """The exact-match rate as ``n/N (r%)``, or the rate alone without a count."""
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


# Width of every report figure, relative to the text block.
FIGURE_WIDTH = "0.72\\linewidth"


def crop_note(crop_before_attack) -> str:
    """The red caveat that every attack ran on cropped audio, or ''."""
    if crop_before_attack is None:
        return ""
    return (
        f"\\textcolor{{red}}{{A crop of {crop_before_attack:.1f}\\% was "
        f"applied to the beginning of the watermarked audio prior to each "
        f"attack. The original (reference) audio was cropped identically, so "
        f"quality metrics compare cropped original vs.\\ cropped attacked "
        f"audio, and BER is measured by detecting the watermark from the "
        f"cropped attacked signal.}}"
    )


def embedding_cost_line(timings, statistics) -> str:
    """The run's "Embedding cost per file" sentence, or '' without a value.

    ``timings`` maps each of ``statistics`` to seconds.
    """
    parts = [
        f"{float(timings[s]):.4f}\\,s ({stat_header(s).lower()})"
        for s in statistics if timings.get(s) is not None
    ]
    if not parts:
        return ""
    return (
        f"\\noindent\\textbf{{Embedding cost per file:}} {', '.join(parts)}\n"
        "\\\\{\\footnotesize Measured once per file, before any attack, so it "
        "does not vary by attack. Like every timing it depends on this machine "
        "and does not reproduce across runs.}\n\n"
    )


def container_section(rows, label: str = "tab:containers") -> str:
    """A section listing the memory each running service held, or '' without rows."""
    if not rows:
        return ""

    body = []
    for kind, name, _, used, limit in rows:
        of_limit = (f"{limit:.0f} & {100.0 * used / limit:.0f}\\%" if limit
                    else "-- & --")
        name = name.replace("_", "\\_")
        body.append(f"    {kind} & {name} & {used:.0f} & {of_limit} \\\\")

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

    Comparison signs are spelled out, so ``"< 5.0s"`` and ``"> 5.0s"`` differ.
    """
    text = (label.replace("<", " lt ").replace("≥", " ge ")
            .replace(">", " gt "))
    return "".join(
        c if c.isalnum() else "_" for c in text
    ).strip("_").replace("__", "_") or "group"


def duration_label_tex(label: str) -> str:
    """A duration-group label as LaTeX text, its comparison signs in math mode.

    ``≥`` has no text-mode glyph under pdflatex's default input encoding.
    """
    return (label.replace("<", "$<$").replace(">", "$>$")
            .replace("≥", "$\\geq$"))


def part_heading(label: str, subtitle: str = "") -> str:
    """A ``\\part`` heading that restarts section numbering at 1."""
    heading = f"{label} ({subtitle})" if subtitle else label
    return (
        f"\\part{{{heading}}}\n"
        "\\setcounter{section}{0}\n\n"
    )


def figure_block(filename: str, caption: str, label: str) -> str:
    """A centred ``figure`` environment ``FIGURE_WIDTH`` wide."""
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


def grid_table(row_header, headers, rows, caption, label) -> str:
    """A longtable with a label column and one centred column per header.

    ``rows`` holds ``(name, cells)`` pairs, or raw lines such as
    ``"    \\midrule"`` that are passed through.
    """
    return build_longtable(
        "l" + "c" * len(headers),
        " & ".join([row_header, *headers]),
        [row if isinstance(row, str)
         else "    " + " & ".join([row[0], *row[1]]) + " \\\\"
         for row in rows],
        caption, label,
    )


# Appended to every timing table's caption.
TIMING_NOTE = (
    " These depend on the machine and on whether the plugin ran natively or "
    "over HTTP, so they do not reproduce across runs the way the "
    "measurements above do."
)


def efficiency_tables(row_header, rows, metrics, statistics_for, subject,
                      label, note=TIMING_NOTE, value_of=None) -> str:
    """Timing tables: one per metric with several statistics, one for the rest.

    ``rows`` holds ``(name, record)`` pairs. A record maps each metric to
    ``{statistic: seconds}``, unless ``value_of(record, metric, statistic)``
    reads it. Captions read ``"<metric> <subject>.<note>"`` and
    ``"Processing time <subject>.<note>"``.
    """
    read = value_of or (lambda record, m, s: (record.get(m) or {}).get(s))

    def seconds(value):
        return "--" if value is None else f"{float(value):.4f}"

    def table(columns, headers, caption, table_label):
        return grid_table(row_header, headers, [
            (name, [seconds(read(record, m, s)) for m, s in columns])
            for name, record in rows
        ], caption, table_label)

    tables = []
    shared = []
    for metric in metrics:
        statistics = statistics_for(metric)
        if len(statistics) > 1:
            tables.append(table(
                [(metric, s) for s in statistics],
                [stat_header(s) for s in statistics],
                f"{metric_label(metric)} {subject}.{note}", f"{label}_{metric}",
            ))
        elif statistics:
            shared.append((metric, statistics[0]))

    if shared:
        tables.append(table(
            shared, [metric_label(m) for m, _ in shared],
            f"Processing time {subject}.{note}", label,
        ))
    return "\n\n".join(tables)


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
