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
        f"{package_block}\n"
        f"\\geometry{{margin=2.5cm}}\n\n"
        f"\\title{{{title}}}\n"
        f"\\author{{{author}}}\n"
        f"\\date{{\\today}}\n\n"
        f"\\begin{{document}}\n"
        f"\\maketitle"
    )


def display_attack_name(attack_name: str, split_camel_case: bool = False) -> str:
    """Render an attack class name as human-readable text.

    Drops the trailing ``Attack`` suffix. When ``split_camel_case`` is
    ``True`` each interior uppercase boundary is expanded to a space,
    matching the style used by the basic benchmark report.
    """
    stripped = attack_name.replace("Attack", "").strip()
    if not split_camel_case:
        return stripped
    return "".join(
        " " + c if c.isupper() and i > 0 else c
        for i, c in enumerate(stripped)
    ).strip()


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
    return (
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
