"""Shared paper-style plotting helpers for the SAE interpretability figure package.

Sans-serif, white background, thin rules. Every figure script imports `apply_style()`,
`save()` and the shared palette/axis helpers from here.
"""
from __future__ import annotations

import os
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

# ── Palette ─────────────────────────────────────────────────────────────────
INK = "#111111"
MUTED = "#555555"
GRID = "#e6e6e6"
DOWN = "#8b1a1a"  # red — trend/decile lines
ACCENT = "#1f4e79"  # in-domain primary

# Categorical palette cycled by index, for charts over a caller-supplied list of names.
PALETTE = ["#c0392b", "#2e86c1", "#e08e0b", "#7d3c98", "#1a5c30", "#8b1a1a", "#2563eb", "#94a3b8"]


def distinct_colors(n: int) -> list[str]:
    """`n` distinct colours: PALETTE while long enough, then tab10/tab20, then sampled turbo.
    Never repeats, unlike cycling PALETTE by `i % len(PALETTE)`."""
    if n <= len(PALETTE):
        return PALETTE[:n]
    if n <= 20:
        cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
        return [to_hex(c) for c in cmap.colors[:n]]
    turbo = plt.get_cmap("turbo")
    return [to_hex(turbo(i / (n - 1))) for i in range(n)]


def apply_style() -> None:
    """Set global Matplotlib rcParams for the shared publication look. Call before drawing."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.titlepad": 10,
    })


def thin_grid(ax, axis: str = "y") -> None:
    """Draw a light, print-friendly reference grid on the given axes, pushed behind the data."""
    ax.grid(axis=axis, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def add_model_footer(fig, model_name: str | None) -> None:
    """Small muted footer, bottom-right, with the full checkpoint identifier. No-op if model_name
    isn't given."""
    if model_name:
        fig.text(0.99, 0.005, model_name, ha="right", va="bottom", fontsize=7, color=MUTED)


def _dated_path(stem: str, output_dir: str, ext: str, dated: bool, date_str: str | None) -> str:
    os.makedirs(output_dir, exist_ok=True)
    stamp = date_str or date.today().isoformat()
    name = f"{stem}_{stamp}.{ext}" if dated else f"{stem}.{ext}"
    return os.path.join(output_dir, name)


def save(fig, stem: str, output_dir: str, *, dated: bool = True, date_str: str | None = None) -> str:
    """Save fig as a 200dpi PNG into output_dir (stem_<date>.png if dated) and return its path."""
    path = _dated_path(stem, output_dir, "png", dated, date_str)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {path}")
    return path


def save_text(text: str, stem: str, output_dir: str, ext: str = "tex", *, dated: bool = True,
              date_str: str | None = None) -> str:
    """Write `text` into output_dir (stem_<date>.<ext> if dated) and return its path — save()'s
    naming convention for a non-figure artifact."""
    path = _dated_path(stem, output_dir, ext, dated, date_str)
    with open(path, "w") as f:
        f.write(text)
    print(f"  ✓ {path}")
    return path


_TEX_ESCAPES = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
                "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
                "\\": r"\textbackslash{}"}


def tex_escape(s: str) -> str:
    """A string with LaTeX's special characters escaped."""
    return "".join(_TEX_ESCAPES.get(c, c) for c in s)


def float_table(tabular: list[str], label: str, caption: str) -> str:
    """The `\\floatconts` table float around the `tabular` lines."""
    return "\n".join(["\\begin{table}[t]", "\\centering", "\\small", "\\floatconts", f"  {{{label}}}",
                      f"  {{\\caption{{{caption}}}}}", "  {%", *("  " + l for l in tabular), "  }",
                      "\\end{table}"])


def grouped_tabular(row_headers: list[str], groups: list[tuple[str, list[str]]],
                    rows: list[tuple[list[str], list[str]]]) -> list[str]:
    """`tabular` lines: the `row_headers` label columns, then one column group per (group label,
    column headers) of `groups`. Each row is (row labels, cells in column order). The group label
    row is dropped when every group label is empty. All strings are LaTeX source."""
    n = len(row_headers)
    spec = "l" * n + "|" + "|".join("r" * len(cols) for _, cols in groups)
    lines = ["\\setlength{\\tabcolsep}{2pt}", "\\setlength{\\aboverulesep}{1pt}",
             "\\setlength{\\belowrulesep}{1pt}", f"\\begin{{tabular}}{{@{{}}{spec}@{{}}}}", "\\toprule"]
    if any(label for label, _ in groups):
        lines.append(" & ".join([""] * n + [
            f"\\multicolumn{{{len(cols)}}}{{c{'|' if i < len(groups) - 1 else ''}}}{{{label}}}"
            for i, (label, cols) in enumerate(groups)]) + " \\\\")
    lines.append(" & ".join(row_headers + [h for _, cols in groups for h in cols]) + " \\\\")
    lines.append("\\midrule")
    lines += [" & ".join(labels + cells) + " \\\\" for labels, cells in rows]
    return lines + ["\\bottomrule", "\\end{tabular}"]
