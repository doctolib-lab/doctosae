"""Print one LaTeX table of every (k, m) cell of a sweep — the appendix counterpart of
latex_results_table.py, which reports a single val-selected config per k instead.

One row group per k, one row per hidden size m inside it, and for each cell the mean +/- 95% CI
over seeds of --metric. The remaining hyperparameters (lambda, temperature, ...) are chosen by the
val metric within the cell. No best-k underline and no baseline colouring.

Example:
    uv run python -m src.analysis.latex_grid_table \\
        --sweep scripts/sweep_cure_topk_bge.sh runs/repo_cure/bge-large-en-v1.5 \\
        --metric test_ndcg --datasets CURE Bioinformatics Biology Medical-Sciences NFCorpus MIRIAD
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.latex_results_table import (GROUP_LABELS, _input_dim, _int_len, _load_embedders,
                                              _pad, _stat, _values, eval_metric_of,
                                              group_header_row)
from src.utils.baselines import get_dataset_baseline
from src.utils.datasets_constants import collection_of, genre_of
from src.utils.sweep_runs import (best_config, filter_by_hparam, format_mean_ci, val_metric_from,
                                  values_of)


def _spanning_row(text: str, n: int, line_after: set[int]) -> str:
    """`text` in the first cell, the rest blank, split at every vertical rule so none is broken."""
    cells, i = [], 0
    while i < n:
        j = i
        while j not in line_after and j < n - 1:
            j += 1
        width = (j - i + 1) + (1 if i == 0 else 0)   # +1 for the row-label column
        cells.append(f"\\multicolumn{{{width}}}{{l{'|' if j in line_after else ''}}}"
                     f"{{{text if i == 0 else ''}}}")
        i = j + 1
    return " & ".join(cells) + " \\\\"


def _cell(stat: tuple[float, float, int] | None, pct: bool = False,
          width: int | None = None) -> str:
    """'mean ± 95% CI', or '--' for a cell with no completed runs."""
    if stat is None:
        return "--"
    m, sem, n = stat
    if pct:
        m, sem = m * 100, sem * 100
    decimals = 2 if pct else 3
    return _pad(m, decimals, width) + format_mean_ci(m, sem, n, decimals=decimals)


def build_grid_table(embedders: list[tuple[str, str, str, dict]], datasets: list[str],
                     test_metric: str, cutoff: int, row_by: str = "hidden",
                     tabcolsep: str | None = "1pt", pct: bool = False) -> str:
    """One row group per k, one row per `row_by` value inside it, one column per dataset."""
    val_metric = val_metric_from(test_metric)
    eval_metric = eval_metric_of(test_metric)
    n = len(datasets)

    # Rules before the first column, after the in-domain one, and after each collection and
    # genre block.
    in_domain_datasets = {in_domain for _, _, in_domain, _ in embedders}
    boundary = max((i for i, ds in enumerate(datasets) if ds in in_domain_datasets), default=-1)
    groups = [collection_of(ds) or "" for ds in datasets]
    genres = [genre_of(ds) or "" for ds in datasets]
    line_after = {i for labels in (groups, genres) for i in range(n)
                  if labels[i] and (i == n - 1 or labels[i + 1] != labels[i])}
    if boundary != -1:
        line_after.add(boundary)
    line_after.discard(n - 1)

    col_spec = "".join("c" + ("|" if i in line_after else "") for i in range(n))
    header_top = " & " + " & ".join(
        f"\\multicolumn{{1}}{{c{'|' if i in line_after else ''}}}{{{ds}}}"
        for i, ds in enumerate(datasets)) + " \\\\"
    header_group = group_header_row(groups, n, 1, line_after, boundary)
    # Genre spans freely across `boundary`, in italic.
    header_genre = group_header_row([f"\\emph{{{g}}}" if g else "" for g in genres],
                                    n, 1, line_after, boundary=-1)

    lines = ["\\small"]   # appendix table: 12 rows x 6 columns needs the smaller size to fit
    if tabcolsep:
        lines.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}")
    if line_after:
        # 1pt, not 0pt.
        lines += ["\\setlength{\\aboverulesep}{1pt}", "\\setlength{\\belowrulesep}{1pt}"]
    lines += [f"\\begin{{tabular}}{{l|{col_spec}}}", "\\toprule"]
    if header_genre:
        lines.append(header_genre)
    if header_group:
        lines.append(header_group)
    lines.append(header_top)

    row_label = GROUP_LABELS.get(row_by, row_by)
    base_decimals = 2 if pct else 4
    cell_decimals = 2 if pct else 3

    # First pass: every cell's stats, for the shared column widths.
    per_embedder = []
    for label, embedder, in_domain, data in embedders:
        baselines = {ds: get_dataset_baseline(ds, embedder, eval_metric, k=cutoff) for ds in datasets}
        input_dim = _input_dim(data)
        blocks = []
        for k in sorted(values_of(data, "k")):
            kdata = filter_by_hparam(data, "k", k)
            k_desc = "dense" if input_dim is not None and k == input_dim else f"$k$={k}"
            rows = []
            for rvalue in values_of(kdata, row_by):
                sub = filter_by_hparam(kdata, row_by, rvalue)
                try:
                    seeds = sub[best_config(sub, val_metric)]
                except ValueError:
                    seeds = None
                rows.append((rvalue, {ds: (_stat(_values(seeds, ds, in_domain, test_metric,
                                                         eval_metric, recon=False))
                                          if seeds else None)
                                     for ds in datasets}))
            blocks.append((k_desc, rows))
        per_embedder.append((label, baselines, blocks))

    widths = {}
    for ds in datasets:
        lens = []
        for _, baselines, blocks in per_embedder:
            if baselines[ds]:
                lens.append(_int_len(baselines[ds] * 100 if pct else baselines[ds], base_decimals))
            lens += [_int_len(stat[0] * 100 if pct else stat[0], cell_decimals)
                     for _, rows in blocks for _, stats in rows if (stat := stats[ds])]
        widths[ds] = max(lens, default=1)

    for label, baselines, blocks in per_embedder:
        lines.append("\\midrule")
        lines.append(_spanning_row(f"\\textbf{{{label}}}", n, line_after))
        base_cells = []
        for ds in datasets:
            base = baselines[ds]
            if not base:
                base_cells.append("--")
                continue
            base = base * 100 if pct else base
            base_cells.append(_pad(base, base_decimals, widths[ds])
                              + f"{base:.{base_decimals}f}")
        lines.append("original & " + " & ".join(base_cells) + " \\\\")

        for k_desc, rows in blocks:
            lines.append(_spanning_row(k_desc, n, line_after))
            for rvalue, stats in rows:
                cells = [_cell(stats[ds], pct=pct, width=widths[ds]) for ds in datasets]
                lines.append(f"{row_label}={rvalue:g} & " + " & ".join(cells) + " \\\\")

    metric_desc = f"{eval_metric.upper()}@{cutoff}" if eval_metric == "ndcg" else eval_metric.upper()
    value_desc = f"{metric_desc} (\\%)" if pct else metric_desc
    lines += ["\\bottomrule", "\\end{tabular}"]
    # GROUP_LABELS values carry their own $...$.
    gl = GROUP_LABELS.get(row_by, row_by)
    caption = (f"{value_desc}, mean $\\pm$ 95\\% CI over seeds, for every "
               f"($k$, {gl}) pair. "
               "\\emph{dense}: $k$ equals the embedding's native dimension. Remaining "
               "hyperparameters are chosen per cell by the best average validation "
               f"{metric_desc}.")
    return "\n".join([
        "\\begin{table*}[t]", "\\centering",
        "\\floatconts", "  {tab:allmk}",
        f"  {{\\caption{{{caption}}}}}",
        "  {%", *("  " + l for l in lines), "  }",
        "\\end{table*}",
    ])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", nargs=2, action="append", required=True, metavar=("SWEEP_SH", "RUNS_DIR"),
                   help="Sweep script and its runs dir. Repeatable, one row group per pair.")
    p.add_argument("--metric", required=True, help="e.g. test_ndcg, test_auc")
    p.add_argument("--datasets", nargs="+", required=True, help="Column order, in-domain first")
    p.add_argument("--cutoff", type=int, default=10, help="NDCG cutoff (default 10)")
    p.add_argument("--row-by", default="hidden",
                   help="Hyperparameter varied within each k group, one row each (default hidden)")
    p.add_argument("--label", nargs=2, action="append", default=[], metavar=("EMBEDDER", "LABEL"),
                   help="Override an embedder's row-group label. Repeatable")
    p.add_argument("--tabcolsep", default="1pt", help="Column padding; '' to leave unset")
    p.add_argument("--percent", action="store_true",
                   help="Show the metric as a percentage with 2 decimals instead of a 0-1 "
                        "value with 3.")
    args = p.parse_args()

    embedders = _load_embedders([(Path(s), Path(d)) for s, d in args.sweep], dict(args.label))
    print(build_grid_table(embedders, args.datasets, args.metric, args.cutoff, args.row_by,
                           args.tabcolsep, args.percent))


if __name__ == "__main__":
    main()
