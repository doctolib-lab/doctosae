"""Print one LaTeX table of SAE retrieval results for a paper: original (no-SAE) baseline, then
--metric's value (or its ratio to that baseline) mean +/- 95% CI over seeds for each swept k, one
row-group per embedder (its own k values) and one column group per --dataset, optionally with a
reconstructed-embedding column alongside the sparse-latent one. Best config per k selected by
val metric (see src.utils.sweep_runs).

Reconstructed-embedding scores are read, never computed here (see src.evaluation.eval_ir
--emb-type "reconstructed embeddings"), from <run_dir>/converted/eval_recon.json in-domain and
<run_dir>/ood_eval/<dataset>/eval_recon.json out of domain. A missing file drops that cell to
"--".

Example:
    uv run python -m src.analysis.latex_results_table \\
        --sweep scripts/sweep_cure_topk_bge.sh runs/repo_cure/bge-large-en-v1.5 \\
        --sweep scripts/sweep_cure_topk_gemini.sh runs/repo_cure/gemini-embedding-001 \\
        --metric test_ndcg --datasets CURE Bioinformatics Biology Medical-Sciences --display ratio --recon
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.sweep_runs import (parse_sweep, load_runs, val_metric_from, best_config,
                                  filter_by_hparam, values_of, mean_sem, format_mean_ci,
                                  parse_hparam_value)
from src.utils.baselines import get_dataset_baseline
from src.utils.datasets_constants import dataset_from_sweep, collection_of, genre_of

# Short column headers, overridable with --label.
EMBEDDER_LABELS = {"bge-large-en-v1.5": "BGE", "MedEmbed-large-v0.1": "MedEmbed",
                   "gemini-embedding-001": "Gemini", "embeddingGemma": "EmbeddingGemma"}
# Paper notation for a --group-by key, matching the plotting scripts.
GROUP_LABELS = {"hidden": "$m$", "lambda_": "$\\lambda$", "temperature": "$\\tau$", "k": "$k$"}

IN_DOMAIN_FILE = "eval_recon.json"          # under <run_dir>/converted/
OOD_FILE       = "eval_recon.json"          # under <run_dir>/ood_eval/<dataset>/


def eval_metric_of(test_metric: str) -> str:
    """'test_ndcg' -> 'ndcg', 'test_auc' -> 'auc' -- matches eval_ir.py's own {metric: score} field
    name (see its --output), so the same string reads summary.json, eval.json and get_dataset_baseline."""
    return test_metric.split("test_", 1)[1] if test_metric.startswith("test_") else test_metric


def _read_json(path: Path, field: str) -> float | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())[field]


def _values(seeds: dict, dataset: str, in_domain: str, test_metric: str, eval_metric: str,
           recon: bool) -> list[float]:
    """Per-seed scores for `dataset` on this config: sparse latents (recon=False) or
    reconstructed embeddings (recon=True). None entries (missing eval) are dropped."""
    out = []
    for s in seeds.values():
        if dataset == in_domain:
            v = s[test_metric] if not recon else \
                _read_json(Path(s["_run_dir"]) / "converted" / IN_DOMAIN_FILE, eval_metric)
        else:
            fname = OOD_FILE if recon else "eval.json"
            v = _read_json(Path(s["_run_dir"]) / "ood_eval" / dataset / fname, eval_metric)
        if v is not None:
            out.append(v)
    return out


def _stat(vals: list[float]) -> tuple[float, float, int] | None:
    return mean_sem(vals) if len(vals) >= 2 else None


def _input_dim(data: dict) -> int | None:
    """The embedder's native embedding dim, read from any one run's sae_config.json (input_dim);
    None if data is empty."""
    for seeds in data.values():
        for s in seeds.values():
            cfg = json.loads((Path(s["_run_dir"]) / "sae_config.json").read_text())
            return cfg["input_dim"]
    return None


def _row_label(k: int, input_dim: int | None, dense_label: str) -> str:
    """'k=128', or '{dense_label} (k=1024)' when k equals the embedder's own input_dim."""
    return f"{dense_label} (k={k})" if input_dim is not None and k == input_dim else f"k={k}"


def _best_k(per_k: dict[int, tuple[float, float, int] | None],
            exclude: int | None = None) -> int | None:
    """The k with the highest mean among non-None stats in `per_k`, minus `exclude`."""
    valid = {k: s for k, s in per_k.items() if s is not None and k != exclude}
    return max(valid, key=lambda k: valid[k][0]) if valid else None


def group_header_row(labels: list[str], n: int, n_sub: int, line_after: set[int],
                     boundary: int) -> str:
    """One header row spanning contiguous equal, non-empty `labels`; blank for "", never merged
    across `boundary` (pass -1 to allow merging anywhere)."""
    if not any(labels):
        return ""
    cells, i = [], 0
    while i < n:
        j = i
        while j + 1 < n and j != boundary and labels[j + 1] == labels[i]:
            j += 1
        sep = "|" if j in line_after else ""
        cells.append(f"\\multicolumn{{{(j - i + 1) * n_sub}}}{{c{sep}}}{{{labels[i]}}}")
        i = j + 1
    return " & " + " & ".join(cells) + " \\\\"


def _int_len(value: float, decimals: int) -> int:
    """Digits before the decimal point once `value` is rounded to `decimals`."""
    return len(f"{value:.{decimals}f}".split(".")[0])


def _pad(value: float, decimals: int, width: int | None) -> str:
    """\\phantom{0} per integer digit `value` is short of `width`, so centred cells line up."""
    if not width:
        return ""
    return "\\phantom{0}" * max(0, width - _int_len(value, decimals))


def _shown(stat: tuple[float, float, int] | None, baseline: float | None, display: str,
          pct: bool) -> tuple[float, float] | None:
    """(mean, SEM) as _format prints them, or None where the cell prints '--'."""
    if stat is None:
        return None
    m, sem, _ = stat
    if display == "ratio":
        return (m / baseline, sem / baseline) if baseline else None
    return (m * 100, sem * 100) if pct else (m, sem)


def _format(stat: tuple[float, float, int] | None, baseline: float | None, display: str,
           bold: bool = False, blue: bool = False, pct: bool = False) -> str:
    shown = _shown(stat, baseline, display, pct)
    if shown is None:
        return "--"
    m, sem = shown
    beats_baseline = blue and baseline is not None and stat[0] > baseline
    decimals = 2 if pct and display == "value" else 3
    s = format_mean_ci(m, sem, stat[2], decimals=decimals)
    if bold:
        s = f"\\underline{{{s}}}"
    if beats_baseline:
        s = f"\\textcolor{{blue}}{{{s}}}"
    return s


def _load_embedders(sweeps: list[tuple[Path, Path]], labels: dict[str, str]) -> list[tuple[str, str, dict]]:
    """(label, in_domain_dataset, data) per --sweep, loaded once and reused across all datasets."""
    out = []
    for sweep_sh, runs_dir in sweeps:
        grid = parse_sweep(sweep_sh)
        grid["runs_dir"] = runs_dir
        data = load_runs(runs_dir, grid["swept_keys"], grid["fixed_config"], grid["swept_values"])
        embedder = runs_dir.name
        default_label = EMBEDDER_LABELS.get(embedder, embedder)
        out.append((labels.get(embedder, default_label), embedder, dataset_from_sweep(grid), data))
    return out


def build_table(embedders: list[tuple[str, str, str, dict]], datasets: list[str], test_metric: str,
                ks: list[int] | None, display: str, cutoff: int, recon: bool,
                tabcolsep: str | None = "1pt", filters: dict | None = None,
                exclude_dense: bool = False, blue: bool = False, pct: bool = False) -> str:
    """One row-group per embedder (its own k values), one column group per dataset."""
    val_metric = val_metric_from(test_metric)
    eval_metric = eval_metric_of(test_metric)
    n_sub = 2 if recon else 1
    header_sub = "" if not recon else " & " + " & ".join("sparse & recon" for _ in datasets) + " \\\\"
    n = len(datasets)

    # Rules after the in-domain column and after each collection and genre block; in-domain
    # columns are assumed to be a prefix of --datasets.
    in_domain_datasets = {in_domain for _, _, in_domain, _ in embedders}
    boundary = max((i for i, ds in enumerate(datasets) if ds in in_domain_datasets), default=-1)
    groups = [collection_of(ds) or "" for ds in datasets]
    genres = [genre_of(ds) or "" for ds in datasets]
    line_after = {i for labels in (groups, genres) for i in range(n)
                  if labels[i] and (i == n - 1 or labels[i + 1] != labels[i])}
    if boundary != -1:
        line_after.add(boundary)
    line_after.discard(n - 1)

    # A gap where the genre or collection changes; the in-domain boundary gets no gap.
    gap_after = {i for labels in (groups, genres) for i in range(n)
                 if labels[i] and (i == n - 1 or labels[i + 1] != labels[i])} - {n - 1}
    col_spec = "".join("r" * n_sub + (" " if i in gap_after else "") for i in range(n))
    # Table column of dataset i: the rotated label and the row label take the first two.
    def _cols(i: int) -> tuple[int, int]:
        return 3 + i * n_sub, 2 + (i + 1) * n_sub

    def _header(labels: list[str], italic: bool) -> list[str]:
        """A spanning header row over contiguous equal labels, plus its \\cmidrule line."""
        if not any(labels):
            return []
        cells, rules, i = [], [], 0
        while i < n:
            j = i
            while j + 1 < n and labels[j + 1] == labels[i]:
                j += 1
            text = labels[i]
            if text:
                cells.append(f"\\multicolumn{{{(j - i + 1) * n_sub}}}{{c}}"
                             f"{{{f'\\emph{{{text}}}' if italic else text}}}")
                rules.append(f"\\cmidrule(lr){{{_cols(i)[0]}-{_cols(j)[1]}}}")
            else:
                cells += [""] * ((j - i + 1) * n_sub)
            i = j + 1
        return ["& & " + " & ".join(cells) + " \\\\", "".join(rules)]
    header_top = "& & " + " & ".join(
        f"\\multicolumn{{{n_sub}}}{{c}}{{{ds}}}" for ds in datasets) + " \\\\"

    header_group = _header(groups, italic=False)
    header_genre = _header(genres, italic=True)

    lines = []
    if tabcolsep:
        lines.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}")
    if line_after:
        # 1pt, not 0pt.
        lines += ["\\setlength{\\aboverulesep}{1pt}", "\\setlength{\\belowrulesep}{1pt}"]
    lines.append("\\renewcommand{\\arraystretch}{0.92}")
    lines += [f"\\begin{{tabular}}{{@{{}}c l{col_spec}@{{}}}}", "\\toprule"]
    lines += header_genre + header_group
    lines.append(header_top)
    if header_sub:
        lines.append(header_sub)

    subs = ["sparse", "recon"] if recon else ["sparse"]
    base_decimals = 2 if pct and display == "value" else 4

    # First pass: every embedder's stats.
    per_embedder = []
    for label, embedder, in_domain, data in embedders:
        baselines = {ds: get_dataset_baseline(ds, embedder, eval_metric, k=cutoff) for ds in datasets}
        input_dim = _input_dim(data)
        row_ks = ks or sorted(values_of(data, "k"))
        # stats[ds]["sparse"|"recon"][k] = (mean, sem, n) | None.
        stats = {ds: {"sparse": {}, "recon": {}} for ds in datasets}
        for k in row_ks:
            sub = filter_by_hparam(data, "k", k) if k in values_of(data, "k") else {}
            picked = None
            try:
                picked = best_config(sub, val_metric, fixed=filters) if sub else None
                seeds = sub[picked] if picked else None
            except ValueError:
                picked = seeds = None  # this k has no config matching every --filter pin
            for ds in datasets:
                stats[ds]["sparse"][k] = _stat(_values(seeds, ds, in_domain, test_metric,
                                                       eval_metric, recon=False)) if seeds else None
                if recon:
                    stats[ds]["recon"][k] = _stat(_values(seeds, ds, in_domain, test_metric,
                                                          eval_metric, recon=True)) if seeds else None

        best = {ds: {sub: _best_k(stats[ds][sub], exclude=input_dim if exclude_dense else None)
                    for sub in subs}
               for ds in datasets}
        per_embedder.append((label, baselines, input_dim, row_ks, stats, best))

    def _base_value(baseline: float | None) -> float | None:
        """The baseline as the "original" row prints it, or None where that row prints '--'."""
        if not baseline:
            return None
        return baseline * 100 if pct and display == "value" else baseline

    for block, (label, baselines, input_dim, row_ks, stats, best) in enumerate(per_embedder):
        lines.append("\\midrule" if block == 0 else "\\addlinespace[2pt]")
        rot = (f"\\multirow{{{1 + len(row_ks)}}}{{*}}"
               f"{{\\rotatebox[origin=c]{{90}}{{\\textbf{{{label}}}}}}}")
        base_cells = []
        for ds in datasets:
            base = _base_value(baselines[ds])
            cell = "--" if base is None else f"{base:.{base_decimals}f}"
            base_cells.append(cell + ("" if not recon else " & --"))
        lines.append(f"{rot} & original & " + " & ".join(base_cells) + " \\\\")

        for k in row_ks:
            cells = []
            for ds in datasets:
                baseline = baselines[ds]
                cell = _format(stats[ds]["sparse"][k], baseline, display,
                               bold=k == best[ds]["sparse"], blue=blue, pct=pct)
                if recon:
                    cell += " & " + _format(stats[ds]["recon"][k], baseline, display,
                                            bold=k == best[ds]["recon"], blue=blue, pct=pct)
                cells.append(cell)
            lines.append(f" & {_row_label(k, input_dim, 'dense')} & "
                         + " & ".join(cells) + " \\\\")

    metric_desc = f"{eval_metric.upper()}@{cutoff}" if eval_metric == "ndcg" else eval_metric.upper()
    lines += ["\\bottomrule", "\\end{tabular}"]
    best_desc = "Best sparsity $k$ underlined" if exclude_dense else "Best $k$ underlined"
    # The word itself in the colour it describes.
    blue_desc = " \\textcolor{blue}{Blue}: beats the original embedding." if blue else ""
    value_desc = f"{metric_desc} (\\%)" if pct else metric_desc
    caption = (f"{value_desc if display == 'value' else 'Ratio to no-SAE baseline'}, mean $\\pm$ 95\\% CI over seeds. "
              "$k$: active SAE latents per embedding. "
              "\\emph{dense}: $k$ equals the embedding's native dimension. "
              f"{best_desc}.{blue_desc}")
    return "\n".join([
        "\\begin{table*}[t]", "\\centering", "\\small",
        "\\floatconts", "  {tab:ndcgsparse}",
        f"  {{\\caption{{{caption}}}}}",
        "  {%", "  \\vspace{-15pt}", *("  " + l for l in lines), "  }",
        "\\end{table*}",
    ])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", nargs=2, action="append", required=True, metavar=("SWEEP_SH", "RUNS_DIR"),
                   help="One per embedder: sweep .sh path and its runs_dir. Repeatable.")
    p.add_argument("--metric", required=True,
                   help="Test metric driving both val-selection and which value is shown "
                        "(e.g. test_ndcg, test_auc -- see eval_ir.py's --metric).")
    p.add_argument("--datasets", nargs="+", required=True,
                   help="Any DATASET_PATHS key (see src.utils.datasets_constants), one row-group per dataset "
                        "in a single table. A dataset matching a sweep's own training data is scored "
                        "in-domain, others as OOD.")
    p.add_argument("--k", type=int, nargs="+", default=None,
                   help="Restrict rows to these k values (default: every k found across the sweeps).")
    p.add_argument("--display", choices=["value", "ratio"], default="value",
                   help="--metric's own value (default, e.g. NDCG@cutoff or AUC) or its ratio to the "
                        "no-SAE baseline.")
    p.add_argument("--cutoff", type=int, default=10, help="NDCG cutoff; ignored for non-ndcg metrics (default 10).")
    p.add_argument("--recon", action="store_true",
                   help="Also print a reconstructed-embeddings column next to the sparse-latent one.")
    p.add_argument("--label", nargs=2, action="append", default=[], metavar=("EMBEDDER", "LABEL"),
                   help="Override an embedder's column header (default: its runs_dir name). Repeatable.")
    p.add_argument("--tabcolsep", default="1pt",
                   help="\\setlength{\\tabcolsep}{...} before the tabular (default '1pt', vs LaTeX's own "
                        "6pt) to narrow column padding on a too-wide table. Pass '' to leave it unset.")
    p.add_argument("--filter", nargs=2, metavar=("HPARAM", "VALUE"), action="append", default=[],
                   help="Pin a hyperparameter for every k's config selection, e.g. --filter hidden 3072. "
                        "Repeatable. A k with no config matching every pin shows '--'.")
    p.add_argument("--exclude-dense", action="store_true",
                   help="Drop the dense-SAE row (k equal to the embedding's native dimension) from the "
                        "best-k underline -- it's still shown, just never underlined.")
    p.add_argument("--blue", action="store_true",
                   help="Color a cell's value blue when it beats the original embedding (no-SAE) "
                        "baseline for that dataset. Requires \\usepackage{xcolor} in the document.")
    p.add_argument("--percent", action="store_true",
                   help="Show --display value as a percentage with 2 decimals instead of a 0-1 "
                        "value with 3. Ignored for --display ratio.")
    args = p.parse_args()

    sweeps = [(Path(s), Path(d)) for s, d in args.sweep]
    embedders = _load_embedders(sweeps, dict(args.label))
    filters = {key: parse_hparam_value(value) for key, value in args.filter}
    print(build_table(embedders, args.datasets, args.metric, args.k, args.display, args.cutoff,
                      args.recon, args.tabcolsep, filters, args.exclude_dense,
                      args.blue, args.percent))


if __name__ == "__main__":
    main()
