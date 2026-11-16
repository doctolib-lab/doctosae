"""ID-vs-OOD neuron firing-rate reuse chart.

Per-neuron firing rate in-domain (x) vs out-of-domain (y), one panel per OOD corpus, Spearman
rho per panel.

Inputs are convert_embeddings.py '*_with_sae.pkl' files: the same checkpoint run against the
in-domain corpus and each OOD corpus. The OOD set is ood_datasets_for(<in-domain name>) minus
--exclude, resolved against --ood-dir/{name}/corpus*_with_sae.pkl.

--queries-file adds a second chart: in-domain corpus firing rate (x) vs in-domain query firing
rate (y).

Usage:
    RUN=runs/repo_cure/bge-large-en-v1.5/cure_k128_l1e-5_a0.1_lr5e-4_temp0.05_s42_h3072
    python -m src.interpretability.plot_firing_rate_id_vs_ood \
        --in-domain CURE:$RUN/converted/corpus_train_with_sae.pkl \
        --ood-dir $RUN/ood_eval --exclude MIRIAD \
        --queries-file $RUN/converted/queries_test_en_with_sae.pkl \
        --model-name cure_k128_l1e-5_a0.1_lr5e-4_temp0.05_s42_h3072 --output-dir $RUN/charts
"""
from __future__ import annotations

import argparse
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from src.interpretability.chart_style import apply_style, add_model_footer, thin_grid, save, INK, MUTED, PALETTE
from src.utils.datasets_constants import ood_corpus_files
from src.utils.interpretability_utils import load_sae_codes, neuron_firing_stats


def _parse_named_file(spec: str) -> tuple[str, Path]:
    if ":" not in spec:
        raise argparse.ArgumentTypeError(f"expected NAME:PATH, got {spec!r}")
    name, path = spec.split(":", 1)
    return name, Path(path)


def _scatter_panel(ax, x_all, y_all, name: str, color: str, x_label: str, y_label: str) -> None:
    """One log-log firing-rate scatter panel, shared by the OOD grid and the corpus-vs-queries
    chart."""
    both = (x_all > 0) & (y_all > 0)
    smallest = min([r[r > 0].min() for r in (x_all, y_all) if (r > 0).any()], default=1e-5)
    floor = smallest / 2  # below any real firing rate; keeps log axis finite
    x, y = np.clip(x_all[both], floor, None), np.clip(y_all[both], floor, None)
    ax.scatter(x, y, s=4, color=color, alpha=0.18, edgecolors="none", rasterized=True)
    lim = [floor, 1.0]
    ax.plot(lim, lim, "-", color=INK, lw=0.8, alpha=0.5, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    rho = spearmanr(x_all[both], y_all[both]).correlation
    x_only = int(((x_all > 0) & (y_all == 0)).sum())
    y_only = int(((x_all == 0) & (y_all > 0)).sum())
    ax.set_title(name, fontsize=12, color=color)
    ax.text(0.04, 0.94, f"ρ = {rho:.2f}   (n={both.sum()} co active)",
            transform=ax.transAxes, fontsize=9.5, va="top", color=INK)
    ax.text(0.04, 0.05, f"{x_label} only live: {x_only}\n{y_label} only live: {y_only}",
            transform=ax.transAxes, fontsize=8, va="bottom", color=MUTED)
    thin_grid(ax, "both")


def plot_firing_rate_id_vs_ood(in_domain: tuple[str, Path], ood: list[tuple[str, Path]],
                               output_dir: str, hidden: int | None = None, model_name: str | None = None,
                               dated: bool = False) -> str:
    in_name, in_path = in_domain
    fr_id = neuron_firing_stats(load_sae_codes(in_path, hidden)[0])
    fr_ood = {name: neuron_firing_stats(load_sae_codes(path, hidden)[0]) for name, path in ood}

    apply_style()
    ncols = 2
    nrows = -(-len(ood) // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax, (name, _), color in zip(axes, ood, cycle(PALETTE)):
        _scatter_panel(ax, fr_id, fr_ood[name], name, color, "ID", "OOD")
    for ax in axes[len(ood):]:
        ax.axis("off")
    for ax in axes[max(0, len(ood) - ncols):len(ood)]:
        ax.set_xlabel(f"firing rate, {in_name} (in domain)")
    for ax in axes[0:len(ood):ncols]:
        ax.set_ylabel("firing rate, OOD corpus")
    fig.suptitle(f"{in_name}: per-neuron firing rate, in-domain vs out-of-domain",
                 fontsize=13, weight="bold", y=0.995)
    add_model_footer(fig, model_name)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    stem = f"sae_feature_reuse_scatter_{model_name}" if model_name is not None else "sae_feature_reuse_scatter"
    return save(fig, stem, output_dir, dated=dated)


def plot_firing_rate_corpus_vs_queries(in_domain: tuple[str, Path], queries_file: Path,
                                       output_dir: str, hidden: int | None = None,
                                       model_name: str | None = None, dated: bool = False) -> str:
    """Single-panel chart: corpus firing rate (x) vs query firing rate (y)."""
    in_name, in_path = in_domain
    fr_id = neuron_firing_stats(load_sae_codes(in_path, hidden)[0])
    fr_q = neuron_firing_stats(load_sae_codes(queries_file, hidden)[0])

    apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    _scatter_panel(ax, fr_id, fr_q, "corpus vs queries", PALETTE[0], "corpus", "queries")
    ax.set_xlabel(f"firing rate, {in_name} (corpus)")
    ax.set_ylabel(f"firing rate, {in_name} (queries)")
    fig.suptitle(f"{in_name}: per-neuron firing rate, corpus vs queries",
                fontsize=13, weight="bold", y=0.99)
    add_model_footer(fig, model_name)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    stem = (f"sae_feature_reuse_scatter_{model_name}_queries" if model_name is not None
           else "sae_feature_reuse_scatter_queries")
    return save(fig, stem, output_dir, dated=dated)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-domain", required=True, type=_parse_named_file,
                   help="NAME:PATH — in-domain corpus name (a src.utils.datasets_constants "
                        "name, e.g. CURE) and its '*_with_sae.pkl'")
    p.add_argument("--ood-dir", required=True, type=Path,
                   help="Directory holding {name}/corpus*_with_sae.pkl for each OOD dataset in "
                        "ood_datasets_for(<in-domain name>)")
    p.add_argument("--exclude", nargs="+", default=[],
                   help="Dataset names to drop from the OOD list")
    p.add_argument("--hidden", type=int, default=None,
                   help="Override the SAE hidden size for sparse sae-embeddings (default: read "
                        "from the pkl's attrs['sae_hidden'], written by convert_embeddings.py)")
    p.add_argument("--queries-file", type=Path, default=None,
                   help="In-domain queries' '*_with_sae.pkl' — generates a second, separate "
                        "corpus-vs-queries firing-rate chart")
    p.add_argument("--model-name", default=None,
                   help="Full checkpoint identifier (e.g. the run's tag) for the title/filename "
                        "only — omit rather than passing a partial identifier like just k")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dated", action="store_true", help="Append today's date to the output filename")
    args = p.parse_args()

    in_name = args.in_domain[0]
    ood = ood_corpus_files(args.ood_dir, in_name, args.exclude)

    print(plot_firing_rate_id_vs_ood(args.in_domain, ood, args.output_dir,
                                      args.hidden, args.model_name, args.dated))
    if args.queries_file is not None:
        print(plot_firing_rate_corpus_vs_queries(args.in_domain, args.queries_file, args.output_dir,
                                                  args.hidden, args.model_name, args.dated))


if __name__ == "__main__":
    main()
