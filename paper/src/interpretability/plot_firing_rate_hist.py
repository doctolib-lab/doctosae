"""Per-unit firing-rate histogram and average active units per document, for one embedding column.

--emb-col "sae embeddings" reads the SAE codes; "embeddings" reads the original dense embeddings
of the same '*_with_sae.pkl'. A unit is active on a document when its value is > 0.

Usage:
    RUN=runs/repo_cure/bge-large-en-v1.5/cure_k64_l1e-5_a0.1_lr5e-4_temp0.05_s42_h3072
    python -m src.interpretability.plot_firing_rate_hist \
        --sae-corpus-file $RUN/converted/corpus_train_with_sae.pkl --emb-col embeddings \
        --model-name cure_k64_l1e-5_a0.1_lr5e-4_temp0.05_s42_h3072 --output-dir $RUN/charts
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.interpretability.chart_style import ACCENT, INK, MUTED, add_model_footer, apply_style, save, thin_grid
from src.interpretability.plot_2d_projection import SPARSE_COL, load_space
from src.utils.interpretability_utils import _fired_and_n_fired, neuron_firing_stats

EMB_COLS = {SPARSE_COL: "SAE", "embeddings": "original"}


def active_per_doc(Z) -> np.ndarray:
    """Number of units > 0 on each document."""
    return np.asarray(_fired_and_n_fired(Z)[0].sum(axis=1)).ravel()


def plot_firing_rate_hist(path: Path, emb_col: str, output_dir: str, hidden: int | None = None,
                          model_name: str | None = None, dated: bool = False) -> str:
    Z = load_space(pd.read_pickle(path), emb_col, hidden)
    fr, n_act = neuron_firing_stats(Z), active_per_doc(Z)
    label = EMB_COLS[emb_col]
    print(f"[{label}] {Z.shape[0]} docs x {Z.shape[1]} units | active units/doc: "
          f"mean={n_act.mean():.1f} std={n_act.std():.1f} | zero-rate units: {(fr == 0).sum()}")

    apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    pos = fr[fr > 0]
    ax.hist(pos, bins=np.logspace(np.log10(pos.min()), 0, 50), color=ACCENT, alpha=0.85, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("firing rate (fraction of documents with value > 0)")
    ax.set_ylabel("number of units")
    ax.set_title(f"{label}: per-unit firing rate", color=INK)
    ax.text(0.03, 0.95, f"active units/doc: {n_act.mean():.1f} ± {n_act.std():.1f} of {Z.shape[1]}"
            f"\nzero-rate units: {(fr == 0).sum()}", transform=ax.transAxes, fontsize=9,
            va="top", color=MUTED)
    thin_grid(ax)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    add_model_footer(fig, model_name)
    stem = "_".join(s for s in ["firing_rate_hist", label, model_name] if s)
    return save(fig, stem, output_dir, dated=dated)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sae-corpus-file", required=True, type=Path, help="A convert_embeddings.py '*_with_sae.pkl'")
    p.add_argument("--emb-col", required=True, choices=list(EMB_COLS))
    p.add_argument("--hidden", type=int, default=None,
                   help="SAE hidden size override (default: the pkl's attrs['sae_hidden'])")
    p.add_argument("--model-name", default=None, help="Full checkpoint identifier, for the footer/filename")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dated", action="store_true", help="Append today's date to the output filename")
    args = p.parse_args()
    plot_firing_rate_hist(args.sae_corpus_file, args.emb_col, args.output_dir, args.hidden,
                          args.model_name, args.dated)


if __name__ == "__main__":
    main()
