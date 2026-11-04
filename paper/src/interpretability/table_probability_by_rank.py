"""LaTeX table of the Activation Prediction LLM's mean predicted probability on positive texts, split
by the neuron's activation rank on the text (--rank-bands), next to the mean on negative (inactive)
texts and rho, the Spearman correlation between rank and predicted probability. One row per dataset
and description type, over every scored (neuron, text) pair.

Usage:
    uv run python -m src.interpretability.table_probability_by_rank --pred-dir $RD/interpretation \
        --rank-bands 1-5 6-10 11-15 16-20 --in-domain CURE \
        --ood NFCorpus Bioinformatics Biology Medical-Sciences MIRIAD --output-dir figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from src.interpretability.chart_style import float_table, grouped_tabular, save_text, tex_escape
from src.interpretability.table_classification import DATASET_LABELS, TYPE_LABELS, add_dataset_args, read_pred


def band_stats(d: dict, bands: list[tuple[int, int]]) -> tuple[list[float], float, float]:
    """(mean probability per rank band on positive texts, mean on negative texts, Spearman rho of
    rank against probability on positive texts) of one prediction file."""
    valid = [p for p in d["predictions"] if not p["parse_error"]]
    ranks, scores = map(np.array, zip(*[(p["rank"], p["score"]) for p in valid if p["ground_truth"]]))
    means = [scores[(ranks >= lo) & (ranks <= hi)].mean() if ((ranks >= lo) & (ranks <= hi)).any()
             else float("nan") for lo, hi in bands]
    negative = float(np.mean([p["score"] for p in valid if not p["ground_truth"]]))
    return means, negative, float(spearmanr(ranks, scores).correlation)


def table_probability_by_rank(pred_dir: Path, rank_bands: list[str], in_domain: str, ood: list[str],
                              description_types: list[str], output_dir: str, label: str | None = None,
                              rank_band: tuple[int, int] = (1, 20)) -> str:
    bands = [tuple(int(x) for x in b.split("-")) for b in rank_bands]
    tag = "" if rank_band == (1, 20) else f"_rank{rank_band[0]}-{rank_band[1]}"
    rows = []
    for name in [in_domain, *ood]:
        for t in description_types:
            d = read_pred(pred_dir, t, None if name == in_domain else name, tag)
            if not d:
                continue
            means, negative, rho = band_stats(d, bands)
            rows.append(([tex_escape(DATASET_LABELS.get(name, name)), TYPE_LABELS[t]],
                         [f"{m:.3f}" if m == m else "--" for m in means] + [f"{negative:.3f}", f"${rho:.3f}$"]))
    groups = [("", [f"{lo}--{hi}" for lo, hi in bands] + ["Inactive", "$\\rho$"])]
    caption = ("Mean predicted probability on positive texts, by the neuron's rank on the text; the mean on "
               "negative (inactive) texts; $\\rho$: Spearman correlation between rank and predicted "
               "probability on positive texts.")
    table = float_table(grouped_tabular(["Corpus", "Interpretation"], groups, rows), label or "tab:prob_by_rank",
                        caption)
    return save_text(table, "sae_probability_by_rank_" + in_domain, output_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", required=True, type=Path,
                   help="Directory holding predict_neurons_binary.py's pred_binary_*.json")
    p.add_argument("--rank-bands", nargs="+", required=True, metavar="LO-HI",
                   help="Rank bands of the neuron among the text's active neurons, e.g. 1-5 6-10 11-20")
    p.add_argument("--rank-band", nargs=2, type=int, default=[1, 20], metavar=("K_MIN", "K_MAX"),
                   help="The rank band the evaluation rows were built with (run_interpretability_pipeline.sh "
                        "--rank-band); picks the pred_binary_*_rankLO-HI*.json files unless it is 1 20")
    add_dataset_args(p)
    a = p.parse_args()
    table_probability_by_rank(a.pred_dir, a.rank_bands, a.in_domain, a.ood, a.description_types,
                              a.output_dir, a.label, tuple(a.rank_band))


if __name__ == "__main__":
    main()
