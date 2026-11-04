"""LaTeX table of the Activation Prediction LLM's per-neuron AUROC (predicted probability against the
positive/negative ground truth), averaged over the neurons every description type scored: one row per
dataset, one column per description type. Prints, per dataset and type, the share of neurons at
AUROC = 1 and at r = 1, and the correlation across neurons between AUROC and Pearson r.

Usage:
    uv run python -m src.interpretability.table_auroc --pred-dir $RD/interpretation --in-domain CURE \
        --ood NFCorpus Bioinformatics Biology Medical-Sciences MIRIAD --output-dir figures
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.interpretability.chart_style import float_table, grouped_tabular, save_text, tex_escape
from src.interpretability.table_classification import (DATASET_LABELS, TYPE_LABELS, add_dataset_args,
                                                        load_dataset_preds)


def neuron_aurocs(d: dict, keep: set[str]) -> dict[str, float]:
    """{neuron: AUROC} over the `keep` neurons scored with both a positive and a negative text."""
    by_neuron = defaultdict(lambda: ([], []))
    for p in d["predictions"]:
        if not p["parse_error"] and str(p["neuron"]) in keep:
            by_neuron[str(p["neuron"])][0].append(bool(p["ground_truth"]))
            by_neuron[str(p["neuron"])][1].append(p["score"])
    return {n: float(roc_auc_score(y, s)) for n, (y, s) in by_neuron.items() if 0 < sum(y) < len(y)}


def table_auroc(pred_dir: Path, in_domain: str, ood: list[str], description_types: list[str],
                output_dir: str, label: str | None = None) -> str:
    rows = []
    for name in [in_domain, *ood]:
        raw, keep = load_dataset_preds(pred_dir, name, in_domain, description_types)
        cells = []
        for t in description_types:
            if not raw[t]:
                cells.append("--")
                continue
            auc = neuron_aurocs(raw[t], keep)
            r = {n: v["pearson"] for n, v in raw[t]["metrics"].items() if n in auc}
            a, rr = np.array([auc[n] for n in auc]), np.array([r[n] for n in auc])
            print(f"{name} {t}: {len(a)} neurons | AUROC=1: {np.mean(a == 1):.1%} | r=1: {np.mean(rr == 1):.1%} "
                  f"| corr(AUROC, r): {np.corrcoef(a, rr)[0, 1]:.3f}")
            cells.append(f"{a.mean():.3f}")
        rows.append(([tex_escape(DATASET_LABELS.get(name, name))], cells))
    groups = [("", [TYPE_LABELS[t] for t in description_types])]
    caption = ("AUROC of the Activation Prediction LLM's predicted probability against the ground truth "
               "(positive/negative), averaged over the neurons every interpretation type scored.")
    table = float_table(grouped_tabular(["Dataset"], groups, rows), label or "tab:neuron_auroc", caption)
    return save_text(table, "sae_neuron_auroc_" + in_domain, output_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", required=True, type=Path,
                   help="Directory holding predict_neurons_binary.py's pred_binary_*.json")
    add_dataset_args(p)
    a = p.parse_args()
    table_auroc(a.pred_dir, a.in_domain, a.ood, a.description_types, a.output_dir, a.label)


if __name__ == "__main__":
    main()
