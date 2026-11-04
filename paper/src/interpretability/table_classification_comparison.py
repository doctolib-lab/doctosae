"""LaTeX table comparing the Activation Prediction r of several models (e.g. the SAE, a non-sparse
model, the original embedding dimensions): one row per dataset, one column group per description type,
one column per model. Each --pred-dir holds one model's pred_binary_*.json; r is averaged over the
neurons every description type scored. The best model per dataset and description type is in bold.

Usage:
    uv run python -m src.interpretability.table_classification_comparison \
        --pred-dir $SAE/interpretation $NONSPARSE/interpretation $ORIGINAL/interpretation \
        --model-names SAE "k=m=d" Original --in-domain CURE \
        --ood NFCorpus Bioinformatics Biology Medical-Sciences MIRIAD --output-dir figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.interpretability.chart_style import float_table, grouped_tabular, save_text, tex_escape
from src.interpretability.table_classification import (DATASET_LABELS, TYPE_LABELS, add_dataset_args,
                                                        agg_pred_by_dataset)


def table_classification_comparison(pred_dirs: list[Path], model_names: list[str], in_domain: str,
                                    ood: list[str], description_types: list[str], output_dir: str,
                                    label: str | None = None) -> str:
    per_model = [agg_pred_by_dataset(d, in_domain, ood, description_types) for d in pred_dirs]
    rows = []
    for i, name in enumerate([in_domain, *ood]):
        cells = []
        for t in description_types:
            vals = [m[i][1][t]["pearson"] if m[i][1][t] else None for m in per_model]
            best = max((round(v, 3) for v in vals if v is not None), default=None)
            cells += ["--" if v is None else f"\\textbf{{{v:.3f}}}" if round(v, 3) == best else f"{v:.3f}"
                      for v in vals]
        rows.append(([tex_escape(DATASET_LABELS.get(name, name))], cells))
    groups = [(TYPE_LABELS[t], [tex_escape(n) for n in model_names]) for t in description_types]
    caption = ("Activation Prediction task: $r$ is the Pearson correlation of the predicted probability "
               "with the ground truth (positive/negative), averaged over the neurons every interpretation "
               "type scored. Per dataset and interpretation type, best $r$ in bold.")
    table = float_table(grouped_tabular(["Dataset"], groups, rows), label or "tab:neuronclassif_comparison", caption)
    return save_text(table, "sae_neuron_classification_comparison_" + in_domain, output_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", nargs="+", required=True, type=Path,
                   help="One directory per model, holding pred_binary_*.json")
    p.add_argument("--model-names", nargs="+", required=True, help="Column label for each --pred-dir")
    add_dataset_args(p)
    a = p.parse_args()
    if len(a.pred_dir) != len(a.model_names):
        p.error("--pred-dir and --model-names need the same number of entries")
    table_classification_comparison(a.pred_dir, a.model_names, a.in_domain, a.ood, a.description_types,
                                    a.output_dir, a.label)


if __name__ == "__main__":
    main()
