"""LaTeX table of the Activation Prediction r for several Activation Prediction LLMs: one row per
dataset, one column group per description type, one column per LLM. Each --pred-dir holds one LLM's
predict_neurons_binary.py output, averaged over the neurons every description type scored.

Usage:
    uv run python -m src.interpretability.table_by_activation_prediction_llm \
        --pred-dir $RD/interpretation $RD/interpretation/judge_gemma-3-4b $RD/interpretation/judge_qwen2.5-3b \
        --llm-names MedGemma Gemma3-4B Qwen2.5-3B --in-domain CURE \
        --ood NFCorpus Bioinformatics Biology Medical-Sciences MIRIAD --output-dir figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.interpretability.chart_style import float_table, grouped_tabular, save_text, tex_escape
from src.interpretability.table_classification import (DATASET_LABELS, TYPE_LABELS, add_dataset_args,
                                                        agg_pred_by_dataset)


def table_by_activation_prediction_llm(pred_dirs: list[Path], llm_names: list[str], in_domain: str,
                                       ood: list[str], description_types: list[str], output_dir: str,
                                       label: str | None = None) -> str:
    per_llm = [agg_pred_by_dataset(d, in_domain, ood, description_types) for d in pred_dirs]
    rows = []
    for i, name in enumerate([in_domain, *ood]):
        cells = [f"{per_llm[j][i][1][t]['pearson']:.3f}" if per_llm[j][i][1][t] else "--"
                 for t in description_types for j in range(len(pred_dirs))]
        rows.append(([tex_escape(DATASET_LABELS.get(name, name))], cells))
    groups = [(TYPE_LABELS[t], [tex_escape(n) for n in llm_names]) for t in description_types]
    caption = ("Activation Prediction task: $r$ is the Pearson correlation of the predicted probability "
               "with the ground truth (positive/negative), averaged over the neurons every interpretation "
               "type scored, for each Activation Prediction LLM.")
    table = float_table(grouped_tabular(["Dataset"], groups, rows), label or "tab:neuronclassif_llms", caption)
    return save_text(table, "sae_neuron_classification_by_llm_" + in_domain, output_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", nargs="+", required=True, type=Path,
                   help="One directory per Activation Prediction LLM, holding pred_binary_*.json")
    p.add_argument("--llm-names", nargs="+", required=True, help="Column label for each --pred-dir")
    add_dataset_args(p)
    a = p.parse_args()
    if len(a.pred_dir) != len(a.llm_names):
        p.error("--pred-dir and --llm-names need the same number of entries")
    table_by_activation_prediction_llm(a.pred_dir, a.llm_names, a.in_domain, a.ood, a.description_types,
                                       a.output_dir, a.label)


if __name__ == "__main__":
    main()
