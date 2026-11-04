"""LaTeX table of the agreement between Activation Prediction LLMs: the Pearson correlation of two
LLMs' predicted probabilities over the (neuron, text) pairs both scored. One row per dataset, one
column group per description type, one column per pair of LLMs. Prints the min and max per type.

Usage:
    uv run python -m src.interpretability.table_llm_agreement \
        --pred-dir $RD/interpretation $RD/interpretation/judge_gemma-3-4b $RD/interpretation/judge_qwen2.5-3b \
        --llm-names MedGemma Gemma3-4B Qwen2.5-3B --in-domain CURE \
        --ood NFCorpus Bioinformatics Biology Medical-Sciences MIRIAD --output-dir figures
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from src.interpretability.chart_style import float_table, grouped_tabular, save_text, tex_escape
from src.interpretability.table_classification import (DATASET_LABELS, TYPE_LABELS, add_dataset_args,
                                                        load_dataset_preds)


def scores(d: dict) -> dict[tuple, float]:
    """{(neuron, row): predicted probability} of one prediction file, parse errors excluded."""
    return {(p["neuron"], p["row"]): p["score"] for p in d["predictions"] if not p["parse_error"]}


def agreement(a: dict, b: dict) -> float:
    """Pearson correlation of two prediction files' scores on the pairs both scored."""
    sa, sb = scores(a), scores(b)
    keys = sorted(set(sa) & set(sb))
    return float(np.corrcoef([sa[k] for k in keys], [sb[k] for k in keys])[0, 1])


def table_llm_agreement(pred_dirs: list[Path], llm_names: list[str], in_domain: str, ood: list[str],
                        description_types: list[str], output_dir: str, label: str | None = None) -> str:
    pairs = list(itertools.combinations(range(len(pred_dirs)), 2))
    per_type = {t: [] for t in description_types}
    rows = []
    for name in [in_domain, *ood]:
        loaded = [load_dataset_preds(d, name, in_domain, description_types)[0] for d in pred_dirs]
        cells = []
        for t in description_types:
            for i, j in pairs:
                r = agreement(loaded[i][t], loaded[j][t]) if loaded[i][t] and loaded[j][t] else None
                cells.append(f"{r:.3f}" if r is not None else "--")
                if r is not None:
                    per_type[t].append(r)
        rows.append(([tex_escape(DATASET_LABELS.get(name, name))], cells))
    for t, vals in per_type.items():
        print(f"{t}: agreement {min(vals):.2f} to {max(vals):.2f} across {len(pairs)} LLM pairs and datasets")
    names = [f"{tex_escape(llm_names[i])} / {tex_escape(llm_names[j])}" for i, j in pairs]
    groups = [(TYPE_LABELS[t], names) for t in description_types]
    caption = ("Agreement between Activation Prediction LLMs: Pearson correlation of their predicted "
               "probabilities over the (neuron, text) pairs both scored.")
    table = float_table(grouped_tabular(["Dataset"], groups, rows), label or "tab:llm_agreement", caption)
    return save_text(table, "sae_llm_agreement_" + in_domain, output_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", nargs="+", required=True, type=Path,
                   help="One directory per Activation Prediction LLM, holding pred_binary_*.json")
    p.add_argument("--llm-names", nargs="+", required=True, help="Label for each --pred-dir")
    add_dataset_args(p)
    a = p.parse_args()
    if len(a.pred_dir) != len(a.llm_names):
        p.error("--pred-dir and --llm-names need the same number of entries")
    table_llm_agreement(a.pred_dir, a.llm_names, a.in_domain, a.ood, a.description_types, a.output_dir, a.label)


if __name__ == "__main__":
    main()
