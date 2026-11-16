"""LaTeX table of predict_neurons_binary.py's results, averaged over neurons: one row per corpus,
one column group per description type. A missing file prints '--'. The column is $r$.

Usage:
    uv run python -m src.interpretability.table_classification \
        --pred-dir $RD/interpretation --in-domain CURE \
        --ood NFCorpus Bioinformatics Biology Medical-Sciences MIRIAD \
        --output-dir figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.interpretability.chart_style import float_table, save_text, tex_escape

TYPE_LABELS = {"tfidf": "TF-IDF terms", "llm": "Interpreter LLM", "snippets": "Snippets"}
DATASET_LABELS = {"Medical-Sciences": "Med-Sciences"}
CORR_METRICS = [("pearson", "$r$")]
# Read from each neuron's own metrics entry; "abstained" is derived from the predictions.
NEURON_KEYS = [("pearson", "$r$")]
# Lower is not better, so this column takes no best-value bold.
UNRANKED = {"abstained"}
_AUTO = object()  # tabcolsep sentinel: "2pt" either way, kept overridable via --tabcolsep


def read_pred(pred_dir: Path, description_type: str, corpus: str | None, band: str = "") -> dict | None:
    """One prediction file's contents, or None when it is missing. `band` is the '_rankLO-HI' tag the
    pipeline adds to a non-default evaluation rank band."""
    suffix = f"_{corpus}" if corpus else ""
    path = pred_dir / f"pred_binary_{description_type}{band}{suffix}.json"
    if not path.exists():
        print(f"  no file for {description_type} {corpus or 'in-domain'} at {path}")
        return None
    return json.load(open(path))


def neuron_keys(d: dict) -> set[str]:
    """The neurons a prediction file scored, as its metrics keys."""
    return {k for k in d["metrics"] if k not in ("micro avg", "macro avg")}


def agg_pred(d: dict, keep: set[str]) -> dict:
    """One prediction file's metrics, averaged over the `keep` neurons (an undefined r, from a neuron with
    only positive or only negative texts, is skipped), plus its run counts."""
    per = [v for k, v in d["metrics"].items() if k in keep]
    out = {k: float(np.nanmean([v[k] for v in per])) for k, _ in NEURON_KEYS}
    scored = [p for p in d["predictions"] if str(p["neuron"]) in keep]
    out["abstained"] = sum(p["score"] == 0.0 for p in scored) / len(scored) if scored else float("nan")
    return {**out, "n_neurons": len(per), "n_calls": d["n_calls"],
            "n_parse_error": d["n_parse_error"], "max_example_chars": d["max_example_chars"]}


def _cell(d: dict | None, key: str, best: float | None) -> str:
    """One cell, bold when it ties the row's best for this metric."""
    if not d:
        return "--"
    return f"\\textbf{{{d[key]:.3f}}}" if round(d[key], 3) == best else f"{d[key]:.3f}"


def build_table(rows: list[tuple[str, dict[str, dict | None]]], types: list[str],
                label: str, metrics: list[tuple[str, str]] = CORR_METRICS,
                tabcolsep: str | None = _AUTO, model_name: str | None = None,
                k: int | None = None) -> str:
    """One row per corpus, one group of `metrics` columns per description type."""
    if tabcolsep is _AUTO:
        tabcolsep = "2pt"
    n_sub = len(metrics)
    lines = [f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}"] if tabcolsep else []
    # Keeps booktabs' rule glue from breaking the vertical rules.
    lines += ["\\setlength{\\aboverulesep}{1pt}", "\\setlength{\\belowrulesep}{1pt}",
              f"\\begin{{tabular}}{{@{{}}l|{'|'.join('r' * n_sub for _ in types)}@{{}}}}", "\\toprule",
              "Dataset & " + " & ".join(
                  f"\\multicolumn{{{n_sub}}}{{c{'|' if i < len(types) - 1 else ''}}}"
                  f"{{{TYPE_LABELS.get(t, t)}}}" for i, t in enumerate(types)) + " \\\\"]
    lines.append(" & " + " & ".join(h for _ in types for _, h in metrics) + " \\\\")
    lines.append("\\midrule")
    for corpus, per_type in rows:
        best = {k: max((round(per_type[t][k], 3) for t in types if per_type[t]), default=None)
                if k not in UNRANKED else None for k, _ in metrics}
        cells = [_cell(per_type[t], k, best[k]) for t in types for k, _ in metrics]
        lines.append(f"{tex_escape(DATASET_LABELS.get(corpus, corpus))} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]

    origin = f", from the {model_name}-trained SAE" if model_name else ""
    if origin and k:
        origin += f" with $k={k}$"
    caption = (f"Activation Prediction Task on neuron's interpretations{origin}. $r$ is the Pearson "
               "correlation of the predicted probability with the ground truth (positive/negative), "
               "averaged over the neurons every interpretation type scored. ")
    if any(k == "abstained" for k, _ in metrics):
        caption += ("$p_0$ is the share of prompts the judge answered with a score of exactly $0$, "
                    "its ``uncertain'' option. ")
    caption += "Per dataset, best $r$ across interpretation types in bold."
    return float_table(lines, label, caption)


def load_dataset_preds(pred_dir: Path, name: str, in_domain: str,
                       description_types: list[str]) -> tuple[dict[str, dict | None], set[str]]:
    """Each description type's prediction file for one dataset (None when missing), and the
    neurons every one of them scored."""
    raw = {t: read_pred(pred_dir, t, None if name == in_domain else name) for t in description_types}
    found = [d for d in raw.values() if d]
    return raw, (set.intersection(*(neuron_keys(d) for d in found)) if found else set())


def agg_pred_by_dataset(pred_dir: Path, in_domain: str, ood: list[str],
                 description_types: list[str]) -> list[tuple[str, dict[str, dict | None]]]:
    """Per dataset, each description type's metrics averaged over the neurons all types scored."""
    rows = []
    for name in [in_domain, *ood]:
        raw, keep = load_dataset_preds(pred_dir, name, in_domain, description_types)
        rows.append((name, {t: agg_pred(d, keep) if d else None for t, d in raw.items()}))
    return rows


def add_dataset_args(p: argparse.ArgumentParser, default_types: tuple[str, ...] = ("tfidf", "llm", "snippets"),
                     label_help: str = "LaTeX label") -> None:
    """The arguments every table script shares: which datasets, which description types, where to write."""
    p.add_argument("--in-domain", required=True,
                   help="In-domain dataset name, read from the unsuffixed pred_binary_{type}.json")
    p.add_argument("--ood", nargs="+", default=[],
                   help="OOD dataset names, each read from pred_binary_{type}_{name}.json")
    p.add_argument("--description-types", nargs="+", default=list(default_types),
                   choices=["snippets", "tfidf", "llm"], help="One column group each")
    p.add_argument("--label", default=None, help=label_help)
    p.add_argument("--output-dir", required=True)


def table_classification(pred_dir: Path, in_domain: str, output_dir: str,
                         ood: list[str] = (), description_types: list[str] = ("tfidf", "llm"),
                         label: str | None = None, model_name: str | None = None,
                         dated: bool = True, tabcolsep: str | None = _AUTO) -> str:
    rows = agg_pred_by_dataset(pred_dir, in_domain, ood, list(description_types))
    for name, per_type in rows:
        for t, d in per_type.items():
            if d:
                print(f"{name} {t}: {d['n_neurons']} neurons, {d['n_calls']} calls, "
                      f"{d['n_parse_error']} parse errors, "
                      f"{d['abstained']:.3f} scored exactly 0")
    label = label or "tab:neuronclassif"
    # k comes from the run's own sae_config.json.
    cfg = Path(pred_dir).parent / "sae_config.json"
    k = json.loads(cfg.read_text())["k"] if cfg.exists() else None
    table = build_table(rows, list(description_types), label, CORR_METRICS, tabcolsep,
                        model_name, k)
    stem = "sae_neuron_classification_" + "_".join(filter(None, [in_domain, model_name]))
    return save_text(table, stem, output_dir, dated=dated)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", required=True, type=Path,
                   help="Directory holding predict_neurons_binary.py's pred_binary_*.json")
    add_dataset_args(p, default_types=("tfidf", "llm"), label_help="LaTeX label (default: tab:neuronclassif)")
    p.add_argument("--tabcolsep", default=_AUTO,
                   help="\\setlength{\\tabcolsep}{...} before the tabular (default '2pt', vs LaTeX's "
                        "own 6pt) to narrow column padding. Pass '' to leave it unset.")
    p.add_argument("--model-name", default=None, help="Checkpoint identifier, for the filename only")
    a = p.parse_args()
    table_classification(a.pred_dir, a.in_domain, a.output_dir, a.ood, a.description_types,
                         a.label, a.model_name, tabcolsep=a.tabcolsep)


if __name__ == "__main__":
    main()
