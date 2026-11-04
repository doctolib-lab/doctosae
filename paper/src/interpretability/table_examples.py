"""LaTeX table of example neuron descriptions: one column per description type, one row per
dataset. Descriptions are built on the in-domain corpus, and the neurons shown are those firing
on the OOD corpus, whose row gives their strongest activating example there.

Usage:
    uv run python -m src.interpretability.table_examples \
        --pred-dir $RD/interpretation --in-domain CURE --ood MIRIAD \
        --output-dir figures
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.interpretability.chart_style import save_text, tex_escape
from src.interpretability.predict_neurons_binary import build_class_description
from src.interpretability.table_classification import DATASET_LABELS, TYPE_LABELS

DEFAULT_TYPES = ("tfidf", "llm", "snippets")
ERROR_TOPICS = ("LABEL_ERROR", "PARSE_ERROR")
RANK_TYPE = "tfidf"  # neurons are the best-classified ones from their TF-IDF description
# Wikipedia section-edit links, e.g. "[edit]" or "[edit on Wikidata]", left in the OOD corpora.
EDIT_MARKER = re.compile(r"\[\s*edit\b[^\]]*\]", re.IGNORECASE)
# Row break inside a cell, kept out of band so tex_escape cannot mangle it.
SNIPPET_BREAK = "\x00"


def ood_examples(pred_dir: Path, corpus: str) -> dict[int, dict]:
    """Each neuron's top activating document over a whole OOD corpus, keyed by neuron."""
    path = pred_dir / f"eval_rows_{corpus}.json"
    if not path.exists():
        print(f"  no file for {corpus} at {path}")
        return {}
    rows = json.load(open(path))
    best: dict[int, dict] = {}
    for row in rows["rows"] if isinstance(rows, dict) else rows:
        for key, value in row["activations"].items():
            neuron, activation = int(key), float(value)
            if activation > best.get(neuron, {"activation": 0.0})["activation"]:
                best[neuron] = {"activation": activation, "text": row["text"]}
    return best


def rank_neurons(pred_dir: Path, rank_type: str, profiles: dict, labels: dict, need_label: bool,
                 candidates: set[int] | None = None,
                 exclude: set[int] = frozenset()) -> list[tuple[int, float]]:
    """Neurons and their in-domain Pearson correlation on `rank_type`, best first, keeping those
    with a profile, a usable label when labels are shown, and in `candidates` but not `exclude`."""
    mets = json.load(open(pred_dir / f"pred_binary_{rank_type}.json"))["metrics"]
    scored = []
    for key, m in mets.items():
        if key in ("micro avg", "macro avg"):
            continue
        neuron = int(key)
        if neuron in exclude or neuron not in profiles:
            continue
        if candidates is not None and neuron not in candidates:
            continue
        if need_label and labels.get(neuron, {}).get("topic") in (None, *ERROR_TOPICS):
            continue
        scored.append((float(m["pearson"]), neuron))
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [(neuron, score) for score, neuron in scored]


def pick_neurons(ranked: list[tuple[int, float]], examples: dict[int, dict],
                 n_neurons: int) -> list[int]:
    """The best-correlated neurons, those tied at the top ordered by their OOD activation."""
    if not ranked:
        return []
    best = round(ranked[0][1], 3)
    tied = [neuron for neuron, score in ranked if round(score, 3) == best]
    if examples:
        tied.sort(key=lambda n: -examples[n]["activation"])
    return tied[:n_neurons]


def _body(snippet: dict) -> str:
    """The snippet's text without the leading title, which the embedded text repeats."""
    text, title = snippet["text"], snippet["title"]
    return text[len(title):].lstrip() if title and text.startswith(title) else text


def snippet_titles(neuron: dict, n: int) -> str:
    """The neuron's top activating corpus documents and top query, activation-prefixed."""
    docs = [f"Doc [{s['activation']:.2f}] {_body(s)}" for s in neuron["corpus_snippets"][:n]]
    queries = [f"Query [{s['activation']:.2f}] {s['text']}" for s in neuron["query_snippets"][:1]]
    return SNIPPET_BREAK.join(docs + queries)


def _cell(text: str, max_chars: int | None) -> str:
    """One cell's text: wiki edit markers dropped, whitespace collapsed, escaped, and each
    break-separated part cut at a word boundary when max_chars is set."""
    parts = []
    for part in str(text).split(SNIPPET_BREAK):
        flat = " ".join(EDIT_MARKER.sub(" ", part).split())
        if max_chars is not None and len(flat) > max_chars:
            head = flat[:max_chars].rstrip()
            flat = (head[:head.rfind(" ")].rstrip() if " " in head else head) + "..."
        parts.append(tex_escape(flat))
    return " \\newline ".join(parts)


def build_table(blocks: list[tuple], types: list[str], in_domain: str, ood: str | None, label: str,
                max_chars: int, model_name: str | None = None, k: int | None = None,
                compact: bool = False, max_corpus_examples: int = 1) -> str:
    """One column per neuron: a row per description type, then the OOD example row. A single
    neuron gives a one-column `table`, more give a page-wide `table*`."""
    single = len(blocks) == 1
    unit, frac = ("\\linewidth", 0.70) if single else ("\\textwidth", 0.82 / len(blocks))
    cols = " ".join(f"p{{{frac:.3f}{unit}}}" for _ in blocks)
    lines = ["\\vspace{-15pt}"] if compact else []
    lines.append("\\setlength{\\tabcolsep}{2pt}")
    if compact:
        lines += ["\\setlength{\\aboverulesep}{0pt}", "\\setlength{\\belowrulesep}{1pt}",
                  "\\renewcommand{\\arraystretch}{1}"]
    lines.append(f"\\begin{{tabular}}{{@{{}}l {cols}@{{}}}}")
    if not compact:
        lines.append("\\toprule")
    lines += [" & " + " & ".join(
                  f"{{\\small\\textbf{{Neuron {neuron}}}"
                  + (f" --- {tex_escape(topic)}" if topic else "") + "}"
                  for neuron, topic, _, _ in blocks) + " \\\\", "\\midrule"]
    for t in types:
        name = TYPE_LABELS.get(t, t)
        # The Interpreter LLM's wording is the object of study, so it is never cut.
        cell_chars = None if t == "llm" else max_chars
        lines.append(f"{{\\small {name}}} & " + " & ".join(
            _cell(descriptions[t], cell_chars) for _, _, descriptions, _ in blocks) + " \\\\")
        lines.append("\\addlinespace[3pt]")
    if ood:
        lines.append("\\midrule")
        lines.append(f"{{\\small {tex_escape(DATASET_LABELS.get(ood, ood))}}} & " + " & ".join(
            "--" if example is None else
            _cell(f"[{float(example['activation']):.2f}] {example['text']}", max_chars)
            for _, _, _, example in blocks) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]

    origin = f", from the {model_name}-trained SAE" if model_name else ""
    if origin and k:
        origin += f" with $k={k}$"
    subject = ("Example interpretations of a neuron" if single else
               "Example neuron interpretations, one column per neuron")
    n_docs = max_corpus_examples
    shown = (f" (only {n_docs} passage{'s' if n_docs > 1 else ''} and 1 query shown in Snippets, "
             "for space)") if compact and "snippets" in types else ""
    caption = (f"{subject}{origin}{shown}. A bracketed [] number is the neuron's activation on the "
               "text that follows it.")
    if ood:
        caption += f" The last row gives its top activating document over {tex_escape(ood)}."
    env = "table" if single else "table*"
    return "\n".join([f"\\begin{{{env}}}[t]", "\\centering",
                      "\\scriptsize" if compact else "\\small", "\\floatconts",
                      f"  {{{label}}}", f"  {{\\caption{{{caption}}}}}", "  {%",
                      *("  " + line for line in lines), "  }",
                      *(["    \\vspace{-10pt}"] if compact else []), f"\\end{{{env}}}"])


def table_examples(pred_dir: Path, in_domain: str, output_dir: str, ood: str | None = None,
                   description_types: list[str] = DEFAULT_TYPES, n_neurons: int = 1,
                   max_chars: int | None = None, label: str = "tab:neuronexamples",
                   model_name: str | None = None, dated: bool = True,
                   max_example_chars: int = 200, max_corpus_examples: int = 1,
                   max_query_examples: int = 1, compact: bool = False,
                   exclude: set[int] = frozenset(), neurons: list[int] = ()) -> str:
    pred_dir = Path(pred_dir)
    max_chars = (100 if compact else 320) if max_chars is None else max_chars
    # Only the run whose description type is "llm" records the labels file it read.
    metas = {t: json.load(open(pred_dir / f"pred_binary_{t}.json")) for t in description_types}
    profiles = {n["neuron"]: n for n in
                json.load(open(next(iter(metas.values()))["neuron_profiles_file"]))["neurons"]}
    labels_file = next((m["labels_file"] for m in metas.values() if m["labels_file"]), None)
    labels = ({r["neuron"]: r for r in json.load(open(labels_file))["neurons"]}
              if labels_file else {})

    # An OOD corpus only restricts the neurons to those firing there and adds its example row;
    # the ranking is the in-domain TF-IDF correlation either way.
    examples = ood_examples(pred_dir, ood) if ood else {}
    if ood:
        print(f"{len(examples)} neurons fire on {ood}")
    ranked = rank_neurons(pred_dir, RANK_TYPE, profiles, labels,
                          "llm" in description_types, set(examples) if ood else None, exclude)
    scores = dict(ranked)
    if neurons:
        missing = [n for n in neurons if n not in profiles]
        if missing:
            raise ValueError(f"no profile for neuron(s) {missing} — known: {len(profiles)} neurons")
    chosen = list(neurons) if neurons else pick_neurons(ranked, examples, n_neurons)
    blocks = []
    for neuron in chosen:
        score = scores.get(neuron)
        # Compact keeps the snippets column to document titles, which survive its shorter cells.
        descriptions = {t: (snippet_titles(profiles[neuron], max_corpus_examples)
                            if compact and t == "snippets" else
                            build_class_description(profiles[neuron], t, labels.get(neuron),
                                                    max_example_chars, max_corpus_examples,
                                                    max_query_examples))
                        for t in description_types}
        topic = labels.get(neuron, {}).get("topic", "")
        blocks.append((neuron, topic, descriptions, examples.get(neuron)))
        act = examples.get(neuron, {}).get("activation")
        r = f"{score:.3f}" if score is not None else "n/a"
        print(f"neuron {neuron}: {topic}, {RANK_TYPE} r={r} on {in_domain}"
              + (f", top {ood} activation {act:.3f}" if act else ""))

    cfg = pred_dir.parent / "sae_config.json"
    k = json.loads(cfg.read_text())["k"] if cfg.exists() else None
    table = build_table(blocks, list(description_types), in_domain, ood, label, max_chars,
                        model_name, k, compact, max_corpus_examples)
    stem = "sae_neuron_examples_" + "_".join(
        filter(None, [in_domain, ood, model_name, "compact" if compact else ""]))
    return save_text(table, stem, output_dir, dated=dated)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", required=True, type=Path,
                   help="Directory holding predict_neurons_binary.py's pred_binary_*.json")
    p.add_argument("--in-domain", required=True, help="In-domain dataset name, for the filename")
    p.add_argument("--ood", default=None,
                   help="OOD dataset name, read from pred_binary_{type}_{name}.json")
    p.add_argument("--description-types", nargs="+", default=list(DEFAULT_TYPES),
                   choices=["snippets", "tfidf", "llm"], help="One column each")
    p.add_argument("--n-neurons", type=int, default=1,
                   help="Neurons to show, one column each (default 1, a one-column table)")
    p.add_argument("--neurons", nargs="+", type=int, default=[],
                   help="Neuron ids to show, in this order, instead of the best-correlated ones")
    p.add_argument("--exclude", nargs="+", type=int, default=[],
                   help="Neuron ids to leave out of the selection")
    p.add_argument("--max-chars", type=int, default=None,
                   help="Characters per cell (default 320, or 100 with --compact)")
    p.add_argument("--compact", action="store_true",
                   help="Quarter-page variant: smaller font, tighter rows, shorter cells")
    p.add_argument("--max-example-chars", type=int, default=200)
    p.add_argument("--max-corpus-examples", type=int, default=1)
    p.add_argument("--max-query-examples", type=int, default=1)
    p.add_argument("--label", default="tab:neuronexamples", help="LaTeX label")
    p.add_argument("--model-name", default=None, help="Checkpoint identifier, for the filename only")
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    table_examples(a.pred_dir, a.in_domain, a.output_dir, a.ood, a.description_types, a.n_neurons,
                   a.max_chars, a.label, a.model_name, max_example_chars=a.max_example_chars,
                   max_corpus_examples=a.max_corpus_examples,
                   max_query_examples=a.max_query_examples, compact=a.compact,
                   exclude=set(a.exclude), neurons=a.neurons)


if __name__ == "__main__":
    main()
