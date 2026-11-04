"""Build a fixed eval set for predict_neurons_binary.py, from one or more *_with_sae.pkl.

Pools every --eval-file (and every OOD corpus under --ood-dir, with --dataset-name), dedups by the
embedded text, and drops rows the labelling LLM has seen: a verbatim --neuron-profiles exemplar, or
another chunk of the same document (--title-col). Every row records its "source".

--per-neuron rows come from each neuron's positives, plus --n-negatives rows firing none of them.
A positive is either the row being among the neuron's --top-n strongest activations, or the neuron
being among the row's --rank-band K_MIN..K_MAX strongest features. Each row records
{neuron: activation} for the neurons firing on it, and "positive_for". Neurons left without a
positive row are dropped from the written "neurons" list.

Usage:
    uv run python -m src.interpretability.build_interpret_eval \
        --neuron-profiles cure_k64_neuron_profiles.json \
        --eval-file .../converted/corpus_train_with_sae.pkl .../converted/corpus_val_with_sae.pkl \
        --top-n 100 --per-neuron 5 --n-negatives 20 --output cure_k64_eval_rows.json

    uv run python -m src.interpretability.build_interpret_eval \
        --neuron-profiles cure_k64_neuron_profiles.json \
        --ood-dir .../cure_k64_..._gao/ood_eval --dataset-name CURE --exclude MIRIAD \
        --rank-band 1 10 --per-neuron 5 --n-negatives 20 --output cure_k64_eval_rows_ood.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.utils.cli import embedded_text_col
from src.utils.datasets_constants import ood_corpus_files
from src.utils.interpretability_utils import (MAX_CHUNKS_PER_DOC, dedup_by_text, load_sae_codes,
                                              neuron_fired_columns, neuron_fired_counts,
                                              neuron_fired_mask, topn_fired_mask)


EVAL_TEXT_COL = "_eval_text"
EVAL_TITLE_COL = "_eval_title"
EVAL_SOURCE_COL = "_eval_source"


def load_pool(eval_files: list[tuple[str, Path]], text_col: str, hidden: int | None,
              exemplar_texts: set, exemplar_titles: set | None = None, title_col: str = "title",
              one_chunk_per_title: bool = True):
    """(Zc, df) for every (source name, eval file) pooled, deduped by text and leakage-excluded. Each
    resolves its own embedded column into EVAL_TEXT_COL; a row both files carry keeps the first
    file's source name. Rows sharing a title with a shown exemplar are dropped too."""
    Zs, dfs = [], []
    for name, f in eval_files:
        Zc, df = load_sae_codes(f, hidden)
        df[EVAL_TEXT_COL] = df[embedded_text_col(df, text_col)].astype(str)
        # Stripped, to match the exemplar titles select_neurons.py records.
        df[EVAL_TITLE_COL] = (df[title_col].astype(str).str.strip() if title_col in df.columns
                              else "")
        df[EVAL_SOURCE_COL] = name
        Zs.append(Zc)
        dfs.append(df[[EVAL_TEXT_COL, EVAL_TITLE_COL, EVAL_SOURCE_COL]])
    Zc = sp.vstack(Zs, format="csr") if sp.issparse(Zs[0]) else np.vstack(Zs)
    df = pd.concat(dfs, ignore_index=True)
    if len(eval_files) > 1:
        print(f"pooled {len(eval_files)} files -> {len(df)} rows")
    Zc, df, _ = dedup_by_text(Zc, df, EVAL_TEXT_COL, "eval")
    leaked_text = df[EVAL_TEXT_COL].isin(exemplar_texts).values
    leaked_title = ((df[EVAL_TITLE_COL] != "") & df[EVAL_TITLE_COL].isin(exemplar_titles or set())).values
    if one_chunk_per_title:
        # Cap the chunks one paper contributes.
        titled = df[EVAL_TITLE_COL] != ""
        rank = df.groupby(EVAL_TITLE_COL).cumcount()
        first = (~titled | (rank < MAX_CHUNKS_PER_DOC)).values
    else:
        first = np.ones(len(df), dtype=bool)
    keep = np.where(~(leaked_text | leaked_title) & first)[0]
    if leaked_text.sum():
        print(f"excluded {int(leaked_text.sum())} rows verbatim-matching a shown exemplar")
    if (leaked_title & ~leaked_text).sum():
        print(f"excluded {int((leaked_title & ~leaked_text).sum())} further rows from a shown "
              f"exemplar's own document (same title)")
    if (~first).sum():
        print(f"kept at most {MAX_CHUNKS_PER_DOC} chunks per paper, dropping "
              f"{int((~first).sum())} further rows")
    return Zc[keep], df.iloc[keep].reset_index(drop=True)


def row_ranks(Zc, neurons: list[int]) -> list[dict]:
    """Per row, {neuron: rank} over `neurons` active on it -- rank 1 is the row's strongest of the k
    features TopK leaves active."""
    C = sp.csr_matrix(neuron_fired_columns(Zc)).tocsr()
    keep = set(neurons)
    out = []
    for r in range(C.shape[0]):
        s, e = C.indptr[r], C.indptr[r + 1]
        idx, val = C.indices[s:e], C.data[s:e]
        out.append({str(idx[j]): rank for rank, j in enumerate(np.argsort(val)[::-1], 1)
                    if idx[j] in keep})
    return out


def topn_band_mask(Zc, neurons: list[int], top_n: int | None) -> np.ndarray:
    """(rows, len(neurons)) boolean: the row is among that neuron's top_n strongest activations."""
    band = neuron_fired_columns(topn_fired_mask(Zc, top_n)) if top_n else neuron_fired_columns(Zc)
    return np.column_stack([neuron_fired_mask(band, n) for n in neurons])


def rank_band_mask(ranks: list[dict], neurons: list[int], lo: int, hi: int) -> np.ndarray:
    """(rows, len(neurons)) boolean: the neuron is among the row's lo..hi strongest features. Widen
    for more positives, lower (11-20) to ask about a present-but-not-dominant neuron."""
    return np.array([[lo <= r.get(str(n), 2**31) <= hi for n in neurons] for r in ranks])


def sample_indices(Zc, neurons: list[int], pos_mask: np.ndarray, per_neuron: int | None,
                   n_negatives: int, max_eval: int, seed: int) -> np.ndarray:
    """Row indices to evaluate on: per_neuron rows from each neuron's positive band (pos_mask), plus
    n_negatives rows firing none of them."""
    rng = np.random.default_rng(seed)
    if per_neuron is None:
        return np.sort(rng.choice(Zc.shape[0], min(max_eval, Zc.shape[0]), replace=False))
    chosen = []
    for j, n in enumerate(neurons):
        strong = np.where(pos_mask[:, j])[0]
        if len(strong) < per_neuron:
            print(f"  neuron {n}: only {len(strong)} rows in its positive band (< {per_neuron})")
        chosen.append(rng.choice(strong, min(per_neuron, len(strong)), replace=False))
    if n_negatives:
        cols = neuron_fired_columns(Zc)
        fires_any = np.column_stack([neuron_fired_mask(cols, n) for n in neurons]).any(axis=1)
        none_fire = np.where(~fires_any)[0]
        chosen.append(rng.choice(none_fire, min(n_negatives, len(none_fire)), replace=False))
    return np.unique(np.concatenate(chosen))


def row_activations(Zc, neurons: list[int]) -> list[dict]:
    """Per row, {neuron: activation} for every neuron in `neurons` firing on it (Zc > 0). An absent
    neuron does not fire at all -- the only sound basis for a negative."""
    cols = neuron_fired_columns(Zc)
    vals = np.column_stack([np.asarray(cols[:, n].todense()).ravel() if sp.issparse(cols)
                            else cols[:, n] for n in neurons])
    return [{str(n): float(v) for n, v in zip(neurons, row) if v > 0} for row in vals]


def band_thresholds(Zc, neurons: list[int], top_n: int | None) -> dict:
    """Each neuron's top_n-th highest activation -- the cut a positive must clear. Read off Zc, not
    topn_fired_mask, which returns a mask (every kept entry 1.0) rather than the values."""
    if not top_n:
        return {}
    cols = neuron_fired_columns(Zc)
    out = {}
    for n in neurons:
        col = np.asarray(cols[:, n].todense()).ravel() if sp.issparse(cols) else cols[:, n]
        fired = np.sort(col[col > 0])[::-1]
        out[str(n)] = float(fired[min(top_n, len(fired)) - 1]) if len(fired) else 0.0
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--neuron-profiles", required=True, type=Path, help="select_neurons.py's output JSON")
    p.add_argument("--eval-file", type=Path, nargs="+", default=[],
                   help="One or more *_with_sae.pkl to pool and sample rows from")
    p.add_argument("--dataset-name", default=None,
                   help="In-domain dataset name (a src.utils.datasets_constants name, e.g. CURE) — "
                        "required by --ood-dir, to resolve ood_datasets_for(--dataset-name)")
    p.add_argument("--ood-dir", type=Path, default=None,
                   help="Directory holding {name}/corpus*_with_sae.pkl for each OOD dataset in "
                        "ood_datasets_for(--dataset-name), pooled in alongside any --eval-file")
    p.add_argument("--exclude", nargs="+", default=[], help="OOD dataset names to drop")
    p.add_argument("--text-col", default="text")
    p.add_argument("--title-col", default="title",
                   help="Column identifying a document, so other chunks of a shown paper are excluded")
    p.add_argument("--keep-all-chunks", action="store_true",
                   help=f"Keep every chunk of a paper in the pool (default: at most "
                        f"{MAX_CHUNKS_PER_DOC} rows per paper)")
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--indices", type=int, nargs="+", default=None, help="Specific pooled row indices")
    p.add_argument("--top-n", type=int, default=100,
                   help="Positives come from each neuron's top-N strongest rows (default 100)")
    p.add_argument("--rank-band", type=int, nargs=2, metavar=("K_MIN", "K_MAX"), default=None,
                   help="Positives are rows where the neuron ranks K_MIN..K_MAX among the row's own "
                        "active features, instead of --top-n. Needs no per-corpus calibration, so an "
                        "OOD corpus is scored by the same test; widen or lower the band for difficulty")
    p.add_argument("--per-neuron", type=int, default=None,
                   help="Stratify: this many firing rows per neuron, instead of a uniform --max-eval sample")
    p.add_argument("--n-negatives", type=int, default=20,
                   help="--per-neuron only: rows where no selected neuron fires (default 20)")
    p.add_argument("--max-eval", type=int, default=50, help="Uniform sample size, without --per-neuron")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    if args.ood_dir and not args.dataset_name:
        p.error("--ood-dir requires --dataset-name")
    # Named by dataset for OOD, by file stem otherwise.
    files = [(f.stem, f) for f in args.eval_file]
    ood_sources = []
    if args.ood_dir:
        ood = ood_corpus_files(args.ood_dir, args.dataset_name, args.exclude)
        files += ood
        ood_sources = [n for n, _ in ood]
    if not files:
        p.error("at least one of --eval-file or --ood-dir is required")

    profiles = json.load(open(args.neuron_profiles))
    neurons = [n["neuron"] for n in profiles["neurons"]]
    # Negative snippets count as shown too.
    snips = [s for n in profiles["neurons"] for s in
             n["corpus_snippets"] + n["query_snippets"] + n.get("negative_snippets", [])]
    exemplar_texts = {s["text"] for s in snips}
    exemplar_titles = {s["title"] for s in snips if s.get("title")}

    Zc, df = load_pool(files, args.text_col, args.hidden, exemplar_texts, exemplar_titles,
                       args.title_col, not args.keep_all_chunks)
    # One criterion, applied once here; the rows record the verdict.
    ranks = row_ranks(Zc, neurons) if args.rank_band else None
    pos_mask = (rank_band_mask(ranks, neurons, *args.rank_band) if args.rank_band
                else topn_band_mask(Zc, neurons, args.top_n))
    idx = (np.array(args.indices) if args.indices is not None else
           sample_indices(Zc, neurons, pos_mask, args.per_neuron, args.n_negatives, args.max_eval,
                          args.seed))
    # Without a positive a neuron is scored on negatives only.
    kept = np.where(pos_mask[idx].any(axis=0))[0]
    if len(kept) < len(neurons):
        counts = neuron_fired_counts(Zc)
        dropped = [n for j, n in enumerate(neurons) if j not in set(kept.tolist())]
        never = sum(1 for n in dropped if counts[n] == 0)
        print(f"dropped {len(dropped)} of {len(neurons)} neurons: {never} never fire on a pooled "
              f"row, {len(dropped) - never} fire but have no positive row")
        neurons = [neurons[j] for j in kept]
        pos_mask = pos_mask[:, kept]
        keep_ids = {str(n) for n in neurons}
        if ranks is not None:
            ranks = [{k: v for k, v in r.items() if k in keep_ids} for r in ranks]
    acts = row_activations(Zc[idx], neurons)
    thresholds = {} if args.rank_band else band_thresholds(Zc, neurons, args.top_n)
    sub = df.iloc[idx]
    texts, sources = sub[EVAL_TEXT_COL].tolist(), sub[EVAL_SOURCE_COL].tolist()
    rows = []
    for i, r in enumerate(idx):
        row = {"row": int(r), "source": sources[i], "text": texts[i], "activations": acts[i],
               "positive_for": [int(n) for j, n in enumerate(neurons) if pos_mask[r, j]]}
        if ranks is not None:
            row["ranks"] = ranks[r]
        rows.append(row)
    n_fire = sum(1 for r in rows if r["activations"])
    n_pos = sum(len(r["positive_for"]) for r in rows)
    print(f"{len(rows)} rows: {n_fire} firing >=1 neuron, {len(rows) - n_fire} firing none; "
          f"{n_pos} positive (row, neuron) pairs")
    if len(files) > 1:
        print("  by source: " + ", ".join(f"{s}: {sources.count(s)}" for s in dict.fromkeys(sources)))

    result = {"neuron_profiles_file": str(args.neuron_profiles),
             "eval_files": {n: str(f) for n, f in files},
             # Which sources are a different corpus from the one the descriptions were built on.
             "ood_sources": ood_sources,
             "seed": args.seed, "per_neuron": args.per_neuron,
             "n_negatives": args.n_negatives if args.per_neuron else None,
             "max_eval": args.max_eval if args.per_neuron is None and args.indices is None else None,
             "indices": args.indices, "neurons": neurons,
             "criterion": ({"type": "rank", "band": args.rank_band} if args.rank_band else
                           {"type": "activation", "top_n": args.top_n}),
             "top_n": None if args.rank_band else args.top_n,
             "band_thresholds": thresholds, "n_rows": len(rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.output, "w"), indent=2)
    print(f"wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
