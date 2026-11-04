"""Select concept-specific SAE neurons for LLM labeling, and pull their top-activating text.

Candidates: neurons whose in-domain corpus firing rate falls in [--fr-min, --fr-max]. With
--ood-dir (and --dataset-name), the mean firing rate over those OOD datasets must fall in the same
band. A candidate must also fire on at least --n-corpus-snippets corpus rows and
--n-query-snippets query rows.

Candidates keep index order unless --entropy-theme is passed, which ranks them by theme entropy
ascending over every doc/query the neuron fires on, or over its top --top-n corpus or query docs.
warn_if_tied_ranking flags a saturated ranking. --diversity-overlap skips a candidate whose top
--n-terms TF-IDF terms overlap an already-kept neuron's.

Writes per kept neuron: firing rate, entropy, dominant theme, theme distribution, top --n-terms
TF-IDF terms, and its top --n-corpus-snippets passages and --n-query-snippets queries with their
activation values. Terms are the mean TF-IDF over those same passages and queries, in a space fit
over every corpus text and query. Read by label_neurons.py and predict_neurons_binary.py.

Usage:
    RUN=runs/repo_cure/gemini-embedding-001/cure_k64_l5e-6_a0.1_lr5e-4_temp0.05_s42_h27648
    uv run python -m src.interpretability.select_neurons \
        --sae-corpus-file $RUN/converted/corpus_train_with_sae.pkl \
        --queries-file $RUN/converted/queries_test_en_with_sae.pkl \
        --theme-col source_folder --query-theme-col discipline \
        --fr-min 0.01 --fr-max 0.20 --entropy-theme corpus \
        --max-neurons 20 \
        --output cure_k64_neuron_profiles.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.utils.cli import PREPENDED_COL, embedded_text_col
from src.utils.datasets_constants import ood_corpus_files
from src.utils.interpretability_utils import (MAX_CHUNKS_PER_DOC, dedup_by_text, fit_tfidf,
                                              load_sae_codes,
                                              neuron_fired_columns, neuron_fired_counts,
                                              neuron_fired_mask,
                                              neuron_firing_stats, neuron_top_activating,
                                              report_selection, select_diverse_neurons,
                                              theme_entropy_from_labels, top_terms_for_mask,
                                              warn_if_tied_ranking)

# Evidence pulled per neuron.
DEFAULT_N_CORPUS_SNIPPETS = 5
DEFAULT_N_QUERY_SNIPPETS = 5
DEFAULT_N_NEGATIVE_SNIPPETS = 2


def _title_of(row, text_col: str, body_col: str | None, title_col: str) -> str | None:
    """The row's title, when title + " " + body reconstructs PREPENDED_COL exactly; None otherwise.
    Kept beside "text" instead of folded in, since "text" is what the leakage check matches on."""
    if text_col != PREPENDED_COL or not body_col or title_col not in row.index:
        return None
    title, body = str(row[title_col]).strip(), str(row[body_col]).strip()
    if not title or not body or str(row[text_col]) != f"{title} {body}":
        return None
    return title


def _evidence_idx(Zc_cols, df: pd.DataFrame, neuron: int, n: int,
                  distinct_col: str | None) -> np.ndarray:
    """Row indices of the neuron's n strongest activations, at most MAX_CHUNKS_PER_DOC per
    distinct_col value."""
    if not distinct_col or distinct_col not in df.columns:
        return neuron_top_activating(Zc_cols, neuron, n)
    # Stripped, as _title_of strips.
    keys = df[distinct_col].astype(str).str.strip().values
    seen, keep = {}, []
    # len(df) asks for every row; neuron_top_activating caps k at how many are > 0.
    for i in neuron_top_activating(Zc_cols, neuron, len(df)):
        key = keys[int(i)]
        if seen.get(key, 0) >= MAX_CHUNKS_PER_DOC:
            continue
        seen[key] = seen.get(key, 0) + 1
        keep.append(int(i))
        if len(keep) == n:
            break
    return np.array(keep, dtype=int)


def _or_none(value) -> float | None:
    """A float, or None when it is NaN."""
    return None if np.isnan(value) else float(value)


def _column(Zc_cols, neuron: int) -> np.ndarray:
    """One neuron's activation column, dense."""
    return (np.asarray(Zc_cols[:, neuron].todense()).ravel() if sp.issparse(Zc_cols)
            else Zc_cols[:, neuron])


def _snippets(Zc_cols, df: pd.DataFrame, text_col: str, id_col: str | None, neuron: int, n: int,
             body_col: str | None = None, title_col: str = "title",
             idx: np.ndarray | None = None) -> list[dict]:
    """Neuron's top-n activating examples as {id, activation, text, title}, strongest first. "text"
    is verbatim what was embedded -- cap it at render, the leakage check matches on it."""
    idx = neuron_top_activating(Zc_cols, neuron, n) if idx is None else idx
    if len(idx) == 0:
        return []
    col = _column(Zc_cols, neuron)
    out = []
    for i in idx:
        row = df.iloc[int(i)]
        out.append({"id": str(row[id_col]) if id_col else int(i),
                   "activation": float(col[i]), "text": str(row[text_col]),
                   "title": _title_of(row, text_col, body_col, title_col)})
    return out


def _negative_snippets(Zc_cols, df: pd.DataFrame, text_col: str, id_col: str | None, neuron: int,
                       n: int, body_col: str | None = None, title_col: str = "title") -> list[dict]:
    """n documents the neuron does not fire on, as {id, text, title}. Drawn at a fixed seed, so the
    same profiles file is reproducible."""
    idx = np.flatnonzero(_column(Zc_cols, neuron) <= 0)
    if len(idx) == 0 or n <= 0:
        return []
    pick = np.random.default_rng(0).choice(idx, min(n, len(idx)), replace=False)
    return [{"id": str(df.iloc[int(i)][id_col]) if id_col else int(i),
             "text": str(df.iloc[int(i)][text_col]),
             "title": _title_of(df.iloc[int(i)], text_col, body_col, title_col)}
            for i in np.sort(pick)]


def select_neurons(sae_corpus_file: Path, text_col: str, theme_col: str | None,
                   corpus_file: Path | None = None, id_col: str | None = None,
                   hidden: int | None = None, queries_file: Path | list[Path] | None = None,
                   query_text_col: str | None = None, query_id_col: str | None = None,
                   query_theme_col: str | None = None,
                   fr_range: tuple[float, float] = (0.01, 0.20), entropy_source: str | None = None,
                   top_n: int | None = None, max_neurons: int | None = None,
                   n_corpus_snippets: int = DEFAULT_N_CORPUS_SNIPPETS,
                   n_query_snippets: int = DEFAULT_N_QUERY_SNIPPETS,
                   query_min_act_frac: float | None = None, neurons: list[int] | None = None,
                   snippet_distinct_col: str | None = "title",
                   diversity_overlap: float | None = None, n_terms: int = 5, dataset_name: str | None = None,
                   ood_dir: Path | None = None, exclude: list[str] | None = None) -> dict:
    """Select neurons and pull their activating examples, applying in order: `neurons`, the corpus
    firing-rate band, the evidence minimums, an entropy ranking, diversity_overlap dedup and
    max_neurons. Returns the run's parameters, candidate/kept counts and a "neurons" list.
    """
    Zc, sae_df = load_sae_codes(sae_corpus_file, hidden)
    corpus_df = pd.read_pickle(corpus_file) if corpus_file else sae_df
    if len(sae_df) != len(corpus_df):
        raise ValueError(f"--sae-corpus-file ({len(sae_df)} rows) and --corpus-file "
                         f"({len(corpus_df)} rows) must be row-aligned (same corpus, same order)")
    if id_col and corpus_file:
        if not (sae_df[id_col].astype(str).values == corpus_df[id_col].astype(str).values).all():
            raise ValueError(f"--sae-corpus-file and --corpus-file are not row-aligned by --id-col {id_col!r}")
    # Dedup key, TF-IDF fit and snippet text all read the string the SAE saw (title+text).
    base_text_col = text_col
    text_col = embedded_text_col(corpus_df, text_col)
    if text_col != base_text_col:
        print(f"corpus: using {text_col!r} (the embedded string), not {base_text_col!r}")
    Zc, corpus_df, _ = dedup_by_text(Zc, corpus_df, text_col, "corpus")

    fr_corpus = neuron_firing_stats(Zc)
    lo, hi = fr_range
    fr_ok = (fr_corpus >= lo) & (fr_corpus <= hi)
    fr_desc = f"corpus fr in [{lo:.1%}, {hi:.1%}]"

    ood_names = []
    fr_ood_mean = None
    if ood_dir is not None:
        if not dataset_name:
            raise ValueError("--ood-dir requires --dataset-name (resolves ood_datasets_for)")
        ood_files = ood_corpus_files(ood_dir, dataset_name, exclude)
        ood_names = [n for n, _ in ood_files]

        def _ood_fr(name, path):
            Zc_o, df_o = load_sae_codes(path, hidden)
            # Resolved per file.
            Zc_o, _, _ = dedup_by_text(Zc_o, df_o, embedded_text_col(df_o, base_text_col), name)
            return neuron_firing_stats(Zc_o)

        fr_ood = [_ood_fr(n, p) for n, p in ood_files]
        fr_ood_mean = np.mean(fr_ood, axis=0)
        fr_ok = fr_ok & (fr_ood_mean >= lo) & (fr_ood_mean <= hi)
        fr_desc = f"corpus fr and mean OOD fr ({', '.join(ood_names)}) both in [{lo:.1%}, {hi:.1%}]"
    print(f"{int(fr_ok.sum())} / {len(fr_ok)} neurons have {fr_desc}")

    Zc_q = q_df = fr_query = None
    query_files = [queries_file] if isinstance(queries_file, (str, Path)) else list(queries_file or [])
    if query_files:
        # Pooled: the firing rate is over every query split given.
        loaded = [load_sae_codes(f, hidden) for f in query_files]
        Zc_q = sp.vstack([z for z, _ in loaded], format="csr") if sp.issparse(loaded[0][0]) \
            else np.vstack([z for z, _ in loaded])
        q_df = pd.concat([d for _, d in loaded], ignore_index=True)
        # Against q_df's own columns.
        base_query_text_col = query_text_col or base_text_col
        query_text_col = embedded_text_col(q_df, base_query_text_col)
        if len(query_files) > 1:
            print(f"pooled {len(query_files)} query files -> {len(q_df)} queries")
        Zc_q, q_df, _ = dedup_by_text(Zc_q, q_df, query_text_col, "queries")
        fr_query = neuron_firing_stats(Zc_q)

    ent = ent_norm = n_fired_entropy_source = theme_dist = theme_labels = None
    if entropy_source == "corpus":
        if not theme_col:
            raise ValueError("--entropy-theme corpus requires --theme-col")
        (ent, ent_norm, n_fired_entropy_source, _, theme_dist, _, theme_labels,
         _) = theme_entropy_from_labels(Zc, corpus_df, theme_col, "corpus", top_n=top_n)
    elif entropy_source == "queries":
        if not (queries_file and query_theme_col):
            raise ValueError("--entropy-theme queries requires --queries-file and --query-theme-col")
        (ent, ent_norm, n_fired_entropy_source, _, theme_dist, _, theme_labels,
         _) = theme_entropy_from_labels(Zc_q, q_df, query_theme_col, "queries", top_n=top_n)

    # The snippet counts double as minimum firing counts.
    enough = neuron_fired_counts(Zc) >= n_corpus_snippets
    if Zc_q is not None:
        enough &= neuron_fired_counts(Zc_q) >= n_query_snippets
    live = fr_ok & enough
    if neurons is not None:
        # Kept in the order given.
        out_of_range = [n for n in neurons if not 0 <= n < Zc.shape[1]]
        if out_of_range:
            raise ValueError(f"neurons {out_of_range[:10]} are outside 0..{Zc.shape[1] - 1}")
        candidates = np.array([n for n in neurons if live[n]], dtype=int)
        print(f"{len(candidates)}/{len(neurons)} pinned neurons are candidates here")
    else:
        candidates = np.where(live)[0]
    print(f"{len(candidates)} candidates (corpus fr in band; "
          f"{int((fr_ok & ~enough).sum())} otherwise-eligible neurons dropped for firing on fewer "
          f"than {n_corpus_snippets} corpus"
          + (f" / {n_query_snippets} query" if Zc_q is not None else "") + " rows)")

    if entropy_source:  # ascending entropy = most theme-pure first
        order = candidates[np.argsort(ent_norm[candidates])]
        n_tied_at_cut = warn_if_tied_ranking(ent_norm, order, max_neurons,
                                             f"{entropy_source} theme entropy",
                                             "--top-n (omit it to score on every doc the neuron fires on)")
    else:
        order, n_tied_at_cut = candidates, 0
    # cols_corpus, not cols_id: this file also takes an --id-col.
    cols_corpus = neuron_fired_columns(Zc)
    cols_q = neuron_fired_columns(Zc_q) if Zc_q is not None else None

    # One TF-IDF space over every text a neuron's terms can come from, corpus rows then query
    # rows. Fit once, for --diversity-overlap's key and the recorded "top_terms".
    q_texts = q_df[query_text_col].astype(str).values if q_df is not None else np.empty(0, dtype=str)
    tfidf, vocab = fit_tfidf(np.concatenate([corpus_df[text_col].astype(str).values, q_texts]))
    n_corpus_rows = Zc.shape[0]

    # One evidence set per neuron.
    evidence_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    dropped_q: dict[int, int] = {}

    def evidence(n):
        """(corpus rows, query rows) -- the neuron's strongest."""
        if n not in evidence_cache:
            c_idx = _evidence_idx(cols_corpus, corpus_df, n, n_corpus_snippets,
                                  snippet_distinct_col)
            q_idx = (neuron_top_activating(cols_q, n, n_query_snippets) if cols_q is not None
                     else np.array([], dtype=int))
            # Query firing is not a selection condition. Judged per neuron.
            if query_min_act_frac is not None and len(c_idx) and len(q_idx):
                floor = query_min_act_frac * _column(cols_corpus, n)[c_idx].max()
                keep = _column(cols_q, n)[q_idx] >= floor
                dropped_q[n] = int((~keep).sum())
                q_idx = q_idx[keep]
            evidence_cache[n] = (c_idx, q_idx)
        return evidence_cache[n]

    def terms_of(n):
        c_idx, q_idx = evidence(n)
        mask = np.zeros(n_corpus_rows + len(q_texts), dtype=bool)
        mask[c_idx] = True
        mask[n_corpus_rows + q_idx] = True
        return top_terms_for_mask(tfidf, vocab, mask, n_terms)

    kept, skipped = select_diverse_neurons(
        order, lambda n: {t.strip() for t in terms_of(n).split(",")}, max_neurons, diversity_overlap)
    rank_desc = (f"lowest {entropy_source} theme entropy (top_n={top_n if top_n else 'all firing'})"
                 if entropy_source else "candidate order")
    report_selection(kept, skipped, len(candidates), rank_desc, diversity_overlap, "top-terms")

    n_no_query_snippets = 0
    neurons_out = []
    for n in kept:
        dom_idx = (int(np.argmax(theme_dist[n]))
                   if theme_dist is not None and theme_dist[n].sum() > 0 else None)
        c_idx, q_idx = evidence(n)
        corpus_snippets = _snippets(cols_corpus, corpus_df, text_col, id_col, n,
                                    n_corpus_snippets, base_text_col, idx=c_idx)
        query_snippets = (_snippets(cols_q, q_df, query_text_col, query_id_col, n,
                                    n_query_snippets, base_query_text_col, idx=q_idx)
                          if cols_q is not None else [])
        if query_files and not query_snippets:
            n_no_query_snippets += 1
        neurons_out.append({
            "neuron": int(n),
            "firing_rate_corpus": float(fr_corpus[n]),
            "firing_rate_query": float(fr_query[n]) if fr_query is not None else None,
            "firing_rate_ood_mean": float(fr_ood_mean[n]) if fr_ood_mean is not None else None,
            "top_terms": terms_of(n),
            # None, not NaN: NaN is not JSON.
            "entropy": _or_none(ent[n]) if ent is not None else None,
            "entropy_norm": _or_none(ent_norm[n]) if ent_norm is not None else None,
            "entropy_source": entropy_source, "top_n": top_n,
            "n_fired_entropy_source": (int(n_fired_entropy_source[n])
                                       if n_fired_entropy_source is not None else None),
            "dominant_theme": theme_labels[dom_idx] if dom_idx is not None else None,
            "theme_dist": ({theme_labels[i]: float(p) for i, p in enumerate(theme_dist[n]) if p > 0}
                           if theme_dist is not None else {}),
            "corpus_snippets": corpus_snippets,
            "query_snippets": query_snippets,
            "negative_snippets": _negative_snippets(cols_corpus, corpus_df, text_col, id_col, n,
                                                    DEFAULT_N_NEGATIVE_SNIPPETS, base_text_col),
        })
    if query_files and n_no_query_snippets:
        print(f"{n_no_query_snippets} / {len(kept)} kept neurons have 0 query snippets "
              f"(corpus fr band is the only selection filter — query firing is not required)")
    dropped_kept = {n: d for n, d in dropped_q.items() if d and n in set(kept)}
    if dropped_kept:
        detail = ", ".join(f"#{n}: {d}" for n, d in dropped_kept.items())
        print(f"dropped {sum(dropped_kept.values())} query snippets below "
              f"{query_min_act_frac:.0%} of their neuron's top corpus activation ({detail})")

    return {
        "sae_corpus_file": str(sae_corpus_file), "queries_files": [str(f) for f in query_files],
        "theme_col": theme_col, "query_theme_col": query_theme_col,
        "fr_min": lo, "fr_max": hi, "ood_datasets": ood_names, "entropy_source": entropy_source,
        "top_n": top_n,
        # n_hidden recorded too.
        "n_hidden": int(Zc.shape[1]), "n_candidates": len(candidates), "n_kept": len(kept),
        "n_corpus_snippets": n_corpus_snippets, "n_query_snippets": n_query_snippets,
        "n_negative_snippets": DEFAULT_N_NEGATIVE_SNIPPETS,
        "query_min_act_frac": query_min_act_frac, "n_pinned": len(neurons) if neurons else None,
        # Did the ranking separate the kept neurons?
        "n_tied_at_cut": int(n_tied_at_cut),
        "n_distinct_entropy_kept": (int(len(set(np.round(ent_norm[kept], 6))))
                                    if ent_norm is not None and len(kept) else 0),
        "neurons": neurons_out,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sae-corpus-file", required=True, type=Path,
                   help="convert_embeddings.py '*_with_sae.pkl' (has the 'sae embeddings' column)")
    p.add_argument("--corpus-file", type=Path,
                   help="Row-aligned corpus pkl with text/theme columns, if --sae-corpus-file doesn't carry them")
    p.add_argument("--id-col", help="Row-alignment check between --sae-corpus-file and --corpus-file, "
                                    "and the id recorded per snippet instead of a row index")
    p.add_argument("--text-col", default="text", help="Corpus text column, overridden by "
                                                     "'_prepended_text' where the embedder wrote one")
    p.add_argument("--theme-col", default=None,
                   help="Native corpus theme column (required for --entropy-theme corpus)")
    p.add_argument("--hidden", type=int, default=None,
                   help="Override the SAE hidden size for sparse sae-embeddings (default: read "
                        "from the pkl's attrs['sae_hidden'])")
    p.add_argument("--queries-file", type=Path, nargs="+", default=None,
                   help="In-domain queries' '*_with_sae.pkl' (same checkpoint), pooled if several — "
                        "required for --entropy-theme queries, optional otherwise (adds query "
                        "snippets/firing rate)")
    p.add_argument("--query-text-col", default=None, help="Queries text column (default: --text-col)")
    p.add_argument("--query-id-col", default=None,
                   help="Id recorded per query snippet instead of a row index (e.g. _id)")
    p.add_argument("--query-theme-col", default=None,
                   help="Native queries theme column (required for --entropy-theme queries)")
    p.add_argument("--fr-min", type=float, default=0.01,
                   help="Lower bound for corpus firing rate, and mean OOD firing rate if --ood-dir is given")
    p.add_argument("--fr-max", type=float, default=0.20,
                   help="Upper bound for corpus firing rate, and mean OOD firing rate if --ood-dir is given")
    p.add_argument("--dataset-name", default=None,
                   help="In-domain dataset name (a src.utils.datasets_constants name, e.g. CURE) — "
                        "required by --ood-dir, to resolve ood_datasets_for(dataset_name)")
    p.add_argument("--ood-dir", type=Path, default=None,
                   help="Directory holding {name}/corpus*_with_sae.pkl for each OOD dataset in "
                        "ood_datasets_for(--dataset-name) — omit for corpus-only (no OOD firing-rate condition)")
    p.add_argument("--exclude", nargs="+", default=[], help="OOD dataset names to drop")
    p.add_argument("--entropy-theme", dest="entropy_source", choices=["corpus", "queries"],
                   default=None,
                   help="Compute theme entropy from corpus or query firing examples, and rank "
                        "candidates by it. Omit and no entropy is computed")
    p.add_argument("--top-n", type=int, default=None,
                   help="Restrict entropy to each neuron's top-N highest-activation examples of "
                        "--entropy-theme. Omit (default) for every example the neuron fires on at "
                        "all — a small N next to the fired sets --fr-min guarantees saturates the "
                        "ranking, tying candidates at entropy 0 regardless of their real purity.")
    p.add_argument("--max-neurons", type=int, default=None,
                   help="Keep this many candidates, in ranked order (default: every candidate)")
    p.add_argument("--neurons", nargs="+", type=int, default=None,
                   help="Profile only these neurons, in this order (default: every neuron). Those "
                        "failing the firing-rate band or the evidence minimums are still dropped")
    p.add_argument("--n-corpus-snippets", type=int, default=DEFAULT_N_CORPUS_SNIPPETS,
                   help=f"Top-activating corpus passages pulled per kept neuron "
                        f"(default: {DEFAULT_N_CORPUS_SNIPPETS})")
    p.add_argument("--n-query-snippets", type=int, default=DEFAULT_N_QUERY_SNIPPETS,
                   help=f"Top-activating queries pulled per kept neuron "
                        f"(default: {DEFAULT_N_QUERY_SNIPPETS})")
    p.add_argument("--snippet-distinct-col", default="title",
                   help="At most {} corpus snippets per distinct value of this column (default "
                        "'title'). '' to disable".format(MAX_CHUNKS_PER_DOC))
    p.add_argument("--query-min-act-frac", type=float, default=None,
                   help="Drop query snippets activating below this fraction of the neuron's top "
                        "corpus activation, e.g. 0.2 (default: keep all)")
    p.add_argument("--diversity-overlap", type=float, default=None,
                   help="Off by default. Skip a candidate at this Jaccard (e.g. 0.5) or more")
    p.add_argument("--n-terms", type=int, default=5,
                   help="TF-IDF terms recorded per neuron, and --diversity-overlap's key (default 5)")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    if args.entropy_source == "corpus" and not args.theme_col:
        p.error("--entropy-theme corpus requires --theme-col")
    if args.entropy_source == "queries" and not (args.queries_file and args.query_theme_col):
        p.error("--entropy-theme queries requires --queries-file and --query-theme-col")
    if args.ood_dir and not args.dataset_name:
        p.error("--ood-dir requires --dataset-name")

    result = select_neurons(args.sae_corpus_file, args.text_col, args.theme_col, args.corpus_file,
                            args.id_col, args.hidden, args.queries_file, args.query_text_col,
                            args.query_id_col, args.query_theme_col,
                            (args.fr_min, args.fr_max), args.entropy_source,
                            args.top_n, args.max_neurons, args.n_corpus_snippets,
                            args.n_query_snippets, args.query_min_act_frac, args.neurons,
                            args.snippet_distinct_col or None,
                            args.diversity_overlap,
                            args.n_terms, args.dataset_name,
                            args.ood_dir, args.exclude)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.output, "w"), indent=2)
    print(f"wrote {args.output} ({result['n_kept']} neurons)")


if __name__ == "__main__":
    main()
