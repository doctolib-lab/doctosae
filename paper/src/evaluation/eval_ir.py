"""Compute an IR metric (NDCG@k or mean per-query AUC) on a queries pkl + one-or-more corpus pkls against a qrels TSV.
Defaults are the relevance column and query_id col; pass --score-col=score --query-id-col=_id for BeIR.

Pass every corpus split the qrels can point at: one --corpus-file for an undivided OOD corpus,
--corpus-dir for a train/val split (loads corpus_train.pkl and corpus_val.pkl, warning if only one is
present). Rows are concatenated in the order given, then deduped by --corpus-id-col keeping the
first occurrence.

Example (in-domain, train/val split — both needed):
         python -m src.evaluation.eval_ir --queries-file queries.pkl --corpus-dir ~/data/cure-beir/bge-large-en-v1.5 --qrels-file qrels.tsv --metric ndcg
Example (OOD target, one corpus):
         python -m src.evaluation.eval_ir --queries-file queries.pkl --corpus-file corpus.pkl --qrels-file qrels.tsv --metric auc
Example (hybrid sparse+dense, weighted Reciprocal Rank Fusion of two embedding columns):
         python -m src.evaluation.eval_ir --queries-file queries_with_sae.pkl --corpus-file corpus_with_sae.pkl --qrels-file qrels.tsv \\
             --emb-type "sae embeddings" --emb-type-b "reconstructed embeddings" --alpha 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.utils.cli import expand_path_args
from src.utils.metrics import (cosine_scores, load_qrels_matrix, ndcg_at_k, mean_auc,
                              precision_recall_at_k, rrf_scores)


def _is_sparse_cell(v) -> bool:
    """True if an embedding cell is a (indices, values) sparse tuple (see
    convert_embeddings.compute_activations(sparse=True)) rather than a dense vector."""
    return isinstance(v, tuple) and len(v) == 2


def _hidden_dim(q_vals: np.ndarray, m_vals: np.ndarray) -> int:
    """Shared embedding width for Q/M: dense vector length if either side is dense (ground
    truth), else max sparse index + 1 across both sides."""
    q_sparse, m_sparse = _is_sparse_cell(q_vals[0]), _is_sparse_cell(m_vals[0])
    if not q_sparse:
        return len(np.asarray(q_vals[0], dtype=np.float32))
    if not m_sparse:
        return len(np.asarray(m_vals[0], dtype=np.float32))
    max_idx = max(
        max((int(idx.max()) for idx, _ in q_vals if len(idx)), default=-1),
        max((int(idx.max()) for idx, _ in m_vals if len(idx)), default=-1),
    )
    return max_idx + 1


def stack_embeddings(vals: np.ndarray, hidden_dim: int):
    """Stack an embedding column into an (N, hidden_dim) matrix: dense via np.vstack, or a
    csr_matrix if cells are (indices, values) sparse tuples."""
    if not _is_sparse_cell(vals[0]):
        return np.vstack(vals).astype(np.float32)
    indptr = [0]
    indices, data = [], []
    for idx, val in vals:
        indices.append(np.asarray(idx, dtype=np.int32))
        data.append(np.asarray(val, dtype=np.float32))
        indptr.append(indptr[-1] + len(idx))
    return sp.csr_matrix((np.concatenate(data), np.concatenate(indices), indptr),
                         shape=(len(vals), hidden_dim), dtype=np.float32)


def load_pickle_or_dir(path: str, keep_cols: list | None = None) -> pd.DataFrame:
    """Load a single pkl or concatenate every part file in a directory; optionally keep only some columns."""
    def _load(p):
        df = pd.read_pickle(p)
        if keep_cols is not None:
            df = df[[c for c in keep_cols if c in df.columns]]
        return df

    if os.path.isdir(path):
        parts = sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if not f.startswith(".") and not f.endswith(".json")
        )
        print(f"Loading {len(parts)} parts from {path}")
        df = pd.concat([_load(p) for p in parts], ignore_index=True)
        print(f"Loaded {len(df)} rows")
        return df
    return _load(path)


def main() -> None:
    """CLI: load queries + corpus + qrels and print the chosen metric for the chosen embedding column."""
    p = argparse.ArgumentParser()
    p.add_argument("--queries-file", required=True, type=Path,
                   help="Pickle with 'id' and embedding columns.")
    corpus_group = p.add_mutually_exclusive_group(required=True)
    corpus_group.add_argument("--corpus-file", type=Path, nargs="+",
                              help="One or more corpus pickles (or a directory of part files); rows are "
                                   "concatenated. One path is right for a single-corpus OOD target; for "
                                   "a train/val-split dataset list every split the qrels can reference "
                                   "(e.g. corpus_train.pkl corpus_val.pkl), else the positives held in "
                                   "the split you left out are missed and the metric understated.")
    corpus_group.add_argument("--corpus-dir", type=Path,
                              help="Directory containing corpus_train.pkl and/or corpus_val.pkl; both are loaded "
                                   "(warning if only one is found) and concatenated.")
    p.add_argument("--qrels-file", required=True, type=Path,
                   help="TSV with relevance scores; AUC requires both positives and negatives.")
    p.add_argument("--metric", choices=["ndcg", "auc", "precision_recall"], default="ndcg",
                   help="ndcg@k (default), mean per-query AUC over judged chunks, or precision/recall@k.")
    p.add_argument("--emb-type", default="embeddings",
                   help="Embedding column to score on (e.g. 'embeddings' or 'sae embeddings').")
    p.add_argument("--emb-type-b", default=None,
                   help="Second embedding column; when set, the two rankings are fused by weighted "
                        "Reciprocal Rank Fusion (see src.utils.metrics.rrf_scores) instead of scoring "
                        "--emb-type alone. E.g. --emb-type 'sae embeddings' --emb-type-b "
                        "'reconstructed embeddings' to combine sparse and dense retrieval. Both "
                        "columns must exist in the same queries/corpus pkls.")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="RRF weight on --emb-type; --emb-type-b gets 1-alpha (default 0.5, equal). "
                        "1.0 recovers --emb-type's own ranking, 0.0 --emb-type-b's.")
    p.add_argument("--rrf-k", type=int, default=60,
                   help="RRF rank-damping constant (default 60, the value from Cormack et al. 2009).")
    p.add_argument("--k", type=int, nargs="+", default=[10],
                   help="Cutoff(s) for NDCG@k or precision/recall@k (ignored when --metric=auc). E.g. --k 10 20.")
    p.add_argument("--corpus-id-col", default="id",
                   help="Column in the corpus pkl that uniquely identifies each row. "
                        "Must match the qrels 'corpus-id'.")
    p.add_argument("--query-id-col", default="query_id",
                   help="Column in the queries pkl that uniquely identifies each row. "
                        "Must match the qrels 'query-id'.")
    p.add_argument("--score-col", default="relevance",
                   help="Column name in the qrels TSV holding the relevance score (NFC uses 'score').")
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--max-queries-seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=None,
                   help="Optional JSON output path for the score.")
    p.add_argument("--auc-hard-negs-only", action="store_true",
                   help="With --metric auc, drop plain negatives before scoring (keep only positives + "
                        "is_hard_negative=True rows). Requires an is_hard_negative column in the qrels TSV.")
    p.add_argument("--exclude-train-positives", type=Path, default=None,
                   help="Trainval qrels TSV; drops positive rows from the eval qrels whose corpus-id is a "
                        "positive (score>0) for any row in this file. Negatives are untouched. Use to measure "
                        "performance only on chunks never seen as positive during training.")
    p.add_argument("--output-per-query", type=Path, default=None,
                   help="If set, write a JSONL with one line per query: {query_id, score, top_k: [{corpus_id, similarity}, ...]}. "
                        "score = per-query AUC or NDCG@k; top_k uses --top-k.")
    p.add_argument("--top-k", type=int, default=50,
                   help="Number of top-scoring corpus rows to dump per query when --output-per-query is set.")
    args = expand_path_args(p.parse_args())
    if args.auc_hard_negs_only and args.metric != "auc":
        raise ValueError("--auc-hard-negs-only only applies with --metric auc.")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError(f"--alpha must be in [0, 1], got {args.alpha}.")
    # Every embedding column to load; the second one only to fuse with RRF.
    emb_types = [args.emb_type] + ([args.emb_type_b] if args.emb_type_b else [])

    # --emb-type "sae embeddings" reads *_with_sae.pkl; else the raw pkls.
    if args.corpus_dir is not None:
        # convert_embeddings writes both columns into the same *_with_sae.pkl.
        suffix = "_with_sae" if any("sae" in e or "reconstructed" in e for e in emb_types) else ""
        train_name = f"corpus_train{suffix}.pkl"
        val_name   = f"corpus_val{suffix}.pkl"
        train_path = args.corpus_dir / train_name
        val_path   = args.corpus_dir / val_name
        if not train_path.exists():
            warnings.warn(f"--corpus-dir {args.corpus_dir}: {train_name} missing.")
            train_path = None
        if not val_path.exists():
            warnings.warn(f"--corpus-dir {args.corpus_dir}: {val_name} missing.")
            val_path = None
        if train_path is None and val_path is None:
            raise ValueError(f"--corpus-dir {args.corpus_dir}: neither {train_name} nor {val_name} found.")
        keep = emb_types + [args.corpus_id_col]
        dfs = []
        if train_path is not None:
            dfs.append(load_pickle_or_dir(str(train_path), keep_cols=keep))
        if val_path is not None:
            dfs.append(load_pickle_or_dir(str(val_path),   keep_cols=keep))
        df_corpus = pd.concat(dfs, ignore_index=True)
        corpus_source = {"corpus_dir": str(args.corpus_dir),
                         "corpus_train": str(train_path) if train_path is not None else None,
                         "corpus_val":   str(val_path)   if val_path   is not None else None}
    else:
        df_corpus = pd.concat(
            [load_pickle_or_dir(str(p_), keep_cols=emb_types + [args.corpus_id_col]) for p_ in args.corpus_file],
            ignore_index=True,
        )
        corpus_source = {"corpus_file": [str(p_) for p_ in args.corpus_file]}
    for e in emb_types:
        if e not in df_corpus.columns:
            raise ValueError(f"Corpus missing column '{e}'. Available: {list(df_corpus.columns)}")
    n_before = len(df_corpus)
    df_corpus = df_corpus.drop_duplicates(subset=args.corpus_id_col, keep="first")
    n_dropped = n_before - len(df_corpus)
    if n_dropped:
        warnings.warn(f"dropped {n_dropped} duplicate corpus rows by '{args.corpus_id_col}'.")
    df_corpus = df_corpus.set_index(args.corpus_id_col)
    df_corpus.index = df_corpus.index.astype(str)

    # Queries
    df_q = pd.read_pickle(args.queries_file)
    if args.query_id_col not in df_q.columns:
        raise ValueError(f"--query-id-col {args.query_id_col!r} not in columns: {list(df_q.columns)}")
    df_q["id"] = df_q[args.query_id_col].astype(str)
    missing_q = [e for e in emb_types if e not in df_q.columns]
    if missing_q:
        raise ValueError(f"Queries missing column(s) {missing_q}. Available: {list(df_q.columns)}")
    df_q = df_q[["id"] + emb_types].set_index("id")
    if args.max_queries and len(df_q) > args.max_queries:
        df_q = df_q.sample(n=args.max_queries, random_state=args.max_queries_seed)

    # Dense, or csr_matrix for large-hidden TopK codes.
    mats = {}
    for e in emb_types:
        hd = _hidden_dim(df_q[e].values, df_corpus[e].values)
        mats[e] = (stack_embeddings(df_q[e].values, hd), stack_embeddings(df_corpus[e].values, hd))
    Q, M = mats[args.emb_type]

    # One ranking, or RRF fusion of both.
    S_fused = None
    if args.emb_type_b:
        row_ids_all = np.asarray(df_corpus.index.tolist())
        S_fused = rrf_scores([cosine_scores(*mats[e], row_ids_all)[0] for e in emb_types],
                            weights=[args.alpha, 1.0 - args.alpha], k=args.rrf_k)

    # Load qrels once; apply --exclude-train-positives for both metrics.
    qrels_df = pd.read_csv(args.qrels_file, sep="\t")
    if args.score_col not in qrels_df.columns:
        raise ValueError(f"{args.qrels_file}: score column {args.score_col!r} not in {list(qrels_df.columns)}")
    if args.exclude_train_positives is not None:
        tv = pd.read_csv(args.exclude_train_positives, sep="\t")
        train_pos = set(tv.loc[tv[args.score_col] > 0, "corpus-id"].astype(str))
        is_pos = qrels_df[args.score_col] > 0
        is_seen = qrels_df["corpus-id"].astype(str).isin(train_pos)
        drop = is_pos & is_seen
        n_drop = int(drop.sum())
        qrels_df = qrels_df[~drop]
        warnings.warn(f"--exclude-train-positives: dropped {n_drop} positive rows whose corpus-id was train-positive "
                      f"(train_pos set size = {len(train_pos)}).")

    # What the printed lines call the scorer.
    scored = args.emb_type if not args.emb_type_b else \
        f"RRF(alpha={args.alpha}) {args.emb_type} + {args.emb_type_b}"

    per_query_scores: dict[str, float] | None = None
    if args.metric == "ndcg":
        k = args.k[0]
        row_ids = np.asarray(df_corpus.index.tolist())
        qp = load_qrels_matrix(df_q.reset_index(), qrels_df, df_corpus, score_col=args.score_col)
        if args.output_per_query is not None:
            score, per_q = ndcg_at_k(Q, M, row_ids, qp, k=k, return_per_query=True, S=S_fused)
            per_query_scores = {str(qid): float(v) for qid, v in zip(df_q.index.tolist(), per_q)}
        else:
            score = ndcg_at_k(Q, M, row_ids, qp, k=k, S=S_fused)
        print(f"ndcg@{k}  ({scored})  = {score:.4f}")
        stats = {}
    elif args.metric == "precision_recall":
        row_ids = np.asarray(df_corpus.index.tolist())
        qid_to_row = {qid: i for i, qid in enumerate(df_q.index.tolist())}
        pr = precision_recall_at_k(Q, M, row_ids, qid_to_row, qrels_df, ks=args.k,
                                  score_col=args.score_col, S=S_fused)
        for k, vals in sorted(pr.items()):
            print(f"precision@{k}  ({scored}) = {vals['precision']:.4f}   "
                  f"recall@{k} = {vals['recall']:.4f}   (n_queries={vals['n_queries']})")
        score = pr[args.k[0]]["precision"]
        stats = {f"p@{k}": v["precision"] for k, v in pr.items()}
        stats.update({f"r@{k}": v["recall"] for k, v in pr.items()})
    else:  # auc
        qid_to_row = {qid: i for i, qid in enumerate(df_q.index.tolist())}
        cid_to_row = {cid: i for i, cid in enumerate(df_corpus.index.tolist())}
        if args.auc_hard_negs_only:
            if "is_hard_negative" not in qrels_df.columns:
                raise ValueError(f"{args.qrels_file}: --auc-hard-negs-only requires an 'is_hard_negative' column.")
            keep = (qrels_df[args.score_col] > 0) | qrels_df["is_hard_negative"].astype(bool)
            n_dropped = (~keep).sum()
            qrels_df = qrels_df[keep]
            if n_dropped:
                warnings.warn(f"--auc-hard-negs-only: dropped {n_dropped} plain-negative qrels rows.")
        want_pq = args.output_per_query is not None
        score, stats = mean_auc(Q, M, qid_to_row, cid_to_row, qrels_df, score_col=args.score_col,
                                return_per_query=want_pq, S=S_fused)
        if want_pq:
            per_query_scores = stats.pop("per_query")
        print(f"mean per-query AUC ({scored}) = {score:.4f}   "
              f"(scored: {stats['n_queries_scored']}, skipped no-pos: {stats['skipped_no_pos']}, skipped no-neg: {stats['skipped_no_neg']})")

    # Optional per-query JSONL with top-K corpus rows by score (cosine, or the RRF fusion).
    if args.output_per_query is not None:
        S = S_fused if S_fused is not None else cosine_scores(Q, M, np.asarray(df_corpus.index.tolist()))[0]
        K = min(args.top_k, S.shape[1])
        top_idx = np.argpartition(-S, K - 1, axis=1)[:, :K]
        # sort within top-K by descending similarity
        row_ix = np.arange(S.shape[0])[:, None]
        ord_in = np.argsort(-S[row_ix, top_idx], axis=1)
        top_idx = top_idx[row_ix, ord_in]
        corpus_ids = df_corpus.index.tolist()
        query_ids = df_q.index.tolist()
        args.output_per_query.parent.mkdir(parents=True, exist_ok=True)
        with args.output_per_query.open("w") as f:
            for i, qid in enumerate(query_ids):
                rec = {
                    "query_id": str(qid),
                    "score": per_query_scores.get(str(qid)) if per_query_scores is not None else None,
                    "top_k": [{"corpus_id": str(corpus_ids[int(j)]), "similarity": float(S[i, int(j)])}
                              for j in top_idx[i]],
                }
                f.write(json.dumps(rec) + "\n")
        print(f"Wrote per-query top-{K} to {args.output_per_query}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "metric": args.metric,
            args.metric: float(score),
            **({"k": args.k} if args.metric == "ndcg" else {}),
            **stats,
            "emb_type": args.emb_type,
            **({"emb_type_b": args.emb_type_b, "fusion": "rrf",
                "alpha": args.alpha, "rrf_k": args.rrf_k} if args.emb_type_b else {}),
            "queries_file": str(args.queries_file),
            **corpus_source,
            "qrels_file": str(args.qrels_file),
            "n_queries": int(len(df_q)),
            "n_corpus": int(len(df_corpus)),
        }, indent=2))


if __name__ == "__main__":
    main()
