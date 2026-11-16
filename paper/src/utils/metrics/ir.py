"""IR metrics shared across training and eval.
load_qrels_matrix reshapes a qrels TSV into a (queries × corpus) score matrix; ndcg_at_k computes NDCG@k after cosine_scores (which requires unique row_ids).
"""
import warnings
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn.metrics
from sklearn.metrics.pairwise import cosine_similarity


def _normalize_sparse_rows(M: sp.csr_matrix) -> sp.csr_matrix:
    """L2-normalize each row of a sparse matrix; zero rows are left as zero (no div-by-zero)."""
    norms = np.sqrt(M.multiply(M).sum(axis=1)).A.ravel()
    norms[norms == 0] = 1.0
    return sp.diags(1.0 / norms) @ M


def load_qrels_matrix(
    queries_df: pd.DataFrame,
    qrels_path: Union[str, Path, pd.DataFrame],
    corpus_df: pd.DataFrame,
    score_col: str = "relevance",
) -> pd.DataFrame:
    """Load a qrels TSV (or pre-loaded DataFrame) and reshape into a (queries × corpus) relevance matrix; ``score_col`` defaults to 'relevance' (pass 'score' for NFC-style qrels)."""
    if isinstance(qrels_path, pd.DataFrame):
        qr = qrels_path.copy()
    else:
        qr = pd.read_csv(qrels_path, sep="\t")
        if qr.shape[1] < 3:
            qr = pd.read_csv(qrels_path, sep=r"\s+", engine="python")
    qr["query-id"] = qr["query-id"].astype(str)
    qr["corpus-id"] = qr["corpus-id"].astype(str)
    if score_col not in qr.columns:
        raise ValueError(f"{qrels_path}: score column {score_col!r} not in columns {list(qr.columns)}")
    ids = queries_df["id"].astype(str).tolist()
    qr = qr[qr["query-id"].isin(ids)]
    qp = qr.pivot(index="query-id", columns="corpus-id", values=score_col).fillna(0)
    miss = [c for c in corpus_df.index if c not in qp.columns]
    if miss:
        qp = pd.concat(
            [qp, pd.DataFrame(0, columns=np.unique(miss), index=qp.index)], axis=1
        )
    return qp.reindex(index=ids, columns=corpus_df.index.tolist()).fillna(0)


# ---------------------------------------------------------------------------
# Similarity + NDCG
# ---------------------------------------------------------------------------

def cosine_scores(
    Q, M, row_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Cosine(Q, M); raises if ``row_ids`` has duplicates. Q/M may be dense ``np.ndarray`` or sparse
    ``scipy.sparse``; sparse inputs are never densified, only the returned score matrix is."""
    uniq, counts = np.unique(row_ids, return_counts=True)
    dups = uniq[counts > 1]
    if dups.size:
        raise ValueError(
            f"row_ids has {dups.size} duplicate id(s) (e.g. {dups[:3].tolist()!r}); "
            f"eval expects one corpus row per id."
        )
    if sp.issparse(Q) or sp.issparse(M):
        Qn = _normalize_sparse_rows(sp.csr_matrix(Q))
        Mn = _normalize_sparse_rows(sp.csr_matrix(M))
        S = np.asarray((Qn @ Mn.T).todense(), dtype=np.float32)
    else:
        S = cosine_similarity(Q, M).astype(np.float32)
    return S, np.asarray(row_ids)


def average_ranks(S: np.ndarray) -> np.ndarray:
    """Row-wise ranks of ``S`` descending (rank 1 = the row's highest score), ties averaged over the
    ranks their group spans. When sweeping fusion weights, call this once and pass ``ranked=True``
    to rrf_scores.
    """
    n = S.shape[1]
    order = np.argsort(-S, axis=1, kind="stable")
    S_sorted = np.take_along_axis(S, order, axis=1)
    pos = np.arange(1, n + 1, dtype=np.float32)          # exact in float32 up to 2**24
    starts_group = np.ones(S.shape, dtype=bool)
    starts_group[:, 1:] = S_sorted[:, 1:] != S_sorted[:, :-1]
    ends_group = np.ones(S.shape, dtype=bool)
    ends_group[:, :-1] = starts_group[:, 1:]
    # First and last rank of each tie group, broadcast across it.
    first = np.maximum.accumulate(np.where(starts_group, pos, 0.0), axis=1)
    last = np.minimum.accumulate(np.where(ends_group, pos, np.inf)[:, ::-1], axis=1)[:, ::-1]
    ranks = np.empty(S.shape, dtype=np.float32)
    np.put_along_axis(ranks, order, (first + last) / 2.0, axis=1)
    return ranks


def rrf_scores(score_mats: list, weights: list = None, k: int = 60, ranked: bool = False) -> np.ndarray:
    """Weighted Reciprocal Rank Fusion (Cormack et al. 2009): ``RRF(q, d) = sum_r w_r / (k + rank_r)``.
    Ties share their group's average rank. ``score_mats`` may hold score matrices, or rank matrices
    with ``ranked=True``.
    """
    if weights is None:
        weights = [1.0] * len(score_mats)
    if len(weights) != len(score_mats):
        raise ValueError(f"{len(score_mats)} score matrices but {len(weights)} weights.")
    fused = np.zeros(score_mats[0].shape, dtype=np.float32)
    for S, w in zip(score_mats, weights):
        fused += w / (k + (S if ranked else average_ranks(S)))
    return fused


def ndcg_at_k(
    Q: np.ndarray,
    M: np.ndarray,
    row_ids: np.ndarray,
    qp: pd.DataFrame,
    k: int = 10,
    return_per_query: bool = False,
    S: np.ndarray = None,
):
    """NDCG@k of Q vs M against a qrels matrix; aggregates rows in M by row_ids before scoring."""
    if S is None:
        S_doc, uniq = cosine_scores(Q, M, row_ids)
    else:
        if len(np.unique(row_ids)) != len(row_ids):
            warnings.warn("row_ids has duplicates but S is pre-computed — duplicate corpus rows will be silently ignored", stacklevel=2)
        S_doc, uniq = S, np.asarray(row_ids)
    gt = (
        qp.T.groupby(level=0)
        .max()
        .T.reindex(columns=uniq, fill_value=0)
        .values.astype(np.float32)
    )
    has = gt.sum(axis=1) > 0
    if not has.any():
        return (0.0, np.zeros(S_doc.shape[0], dtype=np.float32)) if return_per_query else 0.0
    mean_val = float(sklearn.metrics.ndcg_score(gt[has], S_doc[has], k=k))
    if not return_per_query:
        return mean_val
    per_q = np.zeros(S_doc.shape[0], dtype=np.float32)
    for i in range(S_doc.shape[0]):
        if has[i]:
            per_q[i] = float(sklearn.metrics.ndcg_score(gt[i : i + 1], S_doc[i : i + 1], k=k))
    return mean_val, per_q


# ---------------------------------------------------------------------------
# Precision / Recall @k (judged positives only)
# ---------------------------------------------------------------------------

def precision_recall_at_k(
    Q: np.ndarray,
    M: np.ndarray,
    row_ids: np.ndarray,
    qid_to_row: dict,
    qrels_df: pd.DataFrame,
    ks: list[int],
    score_col: str = "relevance",
    S: np.ndarray = None,
) -> dict[int, dict[str, float]]:
    """Mean precision@k and recall@k over queries with at least one positive. ``qid_to_row`` maps each
    query-id to its row in ``Q``. Pass a precomputed ``S`` to skip cosine_similarity. Returns
    {k: {precision, recall, n_queries}}.
    """
    if S is None:
        S, _ = cosine_scores(Q, M, row_ids)
    id_to_col = {str(cid): i for i, cid in enumerate(row_ids)}
    qr = qrels_df.copy()
    qr["query-id"] = qr["query-id"].astype(str)
    qr["corpus-id"] = qr["corpus-id"].astype(str)

    results: dict[int, dict[str, float]] = {}
    for k in ks:
        top_k_cols = np.argpartition(-S, min(k, S.shape[1]) - 1, axis=1)[:, :k]
        # sort within top-k
        row_ix = np.arange(S.shape[0])[:, None]
        top_k_cols = top_k_cols[row_ix, np.argsort(-S[row_ix, top_k_cols], axis=1)]
        top_k_sets = [set(row) for row in top_k_cols]

        precs, recs = [], []
        for qid, g in qr.groupby("query-id"):
            if qid not in qid_to_row:
                continue
            pos_cids = set(g.loc[g[score_col] > 0, "corpus-id"])
            if not pos_cids:
                continue
            pos_col_idx = {id_to_col[c] for c in pos_cids if c in id_to_col}
            tk = top_k_sets[qid_to_row[qid]]
            hits = len(pos_col_idx & tk)
            precs.append(hits / k)
            recs.append(hits / len(pos_cids))
        results[k] = {
            "precision": float(np.mean(precs)) if precs else float("nan"),
            "recall":    float(np.mean(recs))  if recs  else float("nan"),
            "n_queries": len(precs),
        }
    return results


# ---------------------------------------------------------------------------
# AUC (judged-only)
# ---------------------------------------------------------------------------

def mean_auc(
    Q: np.ndarray,
    M: np.ndarray,
    qid_to_row: dict,
    cid_to_row: dict,
    qrels_df: pd.DataFrame,
    score_col: str = "relevance",
    return_per_query: bool = False,
    S: np.ndarray = None,
) -> tuple[float, dict]:
    """Per-query AUC of cosine(Q, M) over judged chunks (relevance>0 positive, <=0 negative), averaged
    across queries. Pass a precomputed ``S`` to skip cosine_similarity; ``return_per_query`` adds a
    ``per_query`` dict."""
    if S is None:
        if len(cid_to_row) != M.shape[0]:
            raise ValueError(
                f"cid_to_row has {len(cid_to_row)} entries but M has {M.shape[0]} rows "
                f"(duplicate corpus ids collapsed silently); eval expects one row per id."
            )
        if sp.issparse(Q) or sp.issparse(M):
            Qn = _normalize_sparse_rows(sp.csr_matrix(Q))
            Mn = _normalize_sparse_rows(sp.csr_matrix(M))
            S = np.asarray((Qn @ Mn.T).todense(), dtype=np.float32)
        else:
            S = cosine_similarity(Q, M).astype(np.float32)
    qr = qrels_df.copy()
    qr["query-id"] = qr["query-id"].astype(str)
    qr["corpus-id"] = qr["corpus-id"].astype(str)
    qr = qr[qr["query-id"].isin(qid_to_row) & qr["corpus-id"].isin(cid_to_row)]
    aucs: list[float] = []
    per_query: dict[str, float] = {}
    skipped_no_pos = skipped_no_neg = 0
    for qid, g in qr.groupby("query-id"):
        y = (g[score_col] > 0).astype(int).values
        if y.sum() == 0:
            skipped_no_pos += 1; continue
        if y.sum() == len(y):
            skipped_no_neg += 1; continue
        col_idx = g["corpus-id"].map(cid_to_row).values
        auc = float(sklearn.metrics.roc_auc_score(y, S[qid_to_row[qid], col_idx]))
        aucs.append(auc)
        if return_per_query:
            per_query[str(qid)] = auc
    stats = {
        "n_queries_scored": len(aucs),
        "skipped_no_pos": int(skipped_no_pos),
        "skipped_no_neg": int(skipped_no_neg),
    }
    if return_per_query:
        stats["per_query"] = per_query
    return (float(np.mean(aucs)) if aucs else float("nan")), stats


