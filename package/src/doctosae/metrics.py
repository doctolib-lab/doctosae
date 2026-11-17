"""Retrieval metrics with the definitions of the paper code (``paper/src/utils/metrics/ir.py``).

NDCG@k is ``sklearn.metrics.ndcg_score`` over the full document pool: linear gains (a grade 2 document
counts twice a grade 1 document), log2 discount, ideal DCG from every judged document of the pool, and ties
averaged (McSherry and Najork, 2008), which matters with sparse codes where many documents score exactly 0.
Only queries with at least one relevant document in the pool enter the mean. Recall@k follows the paper's
precision and recall code; MRR@k is not reported in the paper and is provided for convenience.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import ndcg_score

from doctosae._ids import embedding_differences, embedding_settings, resolve_embedder
from doctosae.data.dataset import EmbeddedDataset, RetrievalDataset
from doctosae.retrieval import Codes, encode_corpus, iter_cosine_scores, top_k_columns
from doctosae.sae import DoctoSAE

__all__ = [
    "Judgments",
    "ndcg_at_k",
    "recall_at_k",
    "mrr_at_k",
    "ndcg_from_scores",
    "pool_ndcg",
    "ranking_metrics",
    "evaluate",
    "warn_on_other_settings",
]

logger = logging.getLogger(__name__)

Run = Mapping[str, Mapping[str, float]]
Qrels = Mapping[str, Mapping[str, int]]


# Metrics on run and qrels dictionaries.


def _discounts(n: int, k: int) -> np.ndarray:
    """``1 / log2(rank + 1)`` for ranks 1 to ``n``, zero after rank ``k`` (as sklearn computes it)."""
    discount = 1 / (np.log(np.arange(n) + 2) / np.log(2))
    discount[k:] = 0
    return discount


def _tie_averaged_dcg(gains: np.ndarray, scores: np.ndarray, k: int) -> float:
    """DCG@k where tied documents share the average gain of their group (sklearn's ``_tie_averaged_dcg``)."""
    discount_cumsum = np.cumsum(_discounts(len(scores), k))
    _, inverse, counts = np.unique(-scores, return_inverse=True, return_counts=True)
    group_gains = np.zeros(len(counts))
    np.add.at(group_gains, inverse, gains)
    group_gains /= counts
    group_discounts = np.diff(discount_cumsum[np.cumsum(counts) - 1], prepend=0.0)
    return float((group_gains * group_discounts).sum())


def _ranked_docs(doc_scores: Mapping[str, float]) -> list[str]:
    """Documents by decreasing score; ties keep the run order."""
    return sorted(doc_scores, key=doc_scores.__getitem__, reverse=True)


def ndcg_at_k(run: Run, qrels: Qrels, k: int = 10) -> float:
    """Mean NDCG@k of a run ``{query_id: {doc_id: score}}`` against qrels ``{query_id: {doc_id: grade}}``.

    Same definition as :func:`ndcg_from_scores` (and the paper): linear gains, ties averaged, ideal DCG
    from the query's judged documents. Judged documents missing from the run add nothing to the DCG but
    count in the ideal DCG, so a run truncated to its top documents is scored like ``trec_eval``. A run
    that scores the whole pool gives exactly the paper's value when the qrels only judge documents of the
    pool: the paper drops judged documents missing from the corpus, whereas here they still count in the
    ideal DCG and in the choice of queries (:func:`evaluate` drops them, as the paper does). Queries of the
    run without a relevant judgment are left out of the mean, and queries absent from the run are not
    evaluated. Returns 0.0 when no query qualifies.
    """
    values = []
    for query_id, doc_scores in run.items():
        judged = qrels.get(query_id, {})
        positive = np.array([grade for grade in judged.values() if grade > 0], dtype=np.float64)
        if not positive.size:
            continue
        ideal_gains = np.sort(positive)[::-1][:k]
        ideal = float(ideal_gains @ _discounts(len(ideal_gains), k))
        docs = list(doc_scores)
        gains = np.array([max(judged.get(doc, 0), 0) for doc in docs], dtype=np.float64)
        scores = np.array([doc_scores[doc] for doc in docs], dtype=np.float64)
        values.append(_tie_averaged_dcg(gains, scores, k) / ideal if docs else 0.0)
    return float(np.mean(values)) if values else 0.0


def _relevant_queries(run: Run, qrels: Qrels, threshold: int) -> Iterable[tuple[list[str], set[str]]]:
    for query_id, doc_scores in run.items():
        relevant = {doc for doc, grade in qrels.get(query_id, {}).items() if grade >= threshold}
        if relevant:
            yield _ranked_docs(doc_scores), relevant


def recall_at_k(run: Run, qrels: Qrels, k: int = 100, threshold: int = 1) -> float:
    """Mean share of each query's relevant documents (``grade >= threshold``) found in its top ``k``.

    The denominator counts every relevant document of the qrels, retrieved or not, as the paper's recall code.
    Queries without a relevant document are left out; NaN when no query qualifies.
    """
    values = [
        len(set(ranked[:k]) & relevant) / len(relevant)
        for ranked, relevant in _relevant_queries(run, qrels, threshold)
    ]
    return float(np.mean(values)) if values else math.nan


def mrr_at_k(run: Run, qrels: Qrels, k: int = 10, threshold: int = 1) -> float:
    """Mean reciprocal rank of the first relevant document (``grade >= threshold``) in the top ``k``.

    A query without a relevant document in its top ``k`` scores 0; queries without any relevant judgment are
    left out, and the result is NaN when no query qualifies.
    """
    values = []
    for ranked, relevant in _relevant_queries(run, qrels, threshold):
        rank = next((position for position, doc in enumerate(ranked[:k], start=1) if doc in relevant), None)
        values.append(1.0 / rank if rank else 0.0)
    return float(np.mean(values)) if values else math.nan


# Metrics over a full document pool.


@dataclass
class Judgments:
    """Relevance judgments of the queries of one split, aligned with the rows of a document pool.

    Attributes:
        doc_rows: Per query, the pool rows of its judged documents (documents missing from the pool are
            dropped, as in the paper's relevance matrix).
        gains: Per query, the grades of those documents as float32 gains; negative grades count as 0.
        relevant_rows: Per query, the pool rows of its relevant documents (``grade >= threshold``).
        n_relevant: Per query, the number of distinct relevant documents in the qrels, in the pool or not: the
            denominator of recall.
        n_docs: Size of the document pool.
    """

    doc_rows: list[np.ndarray]
    gains: list[np.ndarray]
    relevant_rows: list[np.ndarray]
    n_relevant: np.ndarray
    n_docs: int

    @classmethod
    def from_dataset(cls, dataset: RetrievalDataset, split: str) -> Judgments:
        """Judgments of ``dataset.queries[split]`` (in row order) against the whole ``dataset.corpus``."""
        if split not in dataset.queries:
            raise ValueError(f"{dataset.name} has no {split!r} queries; available splits: {dataset.splits}")
        duplicated = dataset.corpus["doc_id"].duplicated()
        if duplicated.any():
            raise ValueError(
                f"{dataset.name} has {int(duplicated.sum())} duplicated doc_id rows; evaluation expects one "
                "corpus row per document (drop duplicates keeping the first, as the paper does)"
            )
        judged = dataset.qrels_dict(split) if split in dataset.qrels else {}
        relevant = dataset.positives(split) if split in dataset.qrels else {}
        index = dataset.doc_index
        doc_rows, gains, relevant_rows, n_relevant = [], [], [], []
        for query_id in dataset.queries[split]["query_id"].tolist():
            pairs = [(index[doc], grade) for doc, grade in judged.get(query_id, {}).items() if doc in index]
            doc_rows.append(np.array([row for row, _ in pairs], dtype=np.int64))
            # sklearn rejects negative gains; a negative judgment is as irrelevant as an unjudged document.
            gains.append(np.array([max(grade, 0) for _, grade in pairs], dtype=np.float32))
            query_relevant = relevant.get(query_id, set())
            in_pool = sorted(index[doc] for doc in query_relevant if doc in index)
            relevant_rows.append(np.array(in_pool, dtype=np.int64))
            n_relevant.append(len(query_relevant))
        return cls(doc_rows, gains, relevant_rows, np.array(n_relevant, dtype=np.int64), len(dataset.corpus))

    def __len__(self) -> int:
        return len(self.doc_rows)

    @property
    def evaluable(self) -> np.ndarray:
        """Boolean mask of the queries with a positive gain in the pool: the queries NDCG averages over."""
        return np.array([bool((gains > 0).any()) for gains in self.gains], dtype=bool)

    def dense_gains(self, start: int, stop: int) -> np.ndarray:
        """Float32 gains ``[stop - start, n_docs]`` of queries ``start`` to ``stop``, 0 when unjudged."""
        matrix = np.zeros((stop - start, self.n_docs), dtype=np.float32)
        for offset, query in enumerate(range(start, stop)):
            matrix[offset, self.doc_rows[query]] = self.gains[query]
        return matrix


def _ndcg_sum(scores: np.ndarray, gains: np.ndarray, k: int) -> tuple[float, int]:
    """Sum of the NDCG@k of the rows with a positive gain, and their number."""
    has = gains.sum(axis=1) > 0
    count = int(has.sum())
    if count == 0:
        return 0.0, 0
    return float(ndcg_score(gains[has], scores[has], k=k)) * count, count


def ndcg_from_scores(scores: np.ndarray, gains: np.ndarray, k: int = 10) -> float:
    """Mean NDCG@k of a score matrix against a gain matrix of the same shape ``[n_queries, n_docs]``.

    The paper's ``ndcg_at_k``: ``sklearn.metrics.ndcg_score`` on the rows with at least one positive gain, 0.0
    when there is none.
    """
    total, count = _ndcg_sum(np.asarray(scores, dtype=np.float32), np.asarray(gains, dtype=np.float32), k)
    return total / count if count else 0.0


def pool_ndcg(query_codes: Codes, corpus_codes: Codes, judgments: Judgments, k: int = 10) -> float:
    """NDCG@k of queries against the full pool of documents, by blocks of queries (dense or sparse codes)."""
    total, count = 0.0, 0
    for start, scores in iter_cosine_scores(query_codes, corpus_codes):
        block_total, block_count = _ndcg_sum(scores, judgments.dense_gains(start, start + len(scores)), k)
        total += block_total
        count += block_count
    return total / count if count else 0.0


def ranking_metrics(
    query_codes: Codes, corpus_codes: Codes, judgments: Judgments, k: int = 10, recall_k: int = 100
) -> dict[str, float]:
    """NDCG@k, recall@recall_k and MRR@k of queries against the full pool, plus the number of NDCG queries."""
    ndcg_total, n_ndcg = 0.0, 0
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    for start, scores in iter_cosine_scores(query_codes, corpus_codes):
        block_total, block_count = _ndcg_sum(scores, judgments.dense_gains(start, start + len(scores)), k)
        ndcg_total += block_total
        n_ndcg += block_count
        top = top_k_columns(scores, max(k, recall_k))
        for offset, ranked in enumerate(top):
            query = start + offset
            if judgments.n_relevant[query] == 0:
                continue
            relevant = judgments.relevant_rows[query]
            recalls.append(np.isin(ranked[:recall_k], relevant).sum() / judgments.n_relevant[query])
            hits = np.flatnonzero(np.isin(ranked[:k], relevant))
            reciprocal_ranks.append(1.0 / (hits[0] + 1) if hits.size else 0.0)
    return {
        f"ndcg@{k}": ndcg_total / n_ndcg if n_ndcg else 0.0,
        f"recall@{recall_k}": float(np.mean(recalls)) if recalls else math.nan,
        f"mrr@{k}": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else math.nan,
        "n_queries": n_ndcg,
    }


def warn_on_other_settings(sae: DoctoSAE, data: EmbeddedDataset) -> None:
    """Warn when the embeddings were computed with other settings than the training embeddings of the SAE
    (query instruction, revision, document instruction, normalization or truncation)."""
    differences = embedding_differences(
        embedding_settings(sae.embedder, sae.query_instruction, sae.metadata.get("embedding")), data.settings
    )
    if differences:
        logger.warning(
            "These embeddings were computed with other settings than the training embeddings of the SAE "
            "(%s): its scores are not comparable with the published ones.",
            "; ".join(differences),
        )


def evaluate(
    sae: DoctoSAE | None,
    data: EmbeddedDataset,
    split: str = "test",
    k: int = 10,
    device: torch.device | str | None = None,
    *,
    batch_size: int = 4096,
) -> dict[str, float]:
    """Retrieval quality of SAE codes (or of the original embeddings when ``sae`` is None) on one query split.

    Every query of the split is scored against the whole corpus by cosine similarity, as in the paper
    (for CURE the pool is the training and held out documents together). Returns ``{"ndcg@10",
    "recall@100", "mrr@10", "n_queries"}`` (with ``k`` in the keys), where ``n_queries`` is the number of
    queries NDCG averages over.
    """
    judgments = Judgments.from_dataset(data.dataset, split)
    if split not in data.queries:
        raise ValueError(f"no {split!r} query embeddings; embedded splits: {sorted(data.queries)}")
    queries: Codes = data.queries[split]
    corpus: Codes = data.corpus
    if sae is not None:
        if (
            sae.embedder
            and data.embedder
            and resolve_embedder(sae.embedder) != resolve_embedder(data.embedder)
        ):
            logger.warning(
                "The SAE was trained on %s embeddings but the data comes from %s: its codes are likely "
                "meaningless.",
                sae.embedder,
                data.embedder,
            )
        else:
            warn_on_other_settings(sae, data)
        queries = encode_corpus(sae, queries, batch_size=batch_size, device=device)
        corpus = encode_corpus(sae, corpus, batch_size=batch_size, device=device)
    return ranking_metrics(queries, corpus, judgments, k=k)
