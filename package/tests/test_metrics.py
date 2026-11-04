"""Retrieval metrics: the paper's NDCG (sklearn, full pool, ties averaged), recall, MRR and evaluate()."""

import logging
import math

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import ndcg_score

import medsae.metrics as metrics_module
from medsae.data.dataset import EmbeddedDataset, RetrievalDataset
from medsae.metrics import (
    Judgments,
    evaluate,
    mrr_at_k,
    ndcg_at_k,
    ndcg_from_scores,
    pool_ndcg,
    ranking_metrics,
    recall_at_k,
)
from medsae.retrieval import cosine_scores, encode_corpus, iter_cosine_scores
from medsae.sae import MedSAE


def _run(scores: np.ndarray, doc_ids: list[str]) -> dict[str, float]:
    return {doc: float(score) for doc, score in zip(doc_ids, scores, strict=True)}


def _grades(gains: np.ndarray, doc_ids: list[str]) -> dict[str, int]:
    return {doc: int(grade) for doc, grade in zip(doc_ids, gains, strict=True) if grade != 0}


# Oracle values computed with sklearn.metrics.ndcg_score at k = 10 (linear gains, ties averaged, ideal DCG
# over the full pool), the function that ndcg_at_k in paper/src/utils/metrics/ir.py calls.


def _tie_with_zero_tail():
    gains = np.zeros(1000, dtype=np.float32)
    scores = np.zeros(1000, dtype=np.float32)
    gains[1], gains[3] = 2, 1
    scores[0], scores[1], scores[2] = 0.9, 0.5, 0.5
    return gains, scores


def test_ndcg_oracle_tie_with_zero_tail():
    gains, scores = _tie_with_zero_tail()
    assert ndcg_from_scores(scores[None], gains[None], k=10) == pytest.approx(0.4307791347470307, abs=1e-12)
    docs = [f"d{i}" for i in range(1000)]
    value = ndcg_at_k({"q": _run(scores, docs)}, {"q": _grades(gains, docs)}, k=10)
    assert value == pytest.approx(0.4307791347470307, abs=1e-12)


def test_ndcg_oracle_zero_query_code():
    gains = np.zeros(1000, dtype=np.float32)
    gains[[10, 500, 999]] = [2, 1, 1]
    scores = np.zeros(1000, dtype=np.float32)
    assert ndcg_from_scores(scores[None], gains[None]) == pytest.approx(0.005804741333344191, abs=1e-12)


@pytest.mark.parametrize(("k", "expected"), [(3, 0.40303028380100486), (5, 0.6641442115010362)])
def test_ndcg_oracle_graded_without_ties(k, expected):
    gains = np.array([[0, 2, 1, 0, 1]], dtype=np.float32)
    scores = np.array([[0.9, 0.8, 0.1, 0.7, 0.6]], dtype=np.float32)
    assert ndcg_from_scores(scores, gains, k=k) == pytest.approx(expected, abs=1e-12)
    docs = list("abcde")
    assert ndcg_at_k({"q": _run(scores[0], docs)}, {"q": _grades(gains[0], docs)}, k=k) == pytest.approx(
        expected, abs=1e-12
    )


def test_ndcg_leaves_out_queries_without_positives():
    gains = np.array([[1, 0, 0], [0, 0, 0]], dtype=np.float32)
    scores = np.array([[0.1, 0.9, 0.5], [0.3, 0.2, 0.1]], dtype=np.float32)
    assert ndcg_from_scores(scores, gains) == pytest.approx(0.5)
    run = {"a": _run(scores[0], list("xyz")), "b": _run(scores[1], list("xyz"))}
    assert ndcg_at_k(run, {"a": {"x": 1}}) == pytest.approx(0.5)
    assert ndcg_from_scores(scores[1:], gains[1:]) == 0.0


def test_dictionary_and_matrix_ndcg_agree_on_full_pool_runs():
    rng = np.random.default_rng(0)
    scores = np.round(rng.random((30, 60)), 1).astype(np.float32)  # rounding creates many ties
    gains = rng.choice([0, 0, 0, 0, 1, 2], size=(30, 60)).astype(np.float32)
    docs = [f"d{i}" for i in range(60)]
    run = {f"q{i}": _run(scores[i], docs) for i in range(30)}
    qrels = {f"q{i}": _grades(gains[i], docs) for i in range(30)}
    for k in (1, 5, 10, 60):
        expected = ndcg_score(gains[gains.sum(1) > 0], scores[gains.sum(1) > 0], k=k)
        assert ndcg_from_scores(scores, gains, k=k) == pytest.approx(expected, abs=1e-12)
        assert ndcg_at_k(run, qrels, k=k) == pytest.approx(expected, abs=1e-12)


def test_ndcg_of_a_truncated_run_counts_missing_judged_documents_in_the_ideal():
    qrels = {"q": {"a": 1, "b": 1, "c": 1}}
    discounts = 1 / np.log2(np.arange(2, 5))
    assert ndcg_at_k({"q": {"a": 0.9, "x": 0.5}}, qrels, k=10) == pytest.approx(1 / discounts.sum())
    assert ndcg_at_k({"q": {}}, qrels) == 0.0
    assert ndcg_at_k({"other": {"a": 1.0}}, qrels) == 0.0  # no evaluated query


def test_recall_and_mrr_on_runs():
    run = {"q1": {"a": 0.9, "b": 0.8, "c": 0.1}, "q2": {"a": 0.2, "c": 0.7}, "q3": {"a": 1.0}}
    qrels = {"q1": {"b": 2, "z": 1}, "q2": {"a": 1, "c": 0}, "q3": {"x": 0}}
    assert recall_at_k(run, qrels, k=1) == pytest.approx((0 + 0) / 2)
    assert recall_at_k(run, qrels, k=2) == pytest.approx((0.5 + 1) / 2)
    assert mrr_at_k(run, qrels, k=10) == pytest.approx((1 / 2 + 1 / 2) / 2)
    assert mrr_at_k(run, qrels, k=1) == 0.0
    assert recall_at_k(run, qrels, k=10, threshold=2) == pytest.approx(1.0)  # only q1 has a grade 2
    assert math.isnan(recall_at_k(run, {}, k=10)) and math.isnan(mrr_at_k(run, {}, k=10))


def _toy_dataset() -> RetrievalDataset:
    corpus = pd.DataFrame({"doc_id": ["d0", "d1", "d2", "d3", "d4"], "text": list("abcde")})
    queries = {"test": pd.DataFrame({"query_id": ["q0", "q1", "q2"], "text": ["x", "y", "z"]})}
    qrels = {
        "test": pd.DataFrame(
            {
                "query_id": ["q0", "q0", "q0", "q1", "q1"],
                "doc_id": ["d1", "d3", "missing", "d2", "d4"],
                "score": [2, 1, 1, -1, 0],
            }
        )
    }
    return RetrievalDataset("toy", corpus, queries, qrels)


def test_judgments_align_qrels_with_the_pool():
    judgments = Judgments.from_dataset(_toy_dataset(), "test")
    assert len(judgments) == 3 and judgments.n_docs == 5
    assert judgments.doc_rows[0].tolist() == [1, 3]  # the document missing from the pool is dropped
    assert judgments.gains[1].tolist() == [0.0, 0.0]  # negative grades count as 0
    assert judgments.n_relevant.tolist() == [3, 0, 0]  # recall counts the missing document too
    assert judgments.relevant_rows[0].tolist() == [1, 3]
    assert judgments.evaluable.tolist() == [True, False, False]
    gains = judgments.dense_gains(0, 2)
    assert gains.dtype == np.float32 and gains.tolist() == [[0, 2, 0, 1, 0], [0, 0, 0, 0, 0]]


def test_judgments_reject_duplicated_documents_and_unknown_splits():
    dataset = _toy_dataset()
    with pytest.raises(ValueError, match="no 'val' queries"):
        Judgments.from_dataset(dataset, "val")
    dataset.corpus.loc[4, "doc_id"] = "d0"
    with pytest.raises(ValueError, match="duplicated doc_id"):
        Judgments.from_dataset(dataset, "test")


def _embedded(n_docs: int = 40, n_queries: int = 12, dim: int = 8, seed: int = 0) -> EmbeddedDataset:
    """Each query is a noisy copy of one document (grade 2) and lightly related to the next one (grade 1)."""
    rng = np.random.default_rng(seed)
    docs = rng.normal(size=(n_docs, dim)).astype(np.float32)
    targets = rng.choice(n_docs - 1, n_queries, replace=False)
    query_vectors = docs[targets] + 0.01 * rng.normal(size=(n_queries, dim)).astype(np.float32)
    corpus = pd.DataFrame({"doc_id": [f"d{i}" for i in range(n_docs)], "text": ["t"] * n_docs})
    query_ids = [f"q{i}" for i in range(n_queries)]
    queries = pd.DataFrame({"query_id": query_ids, "text": ["t"] * n_queries})
    qrels = pd.DataFrame(
        {
            "query_id": query_ids * 2,
            "doc_id": [f"d{t}" for t in targets] + [f"d{t + 1}" for t in targets],
            "score": [2] * n_queries + [1] * n_queries,
        }
    )
    dataset = RetrievalDataset("toy", corpus, {"test": queries}, {"test": qrels})
    return EmbeddedDataset(dataset, "toy-embedder", docs, {"test": query_vectors.astype(np.float32)})


def test_evaluate_dense_embeddings():
    data = _embedded()
    result = evaluate(None, data, "test")
    assert set(result) == {"ndcg@10", "recall@100", "mrr@10", "n_queries"}
    assert result["n_queries"] == 12 and result["mrr@10"] == 1.0 and result["recall@100"] == 1.0
    scores = cosine_scores(data.queries["test"], data.corpus)
    gains = Judgments.from_dataset(data.dataset, "test").dense_gains(0, 12)
    assert result["ndcg@10"] == pytest.approx(ndcg_score(gains, scores, k=10), abs=1e-12)
    assert set(evaluate(None, data, "test", k=5)) == {"ndcg@5", "recall@100", "mrr@5", "n_queries"}


def test_evaluate_sae_codes_matches_the_manual_pipeline():
    data = _embedded()
    torch.manual_seed(0)
    sae = MedSAE(d_in=8, n_latents=24, k=4, embedder="toy-embedder")
    result = evaluate(sae, data, "test", batch_size=5)
    query_codes = encode_corpus(sae, data.queries["test"])
    corpus_codes = encode_corpus(sae, data.corpus)
    judgments = Judgments.from_dataset(data.dataset, "test")
    expected = ranking_metrics(query_codes, corpus_codes, judgments)
    assert result == pytest.approx(expected)
    gains = judgments.dense_gains(0, len(judgments))
    assert result["ndcg@10"] == pytest.approx(
        ndcg_from_scores(cosine_scores(query_codes, corpus_codes), gains), abs=1e-12
    )


def test_evaluate_warns_about_another_embedder_and_rejects_missing_splits(caplog):
    data = _embedded()
    sae = MedSAE(d_in=8, n_latents=24, k=4, embedder="another-embedder")
    with caplog.at_level(logging.WARNING, logger="medsae.metrics"):
        evaluate(sae, data, "test")
    assert "another-embedder" in caplog.text
    with pytest.raises(ValueError, match="no 'val' queries"):
        evaluate(None, data, "val")


def test_block_wise_ndcg_equals_one_block(monkeypatch):
    data = _embedded(n_docs=50, n_queries=12)
    sae = MedSAE(d_in=8, n_latents=32, k=3)
    query_codes, corpus_codes = encode_corpus(sae, data.queries["test"]), encode_corpus(sae, data.corpus)
    judgments = Judgments.from_dataset(data.dataset, "test")
    one_block = pool_ndcg(query_codes, corpus_codes, judgments)
    one_block_metrics = ranking_metrics(query_codes, corpus_codes, judgments)

    def small_blocks(queries, corpus):
        return iter_cosine_scores(queries, corpus, block_bytes=4 * 50 * 5)  # 5 queries per block

    monkeypatch.setattr(metrics_module, "iter_cosine_scores", small_blocks)
    assert pool_ndcg(query_codes, corpus_codes, judgments) == pytest.approx(one_block, abs=1e-12)
    assert ranking_metrics(query_codes, corpus_codes, judgments) == pytest.approx(one_block_metrics)


def test_ranking_metrics_recall_and_mrr_match_the_run_versions():
    data = _embedded(n_docs=30, n_queries=10, seed=3)
    scores = cosine_scores(data.queries["test"], data.corpus)
    docs = data.dataset.corpus["doc_id"].tolist()
    query_ids = data.dataset.queries["test"]["query_id"].tolist()
    run = {query: _run(scores[i], docs) for i, query in enumerate(query_ids)}
    qrels = data.dataset.qrels_dict("test")
    result = ranking_metrics(data.queries["test"], data.corpus, Judgments.from_dataset(data.dataset, "test"))
    assert result["recall@100"] == pytest.approx(recall_at_k(run, qrels, k=100))
    assert result["mrr@10"] == pytest.approx(mrr_at_k(run, qrels, k=10))
    assert result["ndcg@10"] == pytest.approx(ndcg_at_k(run, qrels, k=10), abs=1e-12)
