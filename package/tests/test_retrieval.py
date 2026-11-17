"""Sparse encoding of collections, cosine scores in blocks, and search."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from sklearn.metrics.pairwise import cosine_similarity

from doctosae.retrieval import cosine_scores, encode_corpus, iter_cosine_scores, search, top_k_columns
from doctosae.sae import DoctoSAE


def _sae(d_in: int = 12, n_latents: int = 40, k: int = 5, seed: int = 0, **kwargs) -> DoctoSAE:
    torch.manual_seed(seed)
    return DoctoSAE(d_in=d_in, n_latents=n_latents, k=k, **kwargs)


def _embeddings(n: int, dim: int = 12, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)


def test_encode_corpus_equals_dense_codes():
    sae, x = _sae(), _embeddings(50)
    codes = encode_corpus(sae, x, batch_size=7)
    assert isinstance(codes, sp.csr_matrix) and codes.dtype == np.float32 and codes.shape == (50, 40)
    with torch.no_grad():
        dense = sae.encode(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(codes.toarray(), dense, atol=1e-6)
    assert np.diff(codes.indptr).max() <= 5


def test_encode_corpus_stores_activations_by_decreasing_magnitude():
    codes = encode_corpus(_sae(), _embeddings(30))
    for row in range(codes.shape[0]):
        values = codes.data[codes.indptr[row] : codes.indptr[row + 1]]
        assert np.all(np.diff(np.abs(values)) <= 0)


def test_encode_corpus_is_independent_of_the_batch_size():
    sae, x = _sae(), _embeddings(64)
    small, large = encode_corpus(sae, x, batch_size=3), encode_corpus(sae, x, batch_size=4096)
    np.testing.assert_allclose(small.toarray(), large.toarray(), atol=1e-6)
    assert (small != 0).toarray().tolist() == (large != 0).toarray().tolist()


def test_encode_corpus_keeps_empty_codes_empty():
    sae = _sae()
    with torch.no_grad():
        sae.encoder.bias.fill_(-100.0)  # every pre activation is negative: ReLU zeroes the selected ones
    codes = encode_corpus(sae, _embeddings(6))
    assert codes.nnz == 0 and codes.shape == (6, 40)
    assert not cosine_scores(codes, encode_corpus(_sae(seed=1), _embeddings(4))).any()


def test_encode_corpus_without_relu_keeps_negative_activations():
    sae = _sae(activation="none")
    with torch.no_grad():
        sae.encoder.bias.fill_(-100.0)  # the k largest pre activations are all negative
    codes = encode_corpus(sae, _embeddings(10))
    assert np.all(np.diff(codes.indptr) == 5) and np.all(codes.data < 0)


def test_encode_corpus_restores_the_training_mode_and_checks_shapes():
    sae = _sae()
    sae.train()
    encode_corpus(sae, _embeddings(3))
    assert sae.training
    sae.eval()
    encode_corpus(sae, _embeddings(3))
    assert not sae.training
    with pytest.raises(ValueError, match="shape"):
        encode_corpus(sae, _embeddings(3, dim=5))
    assert encode_corpus(sae, np.zeros((0, 12), dtype=np.float32)).shape == (0, 40)


def test_cosine_scores_dense_and_sparse():
    queries, docs = _embeddings(5, seed=1), _embeddings(9, seed=2)
    dense = cosine_scores(queries, docs)
    assert dense.dtype == np.float32 and dense.shape == (5, 9)
    np.testing.assert_allclose(dense, cosine_similarity(queries, docs), atol=1e-6)
    sae = _sae()
    query_codes, doc_codes = encode_corpus(sae, queries), encode_corpus(sae, docs)
    sparse = cosine_scores(query_codes, doc_codes)
    expected = cosine_similarity(query_codes.toarray(), doc_codes.toarray())
    np.testing.assert_allclose(sparse, expected, atol=1e-6)
    np.testing.assert_allclose(cosine_scores(query_codes.toarray(), doc_codes), sparse, atol=1e-6)


@pytest.mark.parametrize("sparse", [True, False])
def test_score_blocks_concatenate_to_the_full_matrix(sparse):
    queries, docs = _embeddings(23, seed=3), _embeddings(11, seed=4)
    if sparse:
        sae = _sae()
        queries, docs = encode_corpus(sae, queries), encode_corpus(sae, docs)
    blocks = list(iter_cosine_scores(queries, docs, block_bytes=4 * 11 * 5))
    assert [start for start, _ in blocks] == [0, 5, 10, 15, 20]
    full = np.vstack([block for _, block in blocks])
    one_block = next(iter(iter_cosine_scores(queries, docs)))[1]
    if sparse:  # row wise sparse products: bit identical whatever the block size
        assert np.array_equal(full, one_block)
    else:
        np.testing.assert_allclose(full, one_block, atol=1e-6)


def test_top_k_columns():
    scores = np.array([[0.1, 0.9, 0.5, 0.7], [0.3, 0.2, 0.8, 0.0]], dtype=np.float32)
    assert top_k_columns(scores, 2).tolist() == [[1, 3], [2, 0]]
    assert top_k_columns(scores, 10).tolist() == [[1, 3, 2, 0], [2, 0, 1, 3]]
    assert top_k_columns(scores, 0).shape == (2, 0)


def test_search_dense_and_sparse():
    queries, docs = _embeddings(6, seed=5), _embeddings(20, seed=6)
    indices, scores = search(None, queries, docs, top_k=4)
    full = cosine_scores(queries, docs)
    assert indices.shape == scores.shape == (6, 4)
    assert indices.tolist() == np.argsort(-full, axis=1)[:, :4].tolist()
    assert np.all(np.diff(scores, axis=1) <= 0)
    sae = _sae()
    indices, scores = search(sae, queries, docs, top_k=3)
    expected = cosine_scores(encode_corpus(sae, queries), encode_corpus(sae, docs))
    np.testing.assert_allclose(scores, np.take_along_axis(expected, indices, axis=1), atol=1e-7)
    assert np.allclose(scores, -np.sort(-expected, axis=1)[:, :3], atol=1e-7)


def test_search_single_query_and_large_top_k():
    docs = _embeddings(5, seed=7)
    indices, scores = search(None, docs[2], docs, top_k=50)
    assert indices.shape == (1, 5) and indices[0, 0] == 2 and scores[0, 0] == pytest.approx(1.0)
