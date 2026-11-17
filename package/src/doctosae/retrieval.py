"""Sparse codes of embedding collections and cosine retrieval over them.

Scores follow the evaluation code of the paper (``cosine_scores`` in ``paper/src/utils/metrics/ir.py``):
cosine similarity in float32, sparse rows normalised with float32 norms, and zero codes kept at zero, so
a query whose code is empty scores 0 against every document. Scores are produced in blocks of queries so
that the dense score matrix never grows past :data:`SCORE_BLOCK_BYTES`, and corpus codes stay sparse end
to end.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

from doctosae.sae import DoctoSAE

__all__ = [
    "SCORE_BLOCK_BYTES",
    "encode_corpus",
    "cosine_scores",
    "iter_cosine_scores",
    "top_k_columns",
    "search",
]

#: Upper bound on the bytes of one block of float32 scores (queries x documents).
SCORE_BLOCK_BYTES = 128 * 2**20

Codes = np.ndarray | sp.spmatrix | sp.sparray


def _embedding_matrix(embeddings: np.ndarray, d_in: int | None = None) -> np.ndarray:
    matrix = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
    if matrix.ndim != 2 or (d_in is not None and matrix.shape[1] != d_in):
        raise ValueError(f"expected embeddings of shape [n, {d_in or 'd'}], got {np.shape(embeddings)}")
    return matrix


def encode_corpus(
    sae: DoctoSAE,
    embeddings: np.ndarray,
    batch_size: int = 4096,
    device: torch.device | str | None = None,
    show_progress: bool = False,
) -> sp.csr_matrix:
    """Sparse codes of ``embeddings`` (one row per text) as a float32 CSR matrix ``[n, sae.n_latents]``.

    Works for documents and queries alike. Each row stores its at most ``sae.k`` non zero activations in
    decreasing order of magnitude, exactly as ``encode_in_batches_sparse`` of the paper code, so that
    sparse products and the resulting float32 ties match the paper. ``device`` moves the SAE there first;
    by default it runs where the SAE lives.
    """
    matrix = _embedding_matrix(embeddings, sae.d_in)
    if device is not None:
        sae.to(device)
    values: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    was_training = sae.training
    sae.eval()
    try:
        with torch.inference_mode():
            for start in tqdm(range(0, len(matrix), batch_size), desc="encoding", disable=not show_progress):
                codes = sae.encode(torch.tensor(matrix[start : start + batch_size], device=sae.device))
                top = torch.topk(codes.abs(), min(sae.k, codes.shape[1]), dim=1).indices
                top_values = torch.gather(codes, 1, top).cpu().numpy()
                keep = top_values != 0  # rows can hold fewer than k positive activations after the ReLU
                values.append(top_values[keep])
                columns.append(top.cpu().numpy().astype(np.int32)[keep])
                counts.append(keep.sum(axis=1))
    finally:
        sae.train(was_training)
    indptr = np.zeros(len(matrix) + 1, dtype=np.int64)
    if counts:
        np.cumsum(np.concatenate(counts), out=indptr[1:])
    data = np.concatenate(values) if values else np.zeros(0, dtype=np.float32)
    indices = np.concatenate(columns) if columns else np.zeros(0, dtype=np.int32)
    return sp.csr_matrix((data, indices, indptr), shape=(len(matrix), sae.n_latents), dtype=np.float32)


def _normalize_sparse_rows(matrix: sp.csr_matrix) -> sp.csr_matrix:
    """Unit L2 rows, zero rows kept at zero, with the operations of the paper (same float32 rounding)."""
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1))).ravel()
    norms[norms == 0] = 1.0
    return sp.diags(1.0 / norms) @ matrix


def iter_cosine_scores(
    query_codes: Codes, corpus_codes: Codes, block_bytes: int = SCORE_BLOCK_BYTES
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(first_query_row, scores)`` blocks of the float32 cosine score matrix, in query order.

    Inputs are dense arrays or scipy sparse matrices (sparse as soon as either side is). The corpus is
    normalised once; each block of queries holds at most ``block_bytes`` of scores.
    """
    n_queries, n_docs = query_codes.shape[0], corpus_codes.shape[0]
    rows = max(1, block_bytes // (4 * max(n_docs, 1)))
    if sp.issparse(query_codes) or sp.issparse(corpus_codes):
        corpus_t = _normalize_sparse_rows(sp.csr_matrix(corpus_codes, dtype=np.float32)).T.tocsr()
        queries = sp.csr_matrix(query_codes, dtype=np.float32)
        for start in range(0, n_queries, rows):
            block = _normalize_sparse_rows(queries[start : start + rows]) @ corpus_t
            yield start, np.asarray(block.todense(), dtype=np.float32)
    else:
        # sklearn's cosine_similarity, as in the paper, with the corpus normalised once for all blocks.
        corpus_t = normalize(np.asarray(corpus_codes, dtype=np.float32)).T
        dense_queries = np.asarray(query_codes, dtype=np.float32)
        for start in range(0, n_queries, rows):
            block = normalize(dense_queries[start : start + rows]) @ corpus_t
            yield start, block.astype(np.float32, copy=False)


def cosine_scores(query_codes: Codes, corpus_codes: Codes) -> np.ndarray:
    """Float32 cosine similarity matrix ``[n_queries, n_documents]`` of dense arrays or sparse codes."""
    scores = np.empty((query_codes.shape[0], corpus_codes.shape[0]), dtype=np.float32)
    for start, block in iter_cosine_scores(query_codes, corpus_codes):
        scores[start : start + len(block)] = block
    return scores


def top_k_columns(scores: np.ndarray, k: int) -> np.ndarray:
    """Columns of the ``k`` highest scores of each row, best first.

    Selection uses ``np.argpartition`` as the paper's precision and recall code, so documents tied at the cut
    are chosen arbitrarily (but deterministically).
    """
    k = min(int(k), scores.shape[1])
    if k == 0:
        return np.zeros((scores.shape[0], 0), dtype=np.int64)
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1, kind="stable")
    return np.take_along_axis(top, order, axis=1)


def search(
    sae: DoctoSAE | None,
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    top_k: int = 10,
    device: torch.device | str | None = None,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Top ``top_k`` documents of each query by cosine similarity of SAE codes.

    ``sae=None`` scores the original dense embeddings instead. Returns ``(indices, scores)``, two
    ``[n_queries, top_k]`` arrays (int64 corpus rows and float32 scores), best first.
    """
    queries = _embedding_matrix(query_embeddings)
    corpus = _embedding_matrix(corpus_embeddings)
    if sae is not None:
        queries = encode_corpus(sae, queries, batch_size=batch_size, device=device)
        corpus = encode_corpus(sae, corpus, batch_size=batch_size, device=device)
    k = min(int(top_k), corpus.shape[0])
    indices = np.empty((queries.shape[0], k), dtype=np.int64)
    scores = np.empty((queries.shape[0], k), dtype=np.float32)
    for start, block in iter_cosine_scores(queries, corpus):
        top = top_k_columns(block, k)
        indices[start : start + len(block)] = top
        scores[start : start + len(block)] = np.take_along_axis(block, top, axis=1)
    return indices, scores
