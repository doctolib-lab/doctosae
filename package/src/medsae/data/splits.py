"""Deterministic query and corpus splits, built exactly as the paper code builds them.

:func:`split_queries` reproduces ``paper/src/data_prep/split_queries.py`` and :func:`split_corpus`
reproduces ``paper/src/data_prep/split_corpus.py``. Both draw from one ``numpy.random.default_rng(seed)``
and depend on the order of their input: the same ids in another order give another split. For CURE the
input order is the concatenation of the ten discipline files in alphabetical order, which yields
1,400 / 200 / 400 queries and 223,425 / 21,175 corpus rows (see :func:`medsae.data.loaders.load_cure`, which
also compares the result with checksums of the paper split).

NumPy does not formally promise that ``Generator`` streams stay identical across releases, hence
:func:`ids_sha256`, which turns a split into a short fingerprint that can be compared with a published one.
"""

from __future__ import annotations

import hashlib
from collections.abc import Hashable, Iterable, Sequence

import numpy as np
import pandas as pd

from medsae.data.dataset import RetrievalDataset

__all__ = ["split_queries", "split_corpus", "ids_sha256"]


def split_queries(
    ids: Sequence[Hashable],
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    seed: int = 42,
    groups: Sequence[Hashable] | None = None,
) -> dict[str, np.ndarray]:
    """Split queries into train, val and test, as ``split_queries.split_queries`` of the paper code.

    The unique groups, in order of first appearance, are shuffled once with ``default_rng(seed).shuffle``;
    the first ``int(round(n * train_frac))`` go to train, the next ``int(round(n * val_frac))`` to val and
    the rest to test (Python rounds half to even, as in the paper).

    Args:
        ids: Query ids in file order. The order defines the split.
        train_frac: Fraction of groups in train.
        val_frac: Fraction of groups in val; test gets the remainder.
        seed: Seed of ``numpy.random.default_rng``.
        groups: Optional group key per row: rows sharing a key land in the same split (the paper's
            ``--group-by-col``). By default every query is its own group, which is how CURE was split.

    Returns:
        ``{"train": rows, "val": rows, "test": rows}``: ``int64`` row positions into ``ids``, in input order.
    """
    if min(train_frac, val_frac) < 0 or train_frac + val_frac > 1.0 + 1e-9:
        raise ValueError(f"need fractions >= 0 with train_frac + val_frac <= 1, got {train_frac}, {val_frac}")
    if groups is not None and len(groups) != len(ids):
        raise ValueError(f"{len(groups)} group keys for {len(ids)} queries")
    keys = pd.Series(np.asarray(list(ids if groups is None else groups), dtype=object))
    # The paper shuffles df[group_by_col].drop_duplicates().to_numpy(): unique keys, first appearance order.
    unique = keys.drop_duplicates().to_numpy(dtype=object).copy()
    np.random.default_rng(seed).shuffle(unique)
    n_train = int(round(len(unique) * train_frac))
    n_val = int(round(len(unique) * val_frac))
    members = {
        "train": unique[:n_train],
        "val": unique[n_train : n_train + n_val],
        "test": unique[n_train + n_val :],
    }
    return {
        name: np.flatnonzero(keys.isin(set(chosen.tolist())).to_numpy()) for name, chosen in members.items()
    }


def split_corpus(
    corpus: RetrievalDataset | Iterable[str],
    train_required: Iterable[str] | None = None,
    val_required: Iterable[str] = (),
    val_frac: float = 0.1,
    seed: int = 42,
    train_split: str = "train",
) -> dict[str, np.ndarray]:
    """Split corpus rows into train and val, as ``split_corpus.py --force-train-positives`` does.

    Rows whose id is in ``train_required`` go to train and rows whose id is in ``val_required`` go to val
    (to both when both require it, as in the paper). The other rows, in corpus order, are permuted with
    ``default_rng(seed).permutation`` and the first ``int(round(n * val_frac))`` of them go to val, the rest
    to train. With nothing required this is the paper's plain random partition.

    Forcing the training positives into train makes every training pair of the InfoNCE term available. The
    paper does not force validation positives (its README warns against it: test positives never get that
    treatment), hence the empty ``val_required`` default.

    Args:
        corpus: Document ids in corpus order, or a :class:`~medsae.data.RetrievalDataset`. For a dataset,
            ``train_required`` defaults to the positives (``score >= positive_threshold``, that is
            ``score > 0`` for every benchmark of the paper) of its ``train_split`` queries.
        train_required: Document ids forced into train.
        val_required: Document ids forced into val.
        val_frac: Fraction of the rows that are not required sent to val.
        seed: Seed of ``numpy.random.default_rng``.
        train_split: Query split whose positives are forced into train when ``corpus`` is a dataset.

    Returns:
        ``{"train": rows, "val": rows}``: sorted ``int64`` row positions. Each split keeps the original corpus
        order, the order in which the paper's training read it (its per embedder pickles were split with
        ``--ids-from``, which preserves corpus order).
    """
    if not 0.0 <= val_frac <= 1.0:
        raise ValueError(f"val_frac must be in [0, 1], got {val_frac}")
    if isinstance(corpus, RetrievalDataset):
        doc_ids = corpus.corpus["doc_id"].astype(str)
        if train_required is None:
            train_required = _train_positives(corpus, train_split)
    else:
        doc_ids = pd.Series(list(corpus), dtype=object).astype(str)
    train_ids = set() if train_required is None else {str(doc_id) for doc_id in train_required}
    train_mask = doc_ids.isin(train_ids).to_numpy()
    val_mask = doc_ids.isin({str(doc_id) for doc_id in val_required}).to_numpy()
    rest = np.flatnonzero(~(train_mask | val_mask))
    rng = np.random.default_rng(seed)
    n_val = int(round(len(rest) * val_frac))
    order = rng.permutation(len(rest))
    train = np.concatenate([np.flatnonzero(train_mask), rest[order[n_val:]]])
    val = np.concatenate([np.flatnonzero(val_mask), rest[order[:n_val]]])
    return {"train": np.sort(train).astype(np.int64), "val": np.sort(val).astype(np.int64)}


def _train_positives(dataset: RetrievalDataset, split: str) -> set[str]:
    """Positive documents of the queries of ``split`` (the paper's ``_required_corpus_ids``)."""
    if split not in dataset.qrels:
        raise ValueError(f"{dataset.name} has no {split!r} qrels to force into the train corpus")
    query_ids = set(dataset.queries[split]["query_id"]) if split in dataset.queries else None
    return {
        doc
        for query, docs in dataset.positives(split).items()
        if query_ids is None or query in query_ids
        for doc in docs
    }


def ids_sha256(ids: Iterable[Hashable]) -> str:
    """SHA256 of the sorted ids joined by newlines: a fingerprint of a split that ignores row order."""
    return hashlib.sha256("\n".join(sorted(str(item) for item in ids)).encode("utf-8")).hexdigest()
