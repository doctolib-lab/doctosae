"""Batch sampling of the paper's training loop: shuffled row batches, positive pairs and the pair sampler."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from typing import NamedTuple, TypeVar

import numpy as np
import pandas as pd
import torch

__all__ = [
    "PairBatch",
    "PairBatchSampler",
    "ShuffledBatches",
    "iter_forever",
    "positive_pairs",
    "positive_sets",
]

T = TypeVar("T")


class PairBatch(NamedTuple):
    """One batch of (query, positive document) pairs for the InfoNCE term."""

    queries: torch.Tensor  # [B, d] query embeddings (the anchors)
    documents: torch.Tensor  # [B, d] embeddings of their paired documents
    query_rows: np.ndarray  # [B] query rows, used to look up each anchor's known positives
    doc_rows: np.ndarray  # [B] corpus rows of the paired documents


def positive_pairs(
    qrels: pd.DataFrame,
    query_rows: Mapping[str, int],
    doc_rows: Mapping[str, int],
    threshold: int = 1,
) -> np.ndarray:
    """(query row, document row) of every judgment with ``score >= threshold``, as an int64 ``[n, 2]`` array.

    Judgments whose query or document is not in the given mappings are dropped (for training, documents
    held out of the training split). Pairs keep the qrels row order, which fixes the order the pair
    sampler sees queries in, as ``load_pairs`` of the paper code. Grades 1 and 2 count the same.
    """
    relevant = qrels[qrels["score"] >= threshold]
    queries = relevant["query_id"].map(query_rows)
    docs = relevant["doc_id"].map(doc_rows)
    known = (queries.notna() & docs.notna()).to_numpy()
    pairs = np.stack([queries.to_numpy()[known], docs.to_numpy()[known]], axis=1)
    return pairs.astype(np.int64).reshape(-1, 2)


def positive_sets(pairs: np.ndarray) -> dict[int, frozenset[int]]:
    """Query row to the frozen set of its positive document rows (``build_pos_dict`` of the paper code)."""
    bucket: dict[int, set[int]] = defaultdict(set)
    for query, doc in np.asarray(pairs, dtype=np.int64).reshape(-1, 2).tolist():
        bucket[query].add(doc)
    return {query: frozenset(docs) for query, docs in bucket.items()}


class PairBatchSampler:
    """Batches of pair indices in which every query appears at most once.

    One pass shuffles the queries that have at least one pair, draws one of each query's pairs uniformly at
    random and cuts the result into consecutive batches; the last incomplete batch is dropped unless
    ``drop_last`` is False. One iteration over the sampler makes ``n_passes`` passes (about one draw per
    pair on average) and the random state carries over to the next iteration. This is
    ``UniqueQueryBatchSampler`` of the paper code: for the same pairs, batch size and seed both yield the
    same batches.

    Args:
        pairs: ``[n_pairs, 2]`` array of (query, document) rows, as returned by :func:`positive_pairs`.
        batch_size: Pairs per batch.
        shuffle: Shuffle queries and draw a random pair per query; otherwise take each query's first pair
            in order.
        seed: Seed of the sampler's own ``random.Random``.
        drop_last: Drop the incomplete batch at the end of each pass.
    """

    def __init__(
        self,
        pairs: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = True,
    ) -> None:
        self.pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._rng = random.Random(seed)
        self._pairs_of: dict[int, list[int]] = defaultdict(list)
        for index, query in enumerate(self.pairs[:, 0].tolist()):
            self._pairs_of[query].append(index)
        self.queries = list(self._pairs_of)
        # Pairs covered by the full batches of one pass; enough passes to draw about once per pair.
        per_pass = max(len(self.queries) // self.batch_size, 1) * self.batch_size
        self.n_passes = max(1, (len(self.pairs) + per_pass - 1) // per_pass)

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self.n_passes):
            queries = list(self.queries)
            if self.shuffle:
                self._rng.shuffle(queries)
            chosen = [
                self._rng.choice(self._pairs_of[query]) if self.shuffle else self._pairs_of[query][0]
                for query in queries
            ]
            for start in range(0, len(chosen), self.batch_size):
                batch = chosen[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self) -> int:
        return (len(self.queries) // self.batch_size) * self.n_passes


class ShuffledBatches:
    """Row indices ``0 .. n - 1`` in shuffled batches, the incomplete last batch dropped.

    Same sampling as ``DataLoader(shuffle=True, drop_last=True)``: each iteration draws a fresh
    permutation from ``generator``, so cycling with :func:`iter_forever` reshuffles every pass.
    """

    def __init__(self, n: int, batch_size: int, generator: torch.Generator) -> None:
        self.n = int(n)
        self.batch_size = int(batch_size)
        self.generator = generator

    def __len__(self) -> int:
        return self.n // self.batch_size

    def __iter__(self) -> Iterator[torch.Tensor]:
        order = torch.randperm(self.n, generator=self.generator)
        for start in range(0, len(self) * self.batch_size, self.batch_size):
            yield order[start : start + self.batch_size]


def iter_forever(batches: Iterable[T]) -> Iterator[T]:
    """Iterate over ``batches`` again each time it is exhausted.

    Unlike ``itertools.cycle``, every pass calls ``iter`` again, so shuffles and random draws are fresh
    (``iter_forever`` of the paper code). Raises ``ValueError`` on a pass that yields nothing, which would
    otherwise loop forever.
    """
    while True:
        empty = True
        for batch in batches:
            empty = False
            yield batch
        if empty:
            raise ValueError("cannot cycle over batches: one full pass yielded no batch")
