"""Batch sampling: positive pairs, the unique query pair sampler and the shuffled loaders."""

import itertools

import numpy as np
import pandas as pd
import pytest
import torch

from doctosae.training.sampling import (
    PairBatchSampler,
    ShuffledBatches,
    iter_forever,
    positive_pairs,
    positive_sets,
)


def _random_pairs(n_queries: int, n_docs: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pairs = [(q, d) for q in range(n_queries) for d in rng.choice(n_docs, rng.integers(1, 6), replace=False)]
    order = rng.permutation(len(pairs))
    return np.asarray(pairs, dtype=np.int64)[order]


def test_positive_pairs_keep_qrels_order_threshold_and_known_rows():
    qrels = pd.DataFrame(
        {
            "query_id": ["q2", "q1", "q1", "q9", "q2", "q1"],
            "doc_id": ["d1", "d3", "d1", "d1", "d7", "d2"],
            "score": [1, 2, 0, 1, 1, 1],
        }
    )
    pairs = positive_pairs(qrels, {"q1": 0, "q2": 1}, {"d1": 10, "d2": 11, "d3": 12}, threshold=1)
    assert pairs.dtype == np.int64
    assert pairs.tolist() == [[1, 10], [0, 12], [0, 11]]
    assert positive_pairs(qrels, {"q1": 0}, {"d3": 5}, threshold=3).shape == (0, 2)


def test_positive_sets():
    sets = positive_sets(np.array([[0, 4], [1, 4], [0, 7], [0, 4]]))
    assert sets == {0: frozenset({4, 7}), 1: frozenset({4})}
    assert positive_sets(np.zeros((0, 2), dtype=np.int64)) == {}


def test_pair_sampler_batches_hold_each_query_once():
    pairs = _random_pairs(n_queries=50, n_docs=200, seed=0)
    sampler = PairBatchSampler(pairs, batch_size=16, seed=3)
    batches = list(sampler)
    assert len(batches) == len(sampler) == (50 // 16) * sampler.n_passes
    for batch in batches:
        queries = pairs[batch, 0]
        assert len(batch) == 16 and len(set(queries.tolist())) == 16


def test_pair_sampler_passes_and_partial_batches():
    pairs = _random_pairs(n_queries=10, n_docs=40, seed=1)
    full_passes = list(PairBatchSampler(pairs, batch_size=4, seed=0))
    # 10 queries in batches of 4: two full batches per pass, the last 2 queries of each pass are dropped.
    assert len(full_passes) == 2 * PairBatchSampler(pairs, 4).n_passes
    sampler = PairBatchSampler(pairs, batch_size=4, seed=0, drop_last=False)
    batches = list(sampler)
    per_pass = [batches[i : i + 3] for i in range(0, len(batches), 3)]
    assert len(per_pass) == sampler.n_passes
    for one_pass in per_pass:
        assert [len(batch) for batch in one_pass] == [4, 4, 2]
        assert sorted(pairs[sum(one_pass, []), 0].tolist()) == list(range(10))


def test_pair_sampler_n_passes_draws_about_once_per_pair():
    pairs = np.array([[q, d] for q in range(8) for d in range(5)])  # 8 queries, 40 pairs
    assert PairBatchSampler(pairs, batch_size=4).n_passes == 5  # 40 pairs / (2 batches of 4 per pass)
    assert PairBatchSampler(pairs, batch_size=100).n_passes == 1  # fewer queries than one batch


def test_pair_sampler_is_reproducible_and_keeps_its_random_state():
    pairs = _random_pairs(n_queries=30, n_docs=100, seed=2)
    first = PairBatchSampler(pairs, batch_size=8, seed=11)
    second = PairBatchSampler(pairs, batch_size=8, seed=11)
    assert list(first) == list(second)
    assert list(first) != list(PairBatchSampler(pairs, batch_size=8, seed=11))  # state carried over
    assert list(PairBatchSampler(pairs, 8, seed=12)) != list(PairBatchSampler(pairs, 8, seed=11))


def test_pair_sampler_draws_every_positive_of_a_query():
    pairs = np.array([[0, d] for d in range(4)] + [[1, 9]])
    sampler = PairBatchSampler(pairs, batch_size=2, seed=0)
    drawn = {pairs[index, 1] for batch in itertools.islice(iter_forever(sampler), 200) for index in batch}
    assert drawn == {0, 1, 2, 3, 9}


def test_pair_sampler_without_shuffle_takes_first_pairs_in_order():
    pairs = np.array([[2, 0], [0, 1], [2, 5], [1, 3]])
    sampler = PairBatchSampler(pairs, batch_size=3, shuffle=False)
    assert sampler.n_passes == 2  # 4 pairs, 3 per pass
    assert list(sampler) == [[0, 1, 3], [0, 1, 3]]  # queries in order of first appearance


def test_pair_sampler_without_pairs_yields_nothing():
    assert list(PairBatchSampler(np.zeros((0, 2), dtype=np.int64), batch_size=4)) == []


def test_shuffled_batches_drop_the_last_batch_and_reshuffle():
    batches = ShuffledBatches(10, 3, torch.Generator().manual_seed(0))
    first, second = list(batches), list(batches)
    assert len(batches) == 3 and [len(batch) for batch in first] == [3, 3, 3]
    assert len(set(torch.cat(first).tolist())) == 9
    assert not all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    again = list(ShuffledBatches(10, 3, torch.Generator().manual_seed(0)))
    assert all(torch.equal(a, b) for a, b in zip(first, again, strict=True))


def test_iter_forever_restarts_each_pass_and_rejects_empty_passes():
    batches = ShuffledBatches(4, 2, torch.Generator().manual_seed(0))
    drawn = list(itertools.islice(iter_forever(batches), 6))
    assert len(drawn) == 6 and all(len(batch) == 2 for batch in drawn)
    with pytest.raises(ValueError, match="no batch"):
        next(iter_forever([]))
