"""The query and corpus splits against copies of the paper functions, on random inputs.

``paper_split_queries`` and ``paper_split_corpus`` reproduce ``paper/src/data_prep/split_queries.py`` and the
``--force-train-positives`` branch of ``paper/src/data_prep/split_corpus.py`` line for line, so these tests
pin the package to the algorithm that produced the paper split.
"""

from __future__ import annotations

import hashlib
import uuid

import numpy as np
import pandas as pd
import pytest

from doctosae.data import RetrievalDataset, ids_sha256, split_corpus, split_queries


def paper_split_queries(df, train_frac, val_frac, seed, id_col, group_by_col):
    df = df.copy()
    df["query_id"] = df[id_col].astype(str)
    rng = np.random.default_rng(seed)
    groups = df[group_by_col].drop_duplicates().to_numpy().copy()
    rng.shuffle(groups)
    n = len(groups)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train_groups = set(groups[:n_train].tolist())
    val_groups = set(groups[n_train : n_train + n_val].tolist())
    test_groups = set(groups[n_train + n_val :].tolist())
    return {
        "train": df[df[group_by_col].isin(train_groups)].reset_index(drop=True),
        "val": df[df[group_by_col].isin(val_groups)].reset_index(drop=True),
        "test": df[df[group_by_col].isin(test_groups)].reset_index(drop=True),
    }


def paper_split_corpus(df, train_required, val_required, val_frac, seed):
    rng = np.random.default_rng(seed)
    cid = df["_id"].astype(str)
    mask_train_req = cid.isin(train_required)
    mask_val_req = cid.isin(val_required)
    nonreq_df = df[~(mask_train_req | mask_val_req)].reset_index(drop=True)
    n_nonreq = len(nonreq_df)
    n_val_extra = int(round(n_nonreq * val_frac))
    perm = rng.permutation(n_nonreq)
    val_idx = perm[:n_val_extra]
    train_extra_idx = perm[n_val_extra:]
    train_df = pd.concat([df[mask_train_req], nonreq_df.iloc[train_extra_idx]], ignore_index=True)
    val_df = pd.concat([df[mask_val_req], nonreq_df.iloc[val_idx]], ignore_index=True)
    return train_df, val_df


def random_ids(n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    return [str(uuid.UUID(bytes=rng.bytes(16), version=4)) for _ in range(n)]


def test_cure_sized_query_split_is_1400_200_400():
    rows = split_queries(random_ids(2000, 0))
    assert [len(rows[name]) for name in ("train", "val", "test")] == [1400, 200, 400]
    union = np.concatenate(list(rows.values()))
    assert sorted(union.tolist()) == list(range(2000))
    assert all(np.all(np.diff(members) > 0) for members in rows.values())


@pytest.mark.parametrize(
    ("n", "train_frac", "val_frac", "seed"), [(2000, 0.7, 0.1, 42), (37, 0.6, 0.25, 7), (5, 0.5, 0.5, 1)]
)
def test_query_split_matches_the_paper_code(n, train_frac, val_frac, seed):
    ids = random_ids(n, seed)
    rows = split_queries(ids, train_frac, val_frac, seed)
    expected = paper_split_queries(pd.DataFrame({"_id": ids}), train_frac, val_frac, seed, "_id", "_id")
    for name, frame in expected.items():
        assert [ids[i] for i in rows[name]] == frame["query_id"].tolist()


def test_query_groups_stay_together_as_in_the_paper_code():
    rng = np.random.default_rng(3)
    ids = [f"q{i}" for i in range(60)]
    groups = [f"case{value}" for value in rng.integers(0, 15, size=60)]
    rows = split_queries(ids, 0.7, 0.1, seed=42, groups=groups)
    expected = paper_split_queries(pd.DataFrame({"qid": ids, "case": groups}), 0.7, 0.1, 42, "qid", "case")
    for name, frame in expected.items():
        assert [ids[i] for i in rows[name]] == frame["query_id"].tolist()
    split_of = {groups[i]: name for name, members in rows.items() for i in members}
    assert all(split_of[groups[i]] == name for name, members in rows.items() for i in members)


@pytest.mark.parametrize(("train_frac", "val_frac"), [(0.8, 0.3), (-0.1, 0.5)])
def test_query_split_rejects_bad_fractions(train_frac, val_frac):
    with pytest.raises(ValueError, match="fractions"):
        split_queries(["a", "b"], train_frac, val_frac)


def test_query_split_depends_on_input_order():
    ids = random_ids(200, 1)
    shuffled = list(reversed(ids))
    first = {ids[i] for i in split_queries(ids)["test"]}
    second = {shuffled[i] for i in split_queries(shuffled)["test"]}
    assert first != second


@pytest.mark.parametrize("seed", [42, 5])
def test_forced_corpus_split_matches_the_paper_code(seed):
    rng = np.random.default_rng(seed)
    ids = random_ids(500, seed)
    train_required = set(rng.choice(ids, 120, replace=False).tolist())
    val_required = set(rng.choice(ids, 30, replace=False).tolist())  # overlaps train_required on purpose
    rows = split_corpus(ids, train_required, val_required, val_frac=0.1, seed=seed)
    train_df, val_df = paper_split_corpus(pd.DataFrame({"_id": ids}), train_required, val_required, 0.1, seed)
    ids_array = np.asarray(ids)
    assert sorted(ids_array[rows["train"]]) == sorted(train_df["_id"])
    assert sorted(ids_array[rows["val"]]) == sorted(val_df["_id"])
    assert train_required <= set(ids_array[rows["train"]])
    assert np.all(np.diff(rows["train"]) > 0) and np.all(np.diff(rows["val"]) > 0)  # original corpus order


def test_corpus_split_without_requirements_is_the_paper_random_partition():
    ids = random_ids(321, 2)
    rows = split_corpus(ids, val_frac=0.1, seed=42)
    permutation = np.random.default_rng(42).permutation(len(ids))  # split_corpus.py, branch "_unforced"
    n_val = int(round(len(ids) * 0.1))
    assert rows["val"].tolist() == sorted(permutation[:n_val].tolist())
    assert rows["train"].tolist() == sorted(permutation[n_val:].tolist())


def test_corpus_split_accepts_pandas_ids_and_forces_train_positives_of_a_dataset():
    corpus = pd.DataFrame({"doc_id": [f"d{i}" for i in range(40)], "text": ["t"] * 40})
    queries = {name: pd.DataFrame({"query_id": [f"{name}0"], "text": ["q"]}) for name in ("train", "val")}
    qrels = {
        "train": pd.DataFrame({"query_id": ["train0"] * 3, "doc_id": ["d1", "d2", "d3"], "score": [2, 1, 0]}),
        "val": pd.DataFrame({"query_id": ["val0"], "doc_id": ["d4"], "score": [1]}),
    }
    dataset = RetrievalDataset("toy", corpus, queries, qrels)
    rows = split_corpus(dataset, val_frac=0.5, seed=0)
    train_ids = set(corpus["doc_id"].to_numpy()[rows["train"]])
    assert {"d1", "d2"} <= train_ids  # grade 0 (d3) and val positives (d4) are not forced
    explicit = split_corpus(corpus["doc_id"], {"d1", "d2"}, val_frac=0.5, seed=0)
    assert all(np.array_equal(rows[name], explicit[name]) for name in ("train", "val"))
    assert len(rows["val"]) == round(38 * 0.5)


def test_corpus_split_needs_train_qrels_for_a_dataset():
    corpus = pd.DataFrame({"doc_id": ["a"], "text": ["t"]})
    dataset = RetrievalDataset("toy", corpus, {"test": corpus.rename(columns={"doc_id": "query_id"})}, {})
    with pytest.raises(ValueError, match="no 'train' qrels"):
        split_corpus(dataset)


def test_ids_sha256_ignores_order():
    expected = hashlib.sha256(b"a\nb\nc").hexdigest()
    assert ids_sha256(["c", "a", "b"]) == ids_sha256(("a", "b", "c")) == expected
