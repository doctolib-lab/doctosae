"""Numerical equivalence between the package and the code that produced the paper numbers.

The paper modules are imported from the ``paper`` folder of the repository (next to ``package``); the
whole module is skipped when that folder or one of its imports is missing, for example in an sdist.
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from medsae.data.dataset import EmbeddedDataset, RetrievalDataset
from medsae.hub import from_paper_checkpoint
from medsae.metrics import Judgments, evaluate, ndcg_at_k, ranking_metrics
from medsae.retrieval import cosine_scores, encode_corpus
from medsae.sae import MedSAE, geometric_median
from medsae.training import TrainingConfig
from medsae.training.losses import (
    info_nce_loss,
    positive_mask,
    reconstruction_loss,
    squared_error,
    weighted_loss,
)
from medsae.training.sampling import PairBatch, PairBatchSampler, iter_forever, positive_pairs, positive_sets
from medsae.training.trainer import build_optimizer, train_step

PAPER_DIR = Path(__file__).resolve().parents[2] / "paper"
if not (PAPER_DIR / "src" / "utils").is_dir():
    pytest.skip(f"the paper code is not in {PAPER_DIR}", allow_module_level=True)

sys.path.insert(0, str(PAPER_DIR))
try:
    paper_arch = pytest.importorskip("src.utils.sae_architectures")
    paper_contrastive = pytest.importorskip("src.utils.contrastive_utils")
    paper_training = pytest.importorskip("src.utils.training_utils")
    paper_ir = pytest.importorskip("src.utils.metrics.ir")
    paper_encode = pytest.importorskip("src.utils.encode")
    paper_train = pytest.importorskip("src.training.train_sae")
finally:
    sys.path.remove(str(PAPER_DIR))

CPU = torch.device("cpu")


def _models(d: int = 12, m: int = 36, k: int = 5, seed: int = 0) -> tuple[torch.nn.Module, MedSAE]:
    """The paper model and a MedSAE built from the same seed (both draw their weights in the same order)."""
    torch.manual_seed(seed)
    paper = paper_arch.TopkNoReluSparseAutoencoder(d, m, k=k, activation_fn="relu", init="default")
    torch.manual_seed(seed)
    return paper, MedSAE(d_in=d, n_latents=m, k=k)


def _random_pairs(n_queries: int, n_docs: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pairs = [(q, d) for q in range(n_queries) for d in rng.choice(n_docs, rng.integers(1, 8), replace=False)]
    return np.asarray(pairs, dtype=np.int64)[rng.permutation(len(pairs))]


def test_geometric_median_matches_the_paper():
    rng = np.random.default_rng(0)
    small = rng.normal(size=(300, 12)).astype(np.float32)
    assert np.array_equal(geometric_median(small, seed=3), paper_arch.geometric_median(small, seed=3))
    assert np.array_equal(
        geometric_median(small, max_samples=100, seed=41),
        paper_arch.geometric_median(small, max_samples=100, seed=41),
    )
    large = rng.normal(size=(100_200, 4)).astype(np.float32)  # above the 100,000 rows the paper subsamples to
    assert np.array_equal(geometric_median(large, seed=42), paper_arch.geometric_median(large, seed=42))


def test_initialisation_and_topk_encoding_match_the_paper():
    paper, sae = _models()
    for name, tensor in paper.state_dict().items():
        assert torch.equal(sae.state_dict()[name], tensor), name
    b_pre = torch.randn(12)
    paper.set_b_pre(b_pre)
    with torch.no_grad():
        sae.b_pre.copy_(b_pre)
    x = torch.randn(64, 12)
    paper_x_hat, paper_z = paper(x)
    x_hat, z = sae(x)
    assert torch.equal(z, paper_z) and torch.equal(x_hat, paper_x_hat)
    dead = torch.rand(36) < 0.3
    assert torch.equal(sae.aux_recon(x, dead, 4), paper.aux_recon(x, dead, 4))


def test_sparse_codes_and_cosine_scores_match_the_paper():
    paper, sae = _models()
    rng = np.random.default_rng(1)
    queries, docs = (
        rng.normal(size=(40, 12)).astype(np.float32),
        rng.normal(size=(300, 12)).astype(np.float32),
    )
    paper_queries = paper_encode.encode_in_batches_sparse(paper, queries, CPU)
    paper_docs = paper_encode.encode_in_batches_sparse(paper, docs, CPU)
    query_codes, doc_codes = (
        encode_corpus(sae, queries, batch_size=256),
        encode_corpus(sae, docs, batch_size=256),
    )
    for ours, theirs in ((query_codes, paper_queries), (doc_codes, paper_docs)):
        assert ours.shape == theirs.shape
        assert np.array_equal(ours.indptr, theirs.indptr) and np.array_equal(ours.indices, theirs.indices)
        assert np.array_equal(ours.data, theirs.data)
    paper_scores, _ = paper_ir.cosine_scores(paper_queries, paper_docs, np.arange(300))
    assert np.array_equal(cosine_scores(query_codes, doc_codes), paper_scores)
    dense_scores, _ = paper_ir.cosine_scores(queries, docs, np.arange(300))
    np.testing.assert_allclose(cosine_scores(queries, docs), dense_scores, atol=1e-6)
    assert sp.issparse(query_codes)


def _graded_dataset(seed: int = 2) -> EmbeddedDataset:
    """Clustered documents, grades 1 and 2, a judged document outside the pool and an unjudged query."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(10, 12))
    doc_topics = rng.integers(0, 10, 400)
    docs = (centres[doc_topics] + 0.8 * rng.normal(size=(400, 12))).astype(np.float32)
    query_topics = rng.integers(0, 10, 60)
    queries = (centres[query_topics] + 0.8 * rng.normal(size=(60, 12))).astype(np.float32)
    rows = []
    for query, topic in enumerate(query_topics[:-1]):  # the last query has no judgment at all
        same_topic = np.flatnonzero(doc_topics == topic)
        for doc in rng.choice(same_topic, min(8, len(same_topic)), replace=False):
            rows.append((f"q{query}", f"d{doc}", int(rng.integers(1, 3))))
        rows.append((f"q{query}", "absent", 1))
    corpus = pd.DataFrame({"doc_id": [f"d{i}" for i in range(400)], "text": ["text"] * 400})
    query_frame = pd.DataFrame({"query_id": [f"q{i}" for i in range(60)], "text": ["text"] * 60})
    qrels = pd.DataFrame(rows, columns=["query_id", "doc_id", "score"])
    dataset = RetrievalDataset("graded", corpus, {"test": query_frame}, {"test": qrels})
    return EmbeddedDataset(dataset, "toy/embedder", docs, {"test": queries})


def _paper_qrels(data: EmbeddedDataset) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    qrels = data.dataset.qrels["test"].rename(columns={"query_id": "query-id", "doc_id": "corpus-id"})
    queries = pd.DataFrame({"id": data.dataset.queries["test"]["query_id"]})
    uids = data.dataset.corpus["doc_id"].to_numpy()
    return qrels, queries, uids


@pytest.mark.parametrize("with_sae", [False, True])
def test_ndcg_and_recall_match_the_paper_evaluation(with_sae):
    data = _graded_dataset()
    qrels, queries, uids = _paper_qrels(data)
    q_matrix, c_matrix = data.queries["test"], data.corpus
    sae = None
    if with_sae:
        paper, sae = _models()
        q_matrix = paper_encode.encode_in_batches_sparse(paper, q_matrix, CPU)
        c_matrix = paper_encode.encode_in_batches_sparse(paper, c_matrix, CPU)
    scores, _ = paper_ir.cosine_scores(q_matrix, c_matrix, uids)
    relevance = paper_ir.load_qrels_matrix(queries, qrels, pd.DataFrame(index=uids), score_col="score")
    paper_ndcg = paper_ir.ndcg_at_k(q_matrix, c_matrix, uids, relevance, k=10, S=scores)
    query_rows = {query_id: row for row, query_id in enumerate(queries["id"])}
    paper_recall = paper_ir.precision_recall_at_k(
        q_matrix, c_matrix, uids, query_rows, qrels, ks=[100], score_col="score", S=scores
    )[100]
    result = evaluate(sae, data, "test", batch_size=256)
    assert result["ndcg@10"] == pytest.approx(paper_ndcg, abs=1e-12)
    assert result["recall@100"] == pytest.approx(paper_recall["recall"], abs=1e-12)
    assert result["n_queries"] == paper_recall["n_queries"] == 59


def test_dictionary_ndcg_matches_the_paper_on_a_full_pool_run():
    data = _graded_dataset(seed=5)
    qrels, queries, uids = _paper_qrels(data)
    scores, _ = paper_ir.cosine_scores(data.queries["test"], data.corpus, uids)
    relevance = paper_ir.load_qrels_matrix(queries, qrels, pd.DataFrame(index=uids), score_col="score")
    paper_ndcg = paper_ir.ndcg_at_k(data.queries["test"], data.corpus, uids, relevance, k=10, S=scores)
    run = {
        query_id: dict(zip(uids.tolist(), scores[row].tolist(), strict=True))
        for row, query_id in enumerate(queries["id"])
    }
    judged = data.dataset.qrels_dict("test")
    in_pool = {
        query: {doc: grade for doc, grade in docs.items() if doc != "absent"}
        for query, docs in judged.items()
    }
    assert ndcg_at_k(run, in_pool, k=10) == pytest.approx(paper_ndcg, abs=1e-12)


def test_weighted_reconstruction_loss_matches_the_paper():
    torch.manual_seed(3)
    xc, rc, xq, rq = torch.randn(32, 12), torch.randn(32, 12), torch.randn(16, 12), torch.randn(16, 12)
    for alpha_q in (0.0, 0.1, 1.0):
        paper_value = paper_training.weighted_loss(
            paper_training.recon_loss(xc, rc), paper_training.recon_loss(xq, rq), 32, 16, alpha_q
        )
        ours = weighted_loss(reconstruction_loss(xc, rc), reconstruction_loss(xq, rq), 32, 16, alpha_q)
        assert torch.equal(ours, paper_value)
    assert torch.equal(squared_error(xc, rc), paper_training.squared_error(xc, rc))


def test_info_nce_and_positive_masks_match_the_paper():
    torch.manual_seed(4)
    pairs = _random_pairs(n_queries=40, n_docs=60, seed=4)
    positives = positive_sets(pairs)
    assert positives == paper_contrastive.build_pos_dict(pairs)
    batch = pairs[np.random.default_rng(4).choice(len(pairs), 24, replace=False)]
    query_rows, doc_rows = batch[:, 0], batch[:, 1]
    corpus_rows = np.random.default_rng(5).choice(60, 32, replace=False)
    in_batch = positive_mask(query_rows, doc_rows, positives, skip_diagonal=True)
    in_corpus = positive_mask(query_rows, corpus_rows, positives)
    paper_in_batch = paper_contrastive.build_pos_mask(
        torch.from_numpy(query_rows), torch.from_numpy(doc_rows), positives, CPU, skip_diagonal=True
    )
    paper_in_corpus = paper_contrastive.build_pos_mask(query_rows, corpus_rows, positives, CPU)
    assert torch.equal(in_batch, paper_in_batch) and torch.equal(in_corpus, paper_in_corpus)
    assert in_corpus.any()

    codes = [torch.relu(torch.randn(n, 36)).requires_grad_() for n in (24, 24, 32)]
    with torch.no_grad():
        codes[2][0].zero_()  # an empty code
    ours = info_nce_loss(*codes, in_batch, in_corpus, temperature=0.05)
    ours.backward()
    our_grads = [code.grad.clone() for code in codes]
    for code in codes:
        code.grad = None
    paper_value = paper_contrastive.info_nce(
        codes[0],
        codes[1],
        z_neg=codes[2],
        pos_mask=paper_in_batch,
        pos_mask_neg=paper_in_corpus,
        temperature=0.05,
    )
    paper_value.backward()
    assert torch.equal(ours, paper_value)
    for ours_grad, code in zip(our_grads, codes, strict=True):
        assert torch.equal(ours_grad, code.grad)


def test_positive_pairs_match_load_pairs(tmp_path):
    rows = [
        (f"q{q}", f"d{d}", s)
        for q, d, s in [(1, 2, 1), (0, 5, 2), (1, 9, 0), (7, 1, 1), (0, 1, 1), (2, 3, 2)]
    ]
    qrels = pd.DataFrame(rows, columns=["query_id", "doc_id", "score"])
    path = tmp_path / "qrels.tsv"
    qrels.rename(columns={"query_id": "query-id", "doc_id": "corpus-id"}).to_csv(path, sep="\t", index=False)
    query_ids, doc_ids = ["q0", "q1", "q2"], ["d1", "d2", "d5", "d9"]  # q7 and d3 are unknown
    paper_pairs, _ = paper_contrastive.load_pairs(
        pd.DataFrame({"id": query_ids}), pd.DataFrame(index=doc_ids), str(path)
    )
    ours = positive_pairs(
        qrels, {q: i for i, q in enumerate(query_ids)}, {d: i for i, d in enumerate(doc_ids)}
    )
    assert np.array_equal(ours, paper_pairs)


@pytest.mark.parametrize(
    ("n_queries", "batch_size", "seed", "drop_last"),
    [(50, 16, 42, True), (50, 16, 41, True), (10, 4, 0, False), (7, 16, 3, False), (300, 64, 43, True)],
)
def test_pair_sampler_matches_the_paper(n_queries, batch_size, seed, drop_last):
    pairs = _random_pairs(n_queries, n_docs=500, seed=seed)
    paper = paper_contrastive.UniqueQueryBatchSampler(pairs, batch_size, seed=seed, drop_last=drop_last)
    ours = PairBatchSampler(pairs, batch_size, seed=seed, drop_last=drop_last)
    assert (len(ours), ours.n_passes) == (len(paper), paper.n_passes)
    draws = 3 * max(len(paper), 1) + 5  # several iterations: the random state must carry over identically
    assert list(itertools.islice(iter_forever(ours), draws)) == list(
        itertools.islice(paper_train.iter_forever(paper), draws)
    )


@pytest.mark.parametrize("k_aux", [None, 6])
def test_training_steps_match_the_paper(k_aux):
    paper, sae = _models(d=12, m=36, k=4, seed=6)
    b_pre = torch.randn(12)
    paper.set_b_pre(b_pre)
    with torch.no_grad():
        sae.b_pre.copy_(b_pre)
    config = TrainingConfig(
        lr=1e-2, lambda_nce=0.5, temperature=0.05, alpha_q=0.1, k_aux=k_aux, aux_coef=0.25
    )
    paper_optimizer = paper_training.build_optimizer(paper, config.lr, 0.0)
    optimizer = build_optimizer(sae, config.lr)
    pairs = _random_pairs(n_queries=30, n_docs=80, seed=6)
    positives = positive_sets(pairs)
    rng = np.random.default_rng(6)
    corpus, queries = torch.randn(80, 12), torch.randn(30, 12)
    dead = (torch.rand(36) < 0.5) if k_aux else None
    for _ in range(3):
        corpus_rows = rng.choice(80, 16, replace=False)
        query_rows = rng.choice(30, 16, replace=False)
        batch = pairs[rng.choice(len(pairs), 12, replace=False)]
        pair_queries, pair_docs = queries[batch[:, 0]], corpus[batch[:, 1]]
        paper_stats = paper_train.train_step(
            paper, paper_optimizer, corpus[corpus_rows], torch.from_numpy(corpus_rows), queries[query_rows],
            pair_queries, pair_docs, torch.from_numpy(batch[:, 0]), torch.from_numpy(batch[:, 1]),
            torch.ones(12), positives, 5e-3, config.lambda_nce, config.temperature, "mse", CPU,
            alpha_q=config.alpha_q, k_aux=k_aux, aux_coef=config.aux_coef, dead_mask=dead,
        )  # fmt: skip
        stats = train_step(
            sae, optimizer, corpus[corpus_rows], corpus_rows, queries[query_rows],
            PairBatch(pair_queries, pair_docs, batch[:, 0], batch[:, 1]), positives, config, dead,
        )  # fmt: skip
        mse_c, mse_q, _, _, nce, l0, fired, negatives, aux = paper_stats
        assert stats["mse_c"].item() == pytest.approx(mse_c, rel=1e-5)
        assert stats["mse_q"].item() == pytest.approx(mse_q, rel=1e-5)
        assert stats["nce"].item() == pytest.approx(nce, rel=1e-5)
        assert stats["l0"].item() == pytest.approx(l0) and stats["negatives"].item() == pytest.approx(
            negatives
        )
        assert torch.equal(stats["fired"], fired)
        if k_aux:
            assert stats["aux"].item() == pytest.approx(aux, rel=1e-5) and aux > 0
    for name, tensor in paper.state_dict().items():
        torch.testing.assert_close(sae.state_dict()[name], tensor, rtol=1e-5, atol=1e-6)


def test_paper_checkpoints_load_into_medsae(tmp_path):
    paper, _ = _models(seed=7)
    paper.set_b_pre(torch.randn(12))
    config = {"input_dim": 12, "hidden": 36, "sparsity_type": "topknorelu", "k": 5, "activation_fn": "relu"}
    (tmp_path / "sae_config.json").write_text(json.dumps(config))
    torch.save(paper.state_dict(), tmp_path / "best_sae.pth")
    sae = from_paper_checkpoint(tmp_path)
    x = torch.randn(20, 12)
    assert torch.equal(sae.encode(x), paper.encode(x))


def test_judgments_follow_the_paper_relevance_matrix():
    data = _graded_dataset(seed=8)
    qrels, queries, uids = _paper_qrels(data)
    relevance = paper_ir.load_qrels_matrix(queries, qrels, pd.DataFrame(index=uids), score_col="score")
    judgments = Judgments.from_dataset(data.dataset, "test")
    assert np.array_equal(judgments.dense_gains(0, 60), relevance.to_numpy(dtype=np.float32))
    assert ranking_metrics(data.queries["test"], data.corpus, judgments)["n_queries"] == 59


def _write_replay_data(folder: Path) -> tuple[EmbeddedDataset, np.ndarray]:
    """Synthetic pickles and qrels in the layout of the paper scripts, and the same data for the package."""
    rng = np.random.default_rng(0)
    n_topics, dim, n_docs = 12, 16, 480
    centres = rng.normal(size=(n_topics, dim))

    def sample(topic: int) -> np.ndarray:
        point = centres[topic] + 0.7 * rng.normal(size=dim)
        return (point / np.linalg.norm(point)).astype(np.float32)

    doc_topics = rng.integers(0, n_topics, n_docs)
    docs = np.stack([sample(topic) for topic in doc_topics])
    doc_ids = [f"doc{i}" for i in range(n_docs)]
    queries, qrels, offset = {}, {}, 0
    for split, n in {"train": 120, "val": 30, "test": 40}.items():
        topics = rng.integers(0, n_topics, n)
        ids = [f"q{offset + i}" for i in range(n)]
        offset += n
        rows = []
        for query_id, topic in zip(ids, topics, strict=True):
            same = np.flatnonzero(doc_topics == topic)
            for doc in rng.choice(same, min(len(same), int(rng.integers(1, 6))), replace=False):
                rows.append((query_id, doc_ids[doc], int(rng.integers(1, 3))))  # grades 1 and 2
        queries[split] = (ids, np.stack([sample(topic) for topic in topics]))
        qrels[split] = pd.DataFrame(rows, columns=["query-id", "corpus-id", "score"])
    # split_corpus.py --force-train-positives: training positives in train, 10% of the rest held out
    forced = np.isin(doc_ids, qrels["train"]["corpus-id"].unique())
    rest = np.flatnonzero(~forced)
    order = np.random.default_rng(42).permutation(len(rest))
    n_val = round(len(rest) * 0.1)
    val_rows = np.sort(rest[order[:n_val]])
    train_rows = np.sort(np.concatenate([np.flatnonzero(forced), rest[order[n_val:]]]))

    def pickle(path: Path, ids: list[str], vectors: np.ndarray) -> None:
        pd.DataFrame({"_id": ids, "embeddings": list(vectors)}).to_pickle(path)

    pickle(folder / "corpus_train.pkl", [doc_ids[i] for i in train_rows], docs[train_rows])
    pickle(folder / "corpus_val.pkl", [doc_ids[i] for i in val_rows], docs[val_rows])
    for split, (ids, vectors) in queries.items():
        pickle(folder / f"queries_{split}.pkl", ids, vectors)
        qrels[split].to_csv(folder / f"qrels_{split}.tsv", sep="\t", index=False)
    dataset = RetrievalDataset(
        "synthetic",
        pd.DataFrame({"doc_id": doc_ids, "text": ["t"] * n_docs}),
        {
            split: pd.DataFrame({"query_id": ids, "text": ["t"] * len(ids)})
            for split, (ids, _) in queries.items()
        },
        {
            split: frame.rename(columns={"query-id": "query_id", "corpus-id": "doc_id"})
            for split, frame in qrels.items()
        },
        corpus_splits={"train": train_rows, "val": val_rows},
    )
    data = EmbeddedDataset(
        dataset, "toy/emb", docs, {split: vectors for split, (_, vectors) in queries.items()}
    )
    return data, train_rows


@pytest.mark.parametrize(("lam", "k_aux"), [(0.01, None), (0.0, 12)], ids=["infonce", "auxk_without_infonce"])
def test_trainer_fit_replays_the_paper_training_loop(lam, k_aux, tmp_path, monkeypatch):
    """The paper's train_sae.main() records its batches; Trainer.fit replays them and must end identical."""
    import medsae.training.trainer as trainer_module
    from medsae.training import Trainer

    data, train_rows = _write_replay_data(tmp_path)
    k, hidden, batch, epochs, every, seed = 6, 48, 32, 2, 5, 41
    record = {"corpus": [], "queries": [], "pairs": [], "val": []}
    query_row = {vector.tobytes(): row for row, vector in enumerate(data.queries["train"])}
    real_step, real_score = paper_train.train_step, paper_train.score_eval

    def spy_step(model, opt, x_corp, c_idx_corp, x_q, xq, xd, q_idx_p, c_idx_p, *args, **kwargs):
        record["corpus"].append(c_idx_corp.cpu().numpy().copy())
        record["queries"].append(np.array([query_row[vector.tobytes()] for vector in x_q.cpu().numpy()]))
        if q_idx_p is not None:
            record["pairs"].append((q_idx_p.cpu().numpy().copy(), c_idx_p.cpu().numpy().copy()))
        return real_step(model, opt, x_corp, c_idx_corp, x_q, xq, xd, q_idx_p, c_idx_p, *args, **kwargs)

    def spy_score(*args, **kwargs):
        record["val"].append(real_score(*args, **kwargs))
        return record["val"][-1]

    monkeypatch.setattr(paper_train, "train_step", spy_step)
    monkeypatch.setattr(paper_train, "score_eval", spy_score)
    monkeypatch.setattr(paper_train, "_plot_curves", lambda *args, **kwargs: None)
    run = tmp_path / "paper_run"
    argv = [
        "train_sae", "--train-corpus", tmp_path / "corpus_train.pkl",
        "--val-corpus", tmp_path / "corpus_val.pkl", "--train-queries", tmp_path / "queries_train.pkl",
        "--val-queries", tmp_path / "queries_val.pkl",
        "--test-queries", tmp_path / "queries_test.pkl", "--train-qrels", tmp_path / "qrels_train.tsv",
        "--val-qrels", tmp_path / "qrels_val.tsv", "--test-qrels", tmp_path / "qrels_test.tsv",
        "--corpus-id-col", "_id", "--query-id-col", "_id", "--score-col", "score",
        "--sparsity-type", "topknorelu", "--activation-fn", "relu", "--init", "default",
        "--query-recon-weight", "0.1", "--lr", "5e-4", "--batch", batch, "--epochs", epochs,
        "--val-every-steps", every, "--hidden", hidden, "--k", k, "--lambda", lam, "--temperature", "0.05",
        "--seed", seed, "--experiment-folder", run, "--no-cuda",
        *(["--k-aux", k_aux] if k_aux else []),
    ]  # fmt: skip
    monkeypatch.setattr(sys, "argv", [str(arg) for arg in argv])
    paper_train.main()
    paper_best = torch.load(run / "best_sae.pth", weights_only=True)
    paper_summary = json.loads((run / "summary.json").read_text())

    steps_per_epoch = len(train_rows) // batch
    built = []

    class ReplayBatches:
        """The batches of the paper run: training documents first, then training queries."""

        def __init__(self, n, batch_size, generator):
            self.documents = not built
            built.append(self)
            self.n, self.batch_size, self.epoch = n, batch_size, 0

        def __len__(self):
            return self.n // self.batch_size

        def __iter__(self):
            if self.documents:
                start, self.epoch = self.epoch * steps_per_epoch, self.epoch + 1
                yield from map(torch.from_numpy, record["corpus"][start : start + steps_per_epoch])
            else:
                yield from map(torch.from_numpy, record["queries"])

    class ReplayPairs:
        def __init__(self, pairs, batch_size, seed=42, drop_last=True):
            index = {(int(q), int(d)): i for i, (q, d) in enumerate(pairs.tolist())}
            self.batches = [
                [index[(int(q), int(train_rows[c]))] for q, c in zip(qs, cs, strict=True)]
                for qs, cs in record["pairs"]
            ]

        def __iter__(self):
            yield from self.batches

    monkeypatch.setattr(trainer_module, "ShuffledBatches", ReplayBatches)
    monkeypatch.setattr(trainer_module, "PairBatchSampler", ReplayPairs)
    torch.manual_seed(seed)
    sae = MedSAE(d_in=data.dim, n_latents=hidden, k=k)
    config = TrainingConfig(
        lr=5e-4, batch_size=batch, epochs=epochs, lambda_nce=lam, temperature=0.05, alpha_q=0.1,
        eval_every=every, seed=seed, device="cpu", show_progress=False, eval_batch_size=256, k_aux=k_aux,
    )  # fmt: skip
    trainer = Trainer(sae, config).fit(data)

    ours = [row["val_ndcg@10"] for row in trainer.history]
    assert len(record["val"]) == len(ours) + 4  # main() ends with four evaluations of the saved weights
    np.testing.assert_allclose(ours, record["val"][: len(ours)], rtol=0, atol=1e-12)
    assert trainer.best_step == paper_summary["best_step"]
    for name, tensor in paper_best.items():
        assert torch.equal(trainer.best_model.state_dict()[name].cpu(), tensor), name
    test = evaluate(trainer.best_model, data, "test", batch_size=256)
    assert test["ndcg@10"] == pytest.approx(paper_summary["test_ndcg"], abs=1e-9)
