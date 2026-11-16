"""Trainer: configuration, data plumbing, checkpoint selection, outputs, on tiny synthetic data (CPU)."""

import json
import logging
import math
import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch

import medsae.training.trainer as trainer_module
from medsae.data.dataset import EmbeddedDataset, RetrievalDataset
from medsae.sae import MedSAE, geometric_median
from medsae.training import Trainer, TrainingConfig
from medsae.training.trainer import SELECTION_METRIC, build_optimizer, train_step

HISTORY_KEYS = {
    "step", "epoch", "mse_c", "mse_q", "nce", "negatives", "aux", "l0", "dead_features", SELECTION_METRIC,
    "seconds",
}  # fmt: skip


def _toy_data(
    *,
    seed: int = 0,
    n_topics: int = 16,
    docs_per_topic: int = 20,
    queries_per_topic: int = 4,
    dim: int = 16,
    noise: float = 0.6,
    query_splits: tuple[str, ...] = ("train", "val", "test"),
    held_out_fraction: float = 0.1,
    license: str | None = "cc-by-4.0",
) -> EmbeddedDataset:
    """Points scattered around topic centres; a query is relevant to every document of its topic."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_topics, dim))

    def sample(topic: int, n: int) -> np.ndarray:
        points = centres[topic] + noise * rng.normal(size=(n, dim))
        return points / np.linalg.norm(points, axis=1, keepdims=True)

    docs = np.vstack([sample(topic, docs_per_topic) for topic in range(n_topics)]).astype(np.float32)
    doc_topics = np.repeat(np.arange(n_topics), docs_per_topic)
    corpus = pd.DataFrame({"doc_id": [f"d{i}" for i in range(len(docs))], "text": ["text"] * len(docs)})
    queries, qrels, embeddings = {}, {}, {}
    for split in query_splits:
        ids = [f"{split}{topic}_{j}" for topic in range(n_topics) for j in range(queries_per_topic)]
        rows = [
            (query_id, f"d{doc}", 1)
            for query_id, topic in zip(ids, np.repeat(np.arange(n_topics), queries_per_topic), strict=True)
            for doc in np.flatnonzero(doc_topics == topic)
        ]
        queries[split] = pd.DataFrame({"query_id": ids, "text": ["text"] * len(ids)})
        qrels[split] = pd.DataFrame(rows, columns=["query_id", "doc_id", "score"])
        vectors = [sample(topic, queries_per_topic) for topic in range(n_topics)]
        embeddings[split] = np.vstack(vectors).astype(np.float32)
    held_out = np.sort(rng.choice(len(docs), int(held_out_fraction * len(docs)), replace=False))
    corpus_splits = {"train": np.setdiff1d(np.arange(len(docs)), held_out), "val": held_out}
    dataset = RetrievalDataset(
        "toy", corpus, queries, qrels, corpus_splits=corpus_splits, license=license, hf_ids=["toy/toy-corpus"]
    )
    return EmbeddedDataset(dataset, "toy/embedder", docs, embeddings)


def _config(**overrides) -> TrainingConfig:
    values = dict(
        lr=1e-2,
        batch_size=64,
        epochs=3,
        lambda_nce=0.1,
        eval_every=2,
        seed=0,
        device="cpu",
        show_progress=False,
    )
    return TrainingConfig(**(values | overrides))


def _sae(seed: int = 0, **kwargs) -> MedSAE:
    torch.manual_seed(seed)
    return MedSAE(**({"d_in": 16, "n_latents": 48, "k": 4} | kwargs))


def test_config_defaults_are_the_paper_values():
    config = TrainingConfig()
    assert (config.lr, config.batch_size, config.epochs) == (5e-4, 512, 100)
    assert (config.lambda_nce, config.temperature, config.alpha_q) == (1e-5, 0.05, 0.1)
    assert (config.eval_every, config.seed, config.weight_decay) == (100, 42, 0.0)
    assert config.k_aux is None and config.aux_coef == 1 / 32 and config.corpus_val_fraction == 0.1
    assert config.compute_val_loss is False and config.device is None


@pytest.mark.parametrize(
    "overrides",
    [{"lr": 0}, {"batch_size": 0}, {"epochs": 0}, {"eval_every": 0}, {"temperature": 0}, {"lambda_nce": -1},
     {"alpha_q": -0.1}, {"k_aux": 0}, {"corpus_val_fraction": 1.0}],
)  # fmt: skip
def test_config_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        TrainingConfig(**overrides)


def test_hyperparameters_are_json_and_leave_out_runtime_fields():
    hyperparameters = TrainingConfig(device="cpu", show_progress=False).hyperparameters()
    assert {"device", "show_progress", "eval_batch_size"}.isdisjoint(hyperparameters)
    assert json.loads(json.dumps(hyperparameters))["lambda_nce"] == 1e-5


def test_tiny_run_improves_validation_ndcg_and_writes_the_model(tmp_path):
    data = _toy_data()
    sae = _sae()
    trainer = Trainer(sae, _config(), output_dir=tmp_path / "run").fit(data)

    history = trainer.history
    assert [row["step"] for row in history] == [2, 4, 6, 8, 10, 12]  # 288 training documents, batches of 64
    assert all(set(row) == HISTORY_KEYS for row in history)
    values = [row[SELECTION_METRIC] for row in history]
    assert trainer.best_metric > values[0] + 0.05
    assert (
        trainer.best_metric == max(values) and trainer.best_step == history[values.index(max(values))]["step"]
    )
    assert trainer.best_model is sae
    assert torch.allclose(sae.decoder.weight.norm(dim=0), torch.ones(48), atol=1e-5)

    output = tmp_path / "run"
    assert {"config.json", "model.safetensors", "README.md", "history.json", "summary.json"} <= {
        path.name for path in output.iterdir()
    }
    summary_text = (output / "summary.json").read_text()
    assert str(tmp_path) not in summary_text and str(tmp_path) not in (output / "config.json").read_text()
    summary = json.loads(summary_text)
    assert summary[f"best_{SELECTION_METRIC}"] == pytest.approx(trainer.best_metric)
    assert summary["selection_split"] == "val" and summary["steps"] == 12 and summary["steps_per_epoch"] == 4
    assert (summary["train_documents"], summary["held_out_documents"]) == (288, 32)
    assert summary["batch_sizes"] == {"documents": 64, "queries": 64, "pairs": 64}
    assert json.loads((output / "history.json").read_text()) == history

    metadata = sae.metadata
    assert metadata["train_dataset"] == "toy" and metadata["datasets"] == ["toy/toy-corpus"]
    assert metadata["license"] == "cc-by-4.0" and metadata["embedder"] == "toy/embedder"
    assert metadata["training"]["lambda_nce"] == 0.1 and metadata["training"]["steps"] == 12
    assert metadata[f"best_{SELECTION_METRIC}"] == pytest.approx(trainer.best_metric)
    assert sae.embedder == "toy/embedder"

    reloaded = MedSAE.from_pretrained(output)
    for name, tensor in sae.state_dict().items():
        assert torch.equal(reloaded.state_dict()[name], tensor.cpu())
    assert reloaded.metadata == json.loads(json.dumps(metadata))
    card = (output / "README.md").read_text()
    assert "license: cc-by-4.0" in card and "toy/toy-corpus" in card


def test_the_first_strictly_best_evaluation_is_kept(monkeypatch):
    data = _toy_data()
    snapshots = []
    real_encode = trainer_module.encode_corpus

    def spy_encode(sae, embeddings, **kwargs):
        if embeddings is data.corpus:
            snapshots.append({name: tensor.clone() for name, tensor in sae.state_dict().items()})
        return real_encode(sae, embeddings, **kwargs)

    scripted = iter([0.1, 0.5, 0.5, 0.3, 0.2, 0.4])
    monkeypatch.setattr(trainer_module, "encode_corpus", spy_encode)
    monkeypatch.setattr(trainer_module, "pool_ndcg", lambda *args, **kwargs: next(scripted))
    trainer = Trainer(_sae(), _config()).fit(data)
    assert trainer.best_metric == 0.5 and trainer.best_step == 4
    for name, tensor in trainer.best_model.state_dict().items():
        assert torch.equal(tensor, snapshots[1][name])


def test_without_selection_the_last_weights_are_kept(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("no evaluation expected")

    monkeypatch.setattr(trainer_module, "pool_ndcg", fail)
    trainer = Trainer(_sae(), _config()).fit(_toy_data(), val_split=None)
    assert math.isnan(trainer.best_metric) and trainer.best_step == 12
    assert all(row[SELECTION_METRIC] is None for row in trainer.history)
    assert trainer.summary["selection_split"] is None
    assert trainer.best_model.metadata[f"best_{SELECTION_METRIC}"] is None


def test_missing_validation_queries_fall_back_to_training_queries(caplog):
    data = _toy_data(query_splits=("train", "test"))
    with caplog.at_level(logging.WARNING, logger="medsae.training.trainer"):
        trainer = Trainer(_sae(), _config()).fit(data)
    assert "selected on the 'train' queries" in caplog.text
    assert trainer.summary["selection_split"] == "train" and trainer.best_metric > 0


def test_batch_sizes_are_clamped_on_tiny_datasets(caplog):
    with caplog.at_level(logging.WARNING, logger="medsae.training.trainer"):
        trainer = Trainer(_sae(), _config(batch_size=512, epochs=2)).fit(_toy_data())
    assert "smaller batches" in caplog.text
    assert trainer.summary["batch_sizes"] == {"documents": 288, "queries": 64, "pairs": 64}
    assert trainer.summary["steps"] == 2


def test_reconstruction_only_training_needs_no_pairs():
    data = _toy_data()
    del data.dataset.qrels["train"]  # without InfoNCE the training qrels are not needed
    trainer = Trainer(_sae(), _config(lambda_nce=0.0)).fit(data)
    assert all(row["nce"] is None and row["negatives"] is None for row in trainer.history)
    assert trainer.summary["train_pairs"] == 0 and "pairs" not in trainer.summary["batch_sizes"]
    assert trainer.history[-1]["mse_c"] < trainer.history[0]["mse_c"]


def test_held_out_documents_never_enter_the_loss(monkeypatch):
    data = _toy_data()
    train_rows = set(data.dataset.corpus_splits["train"].tolist())
    seen: set[int] = set()
    real_step = trainer_module.train_step

    def spy_step(sae, optimizer, corpus, corpus_rows, queries, pairs, *args):
        seen.update(corpus_rows.tolist())
        seen.update(pairs.doc_rows.tolist())
        return real_step(sae, optimizer, corpus, corpus_rows, queries, pairs, *args)

    monkeypatch.setattr(trainer_module, "train_step", spy_step)
    Trainer(_sae(), _config(epochs=5)).fit(data)
    assert seen and seen <= train_rows


def test_b_pre_starts_at_the_geometric_median_of_training_documents_and_queries(monkeypatch):
    data = _toy_data()
    captured = {}
    real_init = MedSAE.init_b_pre

    def spy_init(self, x, **kwargs):
        captured.update(x=np.array(x), **kwargs)
        real_init(self, x, **kwargs)

    monkeypatch.setattr(MedSAE, "init_b_pre", spy_init)
    sae = _sae()
    Trainer(sae, _config(epochs=1, seed=7)).fit(data)
    train_rows = data.dataset.corpus_splits["train"]
    expected = np.concatenate([data.corpus[train_rows], data.queries["train"]])
    assert captured["seed"] == 7 and np.array_equal(captured["x"], expected)

    captured.clear()
    trained_b_pre = sae.b_pre.detach().clone()
    Trainer(sae, _config(epochs=1, lr=1e-12)).fit(data)  # fine tuning keeps the learned b_pre
    assert not captured and torch.allclose(sae.b_pre, trained_b_pre)
    assert not np.allclose(trained_b_pre.numpy(), geometric_median(expected, seed=7))


def test_corpus_split_comes_from_split_corpus_when_the_dataset_has_none(monkeypatch):
    data = _toy_data()
    data.dataset.corpus_splits = None
    calls = []

    def fake_split_corpus(dataset, **kwargs):
        calls.append(kwargs)
        return {"train": np.arange(0, 320, 2), "val": np.arange(1, 320, 2)}

    monkeypatch.setitem(
        sys.modules, "medsae.data.splits", types.SimpleNamespace(split_corpus=fake_split_corpus)
    )
    trainer = Trainer(_sae(), _config(corpus_val_fraction=0.2, epochs=1)).fit(data)
    assert calls == [{"val_frac": 0.2, "train_split": "train"}]
    assert (trainer.summary["train_documents"], trainer.summary["held_out_documents"]) == (160, 160)


def test_paper_corpus_split_forces_training_positives_into_training():
    data = _toy_data()
    data.dataset.corpus_splits = None
    qrels = data.dataset.qrels["train"]
    data.dataset.qrels["train"] = qrels[qrels["doc_id"].isin([f"d{i}" for i in range(0, 320, 4)])]
    trainer = Trainer(_sae(), _config(corpus_val_fraction=0.5, epochs=1)).fit(data)
    # 80 positives are forced into training; half of the 240 other documents are held out.
    assert (trainer.summary["train_documents"], trainer.summary["held_out_documents"]) == (200, 120)


def test_no_corpus_split_and_no_held_out_fraction_trains_on_every_document():
    data = _toy_data()
    data.dataset.corpus_splits = None
    trainer = Trainer(_sae(), _config(corpus_val_fraction=0.0, epochs=1)).fit(data)
    assert (trainer.summary["train_documents"], trainer.summary["held_out_documents"]) == (320, 0)


def test_the_sae_must_match_the_embeddings():
    data = _toy_data()
    with pytest.raises(ValueError, match="dimensional"):
        Trainer(_sae(d_in=8), _config()).fit(data)
    with pytest.raises(ValueError, match="declared for"):
        Trainer(_sae(embedder="other/model"), _config()).fit(data)
    data.query_instruction = "query: "
    with pytest.raises(ValueError, match="query instruction"):
        Trainer(_sae(query_instruction="search: "), _config()).fit(data)
    sae = _sae()
    Trainer(sae, _config(epochs=1)).fit(data)
    assert (sae.embedder, sae.query_instruction) == ("toy/embedder", "query: ")


def test_missing_training_data_is_reported():
    data = _toy_data()
    with pytest.raises(ValueError, match="no 'dev' query embeddings"):
        Trainer(_sae(), _config()).fit(data, train_split="dev")
    del data.dataset.qrels["train"]
    with pytest.raises(ValueError, match="lambda_nce=0"):
        Trainer(_sae(), _config()).fit(data)


def test_auxiliary_loss_uses_features_silent_for_an_epoch():
    trainer = Trainer(_sae(n_latents=96, k=2), _config(k_aux=8, eval_every=4)).fit(_toy_data())
    first, second = (row for row in trainer.history if row["epoch"] in (1, 2))
    assert first["dead_features"] > 0  # silent during the whole first epoch
    assert first["aux"] == 0 and second["aux"] > 0  # no dead set before the end of the first epoch


def test_validation_loss_is_monitored_on_request():
    trainer = Trainer(_sae(), _config(compute_val_loss=True)).fit(_toy_data())
    for row in trainer.history:
        assert row["val_mse_c"] > 0 and row["val_mse_q"] > 0 and row["val_nce"] > 0
        assert row["val_loss"] > 0.1 * row["val_nce"]  # weighted reconstruction plus lambda_nce * InfoNCE


def test_training_is_deterministic():
    first = Trainer(_sae(), _config()).fit(_toy_data())
    second = Trainer(_sae(), _config()).fit(_toy_data())
    strip = [{key: value for key, value in row.items() if key != "seconds"} for row in first.history]
    assert strip == [{key: value for key, value in row.items() if key != "seconds"} for row in second.history]
    for name, tensor in first.best_model.state_dict().items():
        assert torch.equal(tensor, second.best_model.state_dict()[name])


def test_a_stale_model_card_is_replaced(tmp_path):
    (tmp_path / "README.md").write_text("---\nlicense: mit\n---\nstale card\n")
    Trainer(_sae(), _config(epochs=1), output_dir=tmp_path).fit(_toy_data())
    card = (tmp_path / "README.md").read_text()
    assert "stale card" not in card and "license: cc-by-4.0" in card


def test_a_dataset_without_license_is_flagged(caplog):
    with caplog.at_level(logging.WARNING, logger="medsae.training.trainer"):
        trainer = Trainer(_sae(), _config(epochs=1)).fit(_toy_data(license=None))
    assert "declares no license" in caplog.text and trainer.best_model.metadata["license"] is None


def test_optimizer_spares_the_decoder_and_b_pre_from_weight_decay():
    sae = _sae()
    decayed, kept = build_optimizer(sae, lr=1e-3, weight_decay=0.1).param_groups
    assert decayed["weight_decay"] == 0.1 and kept["weight_decay"] == 0.0
    assert {id(p) for p in kept["params"]} == {id(sae.decoder.weight), id(sae.b_pre)}
    assert {id(p) for p in decayed["params"]} == {id(sae.encoder.weight), id(sae.encoder.bias)}


def test_train_step_keeps_unit_decoder_columns_and_reports_statistics():
    sae = _sae()
    optimizer = build_optimizer(sae, lr=0.05)
    corpus, queries = torch.randn(8, 16), torch.randn(4, 16)
    stats = train_step(sae, optimizer, corpus, np.arange(8), queries, None, None, _config(lambda_nce=0.0))
    assert set(stats) == {"mse_c", "mse_q", "l0", "fired"}
    assert stats["fired"].dtype == torch.bool and stats["l0"].item() <= 4
    assert torch.allclose(sae.decoder.weight.norm(dim=0), torch.ones(48), atol=1e-5)


def test_numpy_values_in_the_config_reach_the_json_files(tmp_path):
    config = _config(batch_size=np.int64(64), epochs=np.int64(1), lr=np.float32(1e-2), seed=np.int64(3))
    assert type(config.batch_size) is int and type(config.lr) is float and type(config.seed) is int
    trainer = Trainer(_sae(), config, output_dir=tmp_path).fit(_toy_data())
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["config"]["batch_size"] == 64 and summary["completed"] is True
    assert trainer.summary["config"]["seed"] == 3
    with pytest.raises(TypeError, match="batch_size must be an integer"):
        TrainingConfig(batch_size=512.0)


def test_a_run_that_stops_keeps_its_best_checkpoint_on_disk(tmp_path, monkeypatch):
    scripted = iter([0.1, 0.5, 0.3])

    def pool_ndcg(*args, **kwargs):
        value = next(scripted, None)
        if value is None:
            raise MemoryError("out of memory during the fourth evaluation")
        return value

    monkeypatch.setattr(trainer_module, "pool_ndcg", pool_ndcg)
    trainer = Trainer(_sae(), _config(), output_dir=tmp_path)
    with pytest.raises(MemoryError):
        trainer.fit(_toy_data())
    assert trainer.best_step == 4 and trainer.best_metric == 0.5 and trainer.best_model is trainer.sae
    saved = MedSAE.from_pretrained(tmp_path)
    for name, tensor in trainer.best_model.state_dict().items():
        assert torch.equal(saved.state_dict()[name], tensor)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["completed"] is False and summary["best_step"] == 4 and summary["last_step"] == 8
    assert saved.metadata["best_step"] == 4
    assert [row["step"] for row in json.loads((tmp_path / "history.json").read_text())] == [2, 4, 6]


def test_an_interrupt_without_output_folder_keeps_the_best_weights_in_memory(monkeypatch):
    calls = {"n": 0}
    real = trainer_module.pool_ndcg

    def pool_ndcg(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(trainer_module, "pool_ndcg", pool_ndcg)
    trainer = Trainer(_sae(), _config())
    with pytest.raises(KeyboardInterrupt):
        trainer.fit(_toy_data())
    assert trainer.best_model is not None and trainer.best_step in (2, 4)
    assert trainer.summary["completed"] is False


def test_fine_tuning_keeps_the_provenance_and_the_strictest_terms(tmp_path, caplog):
    sae = _sae(metadata={"license": "cc-by-nc-4.0", "datasets": ["clinia/CUREv1"], "train_dataset": "cure",
                         "paper_config": {"k": 4}})  # fmt: skip
    with torch.no_grad():
        sae.b_pre.normal_()  # trained weights
    with caplog.at_level(logging.WARNING, logger="medsae.training.trainer"):
        Trainer(sae, _config(epochs=1), output_dir=tmp_path).fit(_toy_data())
    assert "Review the terms" in caplog.text
    metadata = json.loads((tmp_path / "config.json").read_text())["metadata"]
    assert metadata["datasets"] == ["clinia/CUREv1", "toy/toy-corpus"]
    assert (metadata["license"], metadata["license_name"]) == ("other", "cc-by-nc-4.0-and-cc-by-4.0")
    assert metadata["parents"][0]["train_dataset"] == "cure" and "paper_config" in metadata["parents"][0]
    assert "paper_config" not in metadata and metadata["train_dataset"] == "toy"
    card = (tmp_path / "README.md").read_text()
    assert "license: other" in card and "license_name: cc-by-nc-4.0-and-cc-by-4.0" in card
    assert "clinia/CUREv1" in card and "fine tuned" in card

    unknown = _sae()
    with torch.no_grad():
        unknown.b_pre.normal_()  # trained, with no record of its data
    Trainer(unknown, _config(epochs=1)).fit(_toy_data())
    assert unknown.metadata["license"] is None
    card = unknown.generate_model_card()
    assert card.data.license == "other" and card.data.license_name == "unknown"


def test_a_local_embedder_path_never_reaches_the_shared_files(tmp_path):
    folder = tmp_path / "home" / "jane.doe" / "models" / "bge-large-en-v1.5"
    folder.mkdir(parents=True)
    data = _toy_data()
    data.embedder = str(folder)
    sae = _sae()
    Trainer(sae, _config(epochs=1), output_dir=tmp_path / "run").fit(data)
    config = json.loads((tmp_path / "run" / "config.json").read_text())
    assert config["metadata"]["embedder"] == "bge-large-en-v1.5"
    for name in ("summary.json", "README.md"):
        assert "jane.doe" not in (tmp_path / "run" / name).read_text()
    assert "base_model" not in (tmp_path / "run" / "README.md").read_text()
    with pytest.raises(ValueError, match="local path"):
        sae.push_to_hub("me/my-sae")


def test_embedding_settings_are_recorded_and_checked():
    data = _toy_data()
    data.query_instruction = ""
    data.embedding = {"model": "toy/embedder", "revision": "abc", "query_instruction": "",
                      "document_instruction": "", "normalize": True, "max_seq_length": None}  # fmt: skip
    sae = _sae()
    Trainer(sae, _config(epochs=1)).fit(data)
    assert sae.metadata["embedding"] == {
        "revision": "abc",
        "document_instruction": "",
        "normalize": True,
        "max_seq_length": None,
    }
    data.embedding = dict(data.embedding, normalize=False)
    with pytest.raises(ValueError, match="normalize True against False"):
        Trainer(sae, _config(epochs=1)).fit(data)
