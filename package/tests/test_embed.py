"""Embedders: paper defaults, the exact encode calls of the paper code, caching, Gemini batching.

Offline tests replace ``SentenceTransformer`` with a deterministic fake and ``google.genai`` with a fake
module; the ``network`` test embeds a few texts with the small ``BAAI/bge-small-en-v1.5``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import types

import numpy as np
import pandas as pd
import pytest

import medsae.embed as embed
from medsae.data import EmbeddedDataset, RetrievalDataset
from medsae.embed import (
    BGE_QUERY_INSTRUCTION,
    Embedder,
    GeminiEmbedder,
    default_device,
    default_query_instruction,
)
from medsae.sae import MedSAE


class FakeSentenceTransformer:
    """Stand in for ``sentence_transformers.SentenceTransformer``: hashed texts to unit vectors."""

    dim = 8

    def __init__(self, model_id, **kwargs):
        self.model_id = model_id
        self.kwargs = kwargs
        self.max_seq_length = 512
        self.calls = []

    def _vectors(
        self, method, texts, batch_size=32, show_progress_bar=None, normalize_embeddings=False, **kwargs
    ):
        self.calls.append((method, list(texts), batch_size, normalize_embeddings))
        seeds = [int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) for text in texts]
        vectors = np.stack([np.random.default_rng(seed).normal(size=self.dim) for seed in seeds])
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def encode(self, texts, **kwargs):
        return self._vectors("encode", texts, **kwargs)

    def encode_query(self, texts, **kwargs):
        return self._vectors("encode_query", texts, **kwargs)

    def encode_document(self, texts, **kwargs):
        return self._vectors("encode_document", texts, **kwargs)

    def get_embedding_dimension(self):
        return self.dim


class OldSentenceTransformer(FakeSentenceTransformer):
    """A sentence transformers model older than 5.0: no encode_query, no encode_document."""

    def __getattribute__(self, name):
        if name in ("encode_query", "encode_document", "get_embedding_dimension"):
            raise AttributeError(name)
        return super().__getattribute__(name)

    def get_sentence_embedding_dimension(self):
        return self.dim


@pytest.fixture
def models(monkeypatch):
    """Every model the embedders load, in order."""
    loaded = []

    def load(model_id, **kwargs):
        loaded.append(FakeSentenceTransformer(model_id, **kwargs))
        return loaded[-1]

    monkeypatch.setattr(embed, "_load_sentence_transformer", load)
    return loaded


@pytest.fixture
def dataset():
    corpus = pd.DataFrame({"doc_id": ["d1", "d2", "d3"], "text": ["statins and cancer", "fiber", "sleep"]})
    queries = {
        "train": pd.DataFrame({"query_id": ["q1", "q2"], "text": ["do statins work", "is fiber good"]}),
        "test": pd.DataFrame({"query_id": ["q3"], "text": ["sleep and health"]}),
    }
    qrels = {
        split: pd.DataFrame({"query_id": frame["query_id"], "doc_id": ["d1", "d2"][: len(frame)], "score": 1})
        for split, frame in queries.items()
    }
    return RetrievalDataset("toy", corpus, queries, qrels)


def test_paper_defaults_per_model():
    bge = Embedder("bge", device="cpu")
    assert bge.model_id == "BAAI/bge-large-en-v1.5"
    assert (
        bge.query_instruction
        == BGE_QUERY_INSTRUCTION
        == "Represent this sentence for searching relevant passages: "
    )
    assert bge.revision == "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
    assert bge.document_instruction == "" and bge.normalize and bge.max_seq_length is None
    medembed = Embedder("medembed", device="cpu")
    assert medembed.model_id == "abhinand/MedEmbed-large-v0.1" and medembed.query_instruction == ""
    other = Embedder("some/model", device="cpu")
    assert other.query_instruction == "" and other.revision is None
    assert Embedder("bge", query_instruction="", device="cpu").query_instruction == ""
    assert default_query_instruction("bge") == BGE_QUERY_INSTRUCTION
    assert default_query_instruction("medembed") == ""
    with pytest.raises(ValueError, match="GeminiEmbedder"):
        Embedder("gemini")


@pytest.mark.parametrize(
    ("model", "revision"),
    [
        ("BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"),
        ("BAAI/bge-base-en-v1.5", "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"),
        ("baai/BGE-large-en-v1.5", "d4aa6901d3a41ba39fb536a557fa166f842b0e09"),
    ],
)
def test_every_bge_model_gets_the_instruction_of_its_card(model, revision):
    embedder = Embedder(model, device="cpu")
    assert embedder.model_id.startswith("BAAI/bge-") and embedder.model_id.lower() == model.lower()
    assert embedder.query_instruction == BGE_QUERY_INSTRUCTION == default_query_instruction(model)
    assert embedder.revision == revision


def test_a_local_folder_records_the_hub_id_it_is_given(models, tmp_path):
    folder = tmp_path / "copies" / "bge"
    folder.mkdir(parents=True)
    named = Embedder(str(folder), name="BAAI/bge-large-en-v1.5", device="cpu")
    assert named.model_id == "BAAI/bge-large-en-v1.5" and named.query_instruction == BGE_QUERY_INSTRUCTION
    assert named.revision is None  # a local copy is not the pinned commit of its namesake
    named.encode_documents(["text"])
    assert models[0].model_id == str(folder)  # the weights still come from the folder
    unnamed = Embedder(str(folder), device="cpu")
    assert unnamed.model_id == str(folder) and unnamed.query_instruction == ""
    with pytest.raises(ValueError, match="local folder"):
        Embedder("BAAI/bge-large-en-v1.5", name="BAAI/bge-base-en-v1.5")


def test_queries_with_an_instruction_go_through_encode_with_normalization(models):
    embedder = Embedder("bge", device="cpu", batch_size=16)
    vectors = embedder.encode_queries(["do statins work", "fiber"])
    assert models[0].calls == [
        ("encode", [BGE_QUERY_INSTRUCTION + "do statins work", BGE_QUERY_INSTRUCTION + "fiber"], 16, True)
    ]
    assert vectors.dtype == np.float32 and vectors.shape == (2, 8)
    assert models[0].model_id == "BAAI/bge-large-en-v1.5"
    assert models[0].kwargs == {
        "device": "cpu",
        "revision": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        "cache_folder": None,
    }


def test_queries_without_instruction_and_documents_use_the_model_prompts(models):
    embedder = Embedder("medembed", device="cpu")
    embedder.encode_queries(["a query"])
    embedder.encode_documents(["Title body"])
    assert [call[:2] for call in models[0].calls] == [
        ("encode_query", ["a query"]),
        ("encode_document", ["Title body"]),
    ]
    Embedder("medembed", document_instruction="passage: ", device="cpu").encode_documents(["text"])
    assert models[1].calls[0][:2] == ("encode", ["passage: text"])


def test_inputs_and_outputs(models):
    embedder = Embedder("some/model", device="cpu", normalize=False)
    single = embedder.encode_documents("one text, not characters")
    assert single.shape == (1, 8)
    assert models[0].calls[0][3] is False
    assert embedder.encode_queries([]).shape == (0, 8)
    with pytest.raises(TypeError, match="position 1"):
        embedder.encode_documents(["fine", float("nan")])


def test_model_is_loaded_lazily_with_the_requested_settings(models):
    embedder = Embedder("some/model", device="cpu", max_seq_length=128, revision="abc", cache_folder="models")
    assert not models
    assert embedder.dim == 8
    assert models[0].max_seq_length == 128
    assert models[0].kwargs == {"device": "cpu", "revision": "abc", "cache_folder": "models"}


def test_sentence_transformers_without_encode_query_falls_back_to_encode(monkeypatch):
    monkeypatch.setattr(
        embed, "_load_sentence_transformer", lambda model_id, **kwargs: OldSentenceTransformer(model_id)
    )
    embedder = Embedder("some/model", device="cpu")
    embedder.encode_queries(["q"])
    embedder.encode_documents(["d"])
    assert [call[0] for call in embedder.model.calls] == ["encode", "encode"]
    assert embedder.dim == 8


def test_default_device(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert default_device() == "cpu"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert default_device() == "mps"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert default_device() == "cuda"


def test_embed_dataset_rows_follow_the_frames(models, dataset):
    embedder = Embedder("bge", device="cpu")
    embedded = embedder.embed_dataset(dataset)
    assert isinstance(embedded, EmbeddedDataset)
    assert (
        embedded.embedder == "BAAI/bge-large-en-v1.5" and embedded.query_instruction == BGE_QUERY_INSTRUCTION
    )
    assert embedded.dim == 8 and set(embedded.queries) == {"train", "test"}
    np.testing.assert_array_equal(embedded.corpus, embedder.encode_documents(dataset.corpus["text"].tolist()))
    np.testing.assert_array_equal(embedded.queries["test"], embedder.encode_queries(["sleep and health"]))
    assert set(embedder.embed_dataset(dataset, splits="test").queries) == {"test"}
    with pytest.raises(KeyError, match="no split"):
        embedder.embed_dataset(dataset, splits=["val"])


def test_embed_dataset_cache_is_reused_and_extended(models, dataset, tmp_path):
    first = Embedder("bge", device="cpu").embed_dataset(dataset, splits=["test"], cache_dir=tmp_path)
    assert len(models) == 1

    again = Embedder("bge", device="cpu").embed_dataset(dataset, splits=["test"], cache_dir=tmp_path)
    assert len(models) == 1  # served from the cache, the model is never loaded
    np.testing.assert_array_equal(again.corpus, first.corpus)
    np.testing.assert_array_equal(again.queries["test"], first.queries["test"])

    both = Embedder("bge", device="cpu").embed_dataset(dataset, cache_dir=tmp_path)
    assert [call[1] for call in models[1].calls] == [
        [BGE_QUERY_INSTRUCTION + text for text in dataset.queries["train"]["text"]]
    ]
    assert set(both.queries) == {"train", "test"}
    assert EmbeddedDataset.load(
        tmp_path / "toy" / Embedder("bge", device="cpu").cache_name, dataset
    ).queries.keys() == {
        "train",
        "test",
    }


def test_embed_dataset_cache_follows_texts_and_settings(models, dataset, tmp_path):
    Embedder("bge", device="cpu").embed_dataset(dataset, cache_dir=tmp_path)
    dataset.queries["test"].loc[0, "text"] = "another wording"  # same ids, other text (another language...)
    Embedder("bge", device="cpu").embed_dataset(dataset, cache_dir=tmp_path)
    assert [call[1] for call in models[1].calls] == [[BGE_QUERY_INSTRUCTION + "another wording"]]

    Embedder("bge", query_instruction="", device="cpu").embed_dataset(dataset, cache_dir=tmp_path)
    assert {call[0] for call in models[2].calls} == {"encode_document", "encode_query"}
    assert len(list((tmp_path / "toy").iterdir())) == 2  # one folder per embedder setting

    dataset.corpus.loc[0, "text"] = "edited document"
    Embedder("bge", device="cpu").embed_dataset(dataset, cache_dir=tmp_path)
    assert models[3].calls[0][0] == "encode_document"


def test_embedded_dataset_records_the_embedder_settings(models, dataset, tmp_path):
    embedded = Embedder("bge", device="cpu", max_seq_length=256).embed_dataset(dataset)
    assert embedded.embedding == {
        "model": "BAAI/bge-large-en-v1.5",
        "revision": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        "query_instruction": BGE_QUERY_INSTRUCTION,
        "document_instruction": "",
        "normalize": True,
        "max_seq_length": 256,
    }
    embedded.save(tmp_path / "saved")
    assert EmbeddedDataset.load(tmp_path / "saved", dataset).embedding == embedded.embedding


def test_saved_embeddings_refuse_other_texts_with_the_same_ids(models, dataset, tmp_path, caplog):
    Embedder("bge", device="cpu").embed_dataset(dataset).save(tmp_path / "saved")
    other = RetrievalDataset(
        "toy", dataset.corpus, {s: f.copy() for s, f in dataset.queries.items()}, dataset.qrels
    )
    other.queries["test"].loc[0, "text"] = "the same question in French"
    with pytest.raises(ValueError, match="other test query texts"):
        EmbeddedDataset.load(tmp_path / "saved", other)
    assert set(EmbeddedDataset.load(tmp_path / "saved", other, strict=False).queries) == {"train"}
    edited = RetrievalDataset(
        "toy", dataset.corpus.assign(text=["a", "b", "c"]), dataset.queries, dataset.qrels
    )
    with pytest.raises(ValueError, match="other document texts"):
        EmbeddedDataset.load(tmp_path / "saved", edited)
    meta = json.loads((tmp_path / "saved" / "meta.json").read_text())
    del meta["corpus_texts"], meta["query_texts"]  # a file written before text digests existed
    (tmp_path / "saved" / "meta.json").write_text(json.dumps(meta))
    with caplog.at_level(logging.WARNING, logger="medsae.data.dataset"):
        assert EmbeddedDataset.load(tmp_path / "saved", edited).dim == 8
    assert "no text digests" in caplog.text


def test_embed_dataset_writes_the_corpus_once_and_atomically(models, dataset, tmp_path, monkeypatch):
    written = []
    real_save = np.save

    def spy_save(handle, array):
        written.append(handle.name.rsplit("/", 1)[-1])
        real_save(handle, array)

    monkeypatch.setattr(np, "save", spy_save)
    Embedder("bge", device="cpu").embed_dataset(dataset, splits=["test"], cache_dir=tmp_path)
    Embedder("bge", device="cpu").embed_dataset(dataset, cache_dir=tmp_path)  # adds the train split only
    assert written == [".corpus.npy.partial", ".queries_test.npy.partial", ".queries_train.npy.partial"]
    folder = tmp_path / "toy" / Embedder("bge", device="cpu").cache_name
    assert not list(folder.glob(".*.partial"))


def test_an_interrupted_query_save_keeps_the_cached_corpus(models, dataset, tmp_path, monkeypatch):
    real_save = np.save

    def failing_save(handle, array):
        if "queries" in handle.name:
            raise KeyboardInterrupt
        real_save(handle, array)

    monkeypatch.setattr(np, "save", failing_save)
    with pytest.raises(KeyboardInterrupt):
        Embedder("bge", device="cpu").embed_dataset(dataset, cache_dir=tmp_path)
    monkeypatch.setattr(np, "save", real_save)
    Embedder("bge", device="cpu").embed_dataset(dataset, cache_dir=tmp_path)
    assert {call[0] for call in models[1].calls} == {"encode"}  # queries only: the corpus came from the cache


def test_cached_embeddings_follow_the_weights_of_a_local_folder(models, dataset, tmp_path):
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(b"first weights")
    Embedder(str(folder), device="cpu").embed_dataset(dataset, splits=["test"], cache_dir=tmp_path / "cache")
    (folder / "model.safetensors").write_bytes(b"second, longer weights")  # fine tuned and saved again
    Embedder(str(folder), device="cpu").embed_dataset(dataset, cache_dir=tmp_path / "cache")
    assert [call[0] for call in models[1].calls] == ["encode_document", "encode_query", "encode_query"]


def test_embed_dataset_keeps_cached_documents_when_the_query_split_changes(models, dataset, tmp_path):
    Embedder("bge", device="cpu").embed_dataset(dataset, cache_dir=tmp_path)
    resplit = RetrievalDataset(
        "toy", dataset.corpus, {"test": pd.DataFrame({"query_id": ["q9"], "text": ["new"]})}, {}
    )
    embedded = Embedder("bge", device="cpu").embed_dataset(resplit, cache_dir=tmp_path)
    assert [call[0] for call in models[1].calls] == ["encode"]  # only the new queries are embedded
    assert embedded.queries["test"].shape == (1, 8)
    folder = tmp_path / "toy" / Embedder("bge", device="cpu").cache_name
    assert (folder / "medsae_cache.json").is_file() and (folder / "corpus.npy").is_file()


def test_from_sae_rebuilds_the_training_embedder(models, caplog):
    sae = MedSAE(d_in=8, n_latents=16, k=4, embedder="BAAI/bge-small-en-v1.5", query_instruction="query: ")
    embedder = Embedder.from_sae(sae, device="cpu", batch_size=4)
    assert isinstance(embedder, Embedder)
    assert (embedder.model_id, embedder.query_instruction, embedder.batch_size) == (
        "BAAI/bge-small-en-v1.5",
        "query: ",
        4,
    )
    assert Embedder.from_sae(sae, query_instruction="", device="cpu").query_instruction == ""
    with caplog.at_level(logging.WARNING, logger="medsae.embed"):
        paper = MedSAE(d_in=8, n_latents=16, k=4, embedder="bge")  # instruction unknown: the default
        assert Embedder.from_sae(paper, device="cpu").query_instruction == BGE_QUERY_INSTRUCTION
    assert "does not record how its training queries were embedded" in caplog.text
    assert isinstance(Embedder.from_sae(MedSAE(d_in=8, n_latents=16, k=4, embedder="gemini")), GeminiEmbedder)
    with pytest.raises(ValueError, match="does not record its embedder"):
        Embedder.from_sae(MedSAE(d_in=8, n_latents=16, k=4))


def test_no_instruction_is_recorded_as_an_empty_string_and_survives_a_new_default(models, dataset):
    """Queries embedded without instruction stay so, even for a model whose default now has one."""
    embedded = Embedder("BAAI/bge-small-en-v1.5", query_instruction="", device="cpu").embed_dataset(dataset)
    assert embedded.query_instruction == "" and embedded.embedding["query_instruction"] == ""
    sae = MedSAE(d_in=8, n_latents=16, k=4, embedder=embedded.embedder, query_instruction="")
    assert Embedder.from_sae(sae, device="cpu").query_instruction == ""


def test_from_sae_restores_the_recorded_settings(models):
    sae = MedSAE(
        d_in=8, n_latents=16, k=4, embedder="some/model", query_instruction="",
        metadata={"embedding": {"revision": "abc", "document_instruction": "passage: ", "normalize": False,
                                "max_seq_length": 128}},
    )  # fmt: skip
    embedder = Embedder.from_sae(sae, device="cpu")
    assert (
        embedder.revision,
        embedder.document_instruction,
        embedder.normalize,
        embedder.max_seq_length,
    ) == (
        "abc",
        "passage: ",
        False,
        128,
    )
    assert Embedder.from_sae(sae, device="cpu", normalize=True).normalize  # arguments win


# Gemini, with a fake google.genai module.


@pytest.fixture
def genai(monkeypatch):
    """Install a fake ``google.genai`` whose client returns ``[len(text), position in batch, 1]`` vectors."""

    class Models:
        def __init__(self, client):
            self.client = client

        def embed_content(self, model, contents):
            self.client.requests.append((model, list(contents)))
            if self.client.failures:
                self.client.failures -= 1
                raise ConnectionError("quota")
            values = [[float(len(text)), float(i), 1.0] for i, text in enumerate(contents)]
            return types.SimpleNamespace(embeddings=[types.SimpleNamespace(values=v) for v in values])

    class Client:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.requests = []
            self.failures = 0
            self.models = Models(self)
            Client.instances.append(self)

    module = types.ModuleType("google.genai")
    module.Client = Client
    google = types.ModuleType("google")
    google.genai = module
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", module)
    sleeps = []
    monkeypatch.setattr(embed.time, "sleep", sleeps.append)
    return Client, sleeps


def test_gemini_batches_like_the_paper(genai, monkeypatch):
    client_class, sleeps = genai
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    embedder = GeminiEmbedder(batch_size=2, show_progress=False)
    vectors = embedder.encode_documents(["a", "bb", "ccc", "dddd", "eeeee"])
    client = client_class.instances[0]
    assert client.kwargs == {"vertexai": True, "project": "my-project", "location": "europe-west4"}
    assert [contents for _, contents in client.requests] == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]
    assert {model for model, _ in client.requests} == {"gemini-embedding-001"}
    assert sleeps == [1.0, 1.0]  # one pause between consecutive requests, none after the last
    assert vectors.dtype == np.float32 and vectors[:, 0].tolist() == [1, 2, 3, 4, 5]
    assert embedder.encode_queries([]).shape == (0, 3)


def test_gemini_options_and_retries(genai, caplog):
    client_class, sleeps = genai
    embedder = GeminiEmbedder(
        vertexai=False, api_key="test-key", query_instruction="task: search | ", show_progress=False
    )
    embedder._get_client().failures = 1
    vectors = embedder.encode_queries(["q"])
    client = client_class.instances[0]
    assert client.kwargs == {"api_key": "test-key"}
    assert client.requests == [("gemini-embedding-001", ["task: search | q"])] * 2
    assert vectors.shape == (1, 3) and sleeps == [1.0] and "retrying" in caplog.text
    client.failures = 10
    with pytest.raises(ConnectionError):
        embedder.encode_queries(["q"])


def test_gemini_needs_its_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "google", None)  # makes any import of google fail
    with pytest.raises(ImportError, match=r"medsae\[gemini\]"):
        GeminiEmbedder(show_progress=False).encode_documents(["text"])


# A real model (network).


@pytest.mark.network
def test_bge_small_matches_the_paper_encode_calls():
    embedder = Embedder(
        "BAAI/bge-small-en-v1.5",
        query_instruction=BGE_QUERY_INSTRUCTION,
        device="cpu",
        batch_size=32,
        show_progress=False,
    )
    queries = ["Do cholesterol statin drugs cause breast cancer?", "fiber and colon health"]
    documents = ["Statin Use and Breast Cancer Survival A nationwide cohort study.", "Dietary fiber intake."]
    model = embedder.model
    # paper/src/data_prep/embed_texts_hf_models.py, _encode with --query-instruction, then --mode document
    paper_queries = model.encode(
        [BGE_QUERY_INSTRUCTION + text for text in queries], batch_size=32, normalize_embeddings=True
    )
    paper_documents = model.encode_document(documents, batch_size=32)
    np.testing.assert_array_equal(embedder.encode_queries(queries), paper_queries)
    np.testing.assert_allclose(embedder.encode_documents(documents), paper_documents, atol=1e-6)
    assert embedder.dim == 384
    np.testing.assert_allclose(np.linalg.norm(embedder.encode_documents(documents), axis=1), 1.0, atol=1e-6)
