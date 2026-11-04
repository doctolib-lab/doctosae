"""In memory containers for retrieval benchmarks and their dense embeddings.

Every loader of :mod:`medsae.data` returns a :class:`RetrievalDataset`; :class:`medsae.Embedder` turns it into
an :class:`EmbeddedDataset`, which is what training, evaluation and interpretation consume.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from medsae._ids import embedding_settings

__all__ = ["RetrievalDataset", "EmbeddedDataset", "SPLIT_ORDER", "texts_digest"]

logger = logging.getLogger(__name__)

# Canonical split names; loaders map "dev", "validation" or "eval" to "val".
SPLIT_ORDER = ("train", "val", "test")


@dataclass
class RetrievalDataset:
    """A corpus, queries per split and graded relevance judgments (qrels) per split.

    Attributes:
        name: Registry name (``"cure"``, ``"nfcorpus"``...) or any label for a custom dataset.
        corpus: One row per document with columns ``doc_id`` (str) and ``text`` (the exact string that is
            embedded, for example title and body joined by a space). Optional columns: ``title`` and ``body``
            (shown separately to the Interpreter LLM), ``group`` (article id used to cap evidence at two
            passages per article; defaults to ``title`` when absent) and any theme column (``source_folder``
            for CURE).
        queries: Split name to a frame with columns ``query_id`` (str) and ``text`` (str).
        qrels: Split name to a frame with columns ``query_id`` (str), ``doc_id`` (str) and ``score`` (int).
        corpus_splits: Optional split name to row indices into ``corpus``. ``"train"`` rows feed the
            reconstruction and contrastive terms; ``"val"`` rows are held out. Retrieval always scores against
            the full corpus. When ``None``, :func:`medsae.data.splits.split_corpus` builds the paper split on
            demand.
        positive_threshold: A judgment is a positive (training pair, relevant for recall) when
            ``score >= positive_threshold``. Every benchmark of the paper uses ``score > 0``.
        license: License of the benchmark, reported to users and copied to the cards of models trained on it.
        hf_ids: Hugging Face Hub ids the dataset was built from.
    """

    name: str
    corpus: pd.DataFrame
    queries: dict[str, pd.DataFrame]
    qrels: dict[str, pd.DataFrame]
    corpus_splits: dict[str, np.ndarray] | None = None
    positive_threshold: int = 1
    license: str | None = None
    hf_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for column in ("doc_id", "text"):
            if column not in self.corpus.columns:
                raise ValueError(f"corpus needs a {column!r} column")
        self.corpus = self.corpus.reset_index(drop=True)
        self.corpus["doc_id"] = self.corpus["doc_id"].astype(str)
        for split, frame in self.queries.items():
            missing = {"query_id", "text"} - set(frame.columns)
            if missing:
                raise ValueError(f"queries[{split!r}] lacks columns {sorted(missing)}")
            frame = frame.reset_index(drop=True)
            frame["query_id"] = frame["query_id"].astype(str)
            self.queries[split] = frame
        for split, frame in self.qrels.items():
            missing = {"query_id", "doc_id", "score"} - set(frame.columns)
            if missing:
                raise ValueError(f"qrels[{split!r}] lacks columns {sorted(missing)}")
            frame = frame.reset_index(drop=True)
            frame["query_id"] = frame["query_id"].astype(str)
            frame["doc_id"] = frame["doc_id"].astype(str)
            frame["score"] = frame["score"].astype(int)
            self.qrels[split] = frame

    @property
    def splits(self) -> list[str]:
        """Query splits, in train, val, test order, then any other split alphabetically."""
        known = [split for split in SPLIT_ORDER if split in self.queries]
        return known + sorted(set(self.queries) - set(SPLIT_ORDER))

    @cached_property
    def doc_index(self) -> dict[str, int]:
        """Document id to row position in :attr:`corpus` (first occurrence wins)."""
        index: dict[str, int] = {}
        for row, doc_id in enumerate(self.corpus["doc_id"].tolist()):
            index.setdefault(doc_id, row)
        return index

    def qrels_dict(self, split: str) -> dict[str, dict[str, int]]:
        """``{query_id: {doc_id: score}}`` for one split, every judged pair included."""
        judged: dict[str, dict[str, int]] = {}
        frame = self.qrels[split]
        for query_id, doc_id, score in zip(frame["query_id"], frame["doc_id"], frame["score"], strict=True):
            judged.setdefault(query_id, {})[doc_id] = int(score)
        return judged

    def positives(self, split: str) -> dict[str, set[str]]:
        """``{query_id: {doc_id, ...}}`` of the judgments with ``score >= positive_threshold``."""
        frame = self.qrels[split]
        frame = frame[frame["score"] >= self.positive_threshold]
        result: dict[str, set[str]] = {}
        for query_id, doc_id in zip(frame["query_id"], frame["doc_id"], strict=True):
            result.setdefault(query_id, set()).add(doc_id)
        return result

    def describe(self) -> str:
        """One line summary: corpus size, queries and judgments per split, license."""
        parts = [f"{self.name}: {len(self.corpus):,} documents"]
        for split in self.splits:
            n_qrels = len(self.qrels.get(split, ()))
            parts.append(f"{split} {len(self.queries[split]):,} queries / {n_qrels:,} judgments")
        if self.corpus_splits:
            sizes = ", ".join(f"{name} {len(rows):,}" for name, rows in self.corpus_splits.items())
            parts.append(f"corpus split {sizes}")
        if self.license:
            parts.append(f"license {self.license}")
        return " | ".join(parts)


def _ids_digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]


def texts_digest(ids: Sequence[str], texts: Sequence[str]) -> str:
    """Fingerprint of what gets embedded (ids and texts), so that vectors of other texts are never reused."""
    digest = hashlib.sha256()
    for item, text in zip(ids, texts, strict=True):
        digest.update(f"{item}\x1f{text}\x1e".encode())
    return digest.hexdigest()


def _corpus_digest(dataset: RetrievalDataset) -> str:
    return texts_digest(dataset.corpus["doc_id"].tolist(), dataset.corpus["text"].tolist())


def _queries_digest(dataset: RetrievalDataset, split: str) -> str:
    frame = dataset.queries[split]
    return texts_digest(frame["query_id"].tolist(), frame["text"].tolist())


def _replace_atomically(path: Path, write: Any) -> None:
    """Write a temporary file, then rename it: an interrupted write never leaves half a file behind."""
    temporary = path.with_name(f".{path.name}.partial")
    with open(temporary, "wb") as handle:
        write(handle)
    os.replace(temporary, path)


@dataclass
class EmbeddedDataset:
    """Dense embeddings of a :class:`RetrievalDataset`, row aligned with its frames.

    Attributes:
        dataset: The embedded dataset.
        embedder: Id of the frozen embedding model.
        corpus: ``float32`` array ``[n_documents, d]``, row ``i`` embeds ``dataset.corpus.text[i]``.
        queries: Split name to a ``float32`` array ``[n_queries, d]`` aligned with ``dataset.queries[split]``.
        query_instruction: Instruction prepended to queries before embedding: ``""`` for none, ``None`` when
            unknown (arrays built by hand).
        embedding: Settings of the embedder that change the vectors (model, revision, instructions,
            normalization, truncation), as :attr:`medsae.Embedder.config` records them; ``None`` when unknown.
            Training copies them to the SAE, and evaluation warns when an SAE meets other settings.
    """

    dataset: RetrievalDataset
    embedder: str
    corpus: np.ndarray
    queries: dict[str, np.ndarray]
    query_instruction: str | None = None
    embedding: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.corpus = np.asarray(self.corpus, dtype=np.float32)
        if len(self.corpus) != len(self.dataset.corpus):
            raise ValueError(f"{len(self.corpus)} corpus embeddings for {len(self.dataset.corpus)} documents")
        for split, array in list(self.queries.items()):
            array = np.asarray(array, dtype=np.float32)
            n_queries = len(self.dataset.queries[split])
            if len(array) != n_queries:
                raise ValueError(f"{len(array)} {split} query embeddings for {n_queries} queries")
            self.queries[split] = array

    @property
    def dim(self) -> int:
        return int(self.corpus.shape[1])

    @property
    def settings(self) -> dict[str, Any]:
        """Known settings of these embeddings, for comparison with the ones an SAE was trained on."""
        return embedding_settings(self.embedder, self.query_instruction, self.embedding)

    def save(self, path: str | Path, parts: Iterable[str] | None = None) -> None:
        """Write ``corpus.npy``, ``queries_<split>.npy`` and ``meta.json``.

        ``meta.json`` records the embedder settings and digests of the ids and texts, which guard against
        loading the arrays with another dataset. Every file is written through a temporary file, so an
        interrupted save never leaves half an array. ``parts`` (``"corpus"`` and split names) restricts the
        arrays written to those that changed; ``meta.json`` always describes every part of this object.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        parts = {"corpus", *self.queries} if parts is None else set(parts)
        if "corpus" in parts:
            _replace_atomically(path / "corpus.npy", lambda handle: np.save(handle, self.corpus))
        for split, array in self.queries.items():
            if split in parts:
                _replace_atomically(path / f"queries_{split}.npy", lambda handle, a=array: np.save(handle, a))
        meta = {
            "dataset": self.dataset.name,
            "embedder": self.embedder,
            "query_instruction": self.query_instruction,
            "embedding": self.embedding,
            "dim": self.dim,
            "corpus_ids": _ids_digest(self.dataset.corpus["doc_id"].tolist()),
            "corpus_texts": _corpus_digest(self.dataset),
            "query_ids": {
                split: _ids_digest(self.dataset.queries[split]["query_id"].tolist()) for split in self.queries
            },
            "query_texts": {split: _queries_digest(self.dataset, split) for split in self.queries},
        }
        text = json.dumps(meta, indent=2).encode()
        _replace_atomically(path / "meta.json", lambda handle: handle.write(text))

    @classmethod
    def load(cls, path: str | Path, dataset: RetrievalDataset, *, strict: bool = True) -> EmbeddedDataset:
        """Load embeddings saved by :meth:`save`, checking that they embed the ids and texts of ``dataset``.

        A corpus of other ids or texts always raises ``ValueError``. So does a query split of other ids or
        texts, unless ``strict=False``, which leaves such a split out instead. Files written without text
        digests are loaded with a warning.
        """
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        if meta["corpus_ids"] != _ids_digest(dataset.corpus["doc_id"].tolist()):
            raise ValueError(f"{path} was not computed on this corpus ({dataset.name})")
        if "corpus_texts" not in meta:
            logger.warning(
                "%s records no text digests: it cannot be checked against the texts of %s", path, dataset.name
            )
        elif meta["corpus_texts"] != _corpus_digest(dataset):
            raise ValueError(f"{path} embeds other document texts than {dataset.name} (same ids)")
        queries = {}
        for split, digest in meta["query_ids"].items():
            if split not in dataset.queries:
                continue
            problem = None
            if digest != _ids_digest(dataset.queries[split]["query_id"].tolist()):
                problem = f"{path} {split} queries do not match {dataset.name}"
            elif split in meta.get("query_texts", {}) and meta["query_texts"][split] != _queries_digest(
                dataset, split
            ):
                problem = f"{path} embeds other {split} query texts than {dataset.name} (same ids)"
            if problem is None:
                queries[split] = np.load(path / f"queries_{split}.npy")
            elif strict:
                raise ValueError(problem)
            else:
                logger.info("%s: split left out", problem)
        return cls(
            dataset=dataset,
            embedder=meta["embedder"],
            corpus=np.load(path / "corpus.npy"),
            queries=queries,
            query_instruction=meta.get("query_instruction"),
            embedding=meta.get("embedding"),
        )
