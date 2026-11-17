"""Dense embeddings of queries and documents, computed as the paper computed them.

:class:`Embedder` wraps a ``sentence-transformers`` model and reproduces ``_encode`` of
``paper/src/data_prep/embed_texts_hf_models.py``:

* a query with an instruction is embedded as ``model.encode(instruction + text)`` with
  ``normalize_embeddings=True`` (plain concatenation, no separator added);
* a query without instruction goes through ``model.encode_query``, which applies the model's own ``query``
  prompt when it defines one (the paper models define none);
* a document goes through ``model.encode_document`` (the model's own document prompt, none for the paper
  models) without any prefix;
* inputs are truncated at the model's ``max_seq_length``, 512 tokens for the paper models (never changed).

Default query instructions per model id are in :data:`QUERY_INSTRUCTIONS`: the three BGE v1.5 models
(``BAAI/bge-small-en-v1.5``, ``BAAI/bge-base-en-v1.5`` and ``BAAI/bge-large-en-v1.5``) get
:data:`BGE_QUERY_INSTRUCTION`, the retrieval instruction of their model cards and the one the paper used for
BGE large. Nothing in the paper code records an instruction for ``abhinand/MedEmbed-large-v0.1``, so its
queries follow the code default, no instruction (an open question of the release); other models get none.
A query instruction is ``""`` when there is none, and ``None`` only where it is unknown.
:data:`PAPER_EMBEDDERS` records the paper facts. :class:`GeminiEmbedder` reproduces ``embed_texts.py``
(``gemini-embedding-001`` through the ``google-genai`` SDK). The SAE consumes these vectors as they are:
nothing in the package centers or rescales them.
"""

from __future__ import annotations

import abc
import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm.auto import tqdm

from doctosae._ids import resolve_embedder
from doctosae.data.dataset import EmbeddedDataset, RetrievalDataset, texts_digest

if TYPE_CHECKING:
    from doctosae.sae import DoctoSAE

__all__ = [
    "BGE_QUERY_INSTRUCTION",
    "MODEL_REVISIONS",
    "PAPER_EMBEDDERS",
    "QUERY_INSTRUCTIONS",
    "EmbeddedDataset",
    "Embedder",
    "GeminiEmbedder",
    "default_device",
    "default_query_instruction",
]

logger = logging.getLogger(__name__)

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# The three embedders of the paper: pinned model revision and the query instruction it used ("" for none).
PAPER_EMBEDDERS: dict[str, dict[str, str | None]] = {
    "BAAI/bge-large-en-v1.5": {
        "revision": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        "query_instruction": BGE_QUERY_INSTRUCTION,
    },
    # No instruction is recorded for MedEmbed: the paper script default (none) applies.
    "abhinand/MedEmbed-large-v0.1": {
        "revision": "963121bfb9c625475f65b08fb54990ce9c4e7a1a",
        "query_instruction": "",
    },
    "gemini-embedding-001": {"revision": None, "query_instruction": ""},
}

#: Default query instruction per model id: the retrieval instruction of the BGE v1.5 model cards.
QUERY_INSTRUCTIONS: dict[str, str] = dict.fromkeys(
    ("BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5"), BGE_QUERY_INSTRUCTION
)

#: Pinned revision per model id: the paper commits, and the main branch of the smaller BGE models when this
#: release was prepared.
MODEL_REVISIONS: dict[str, str] = {
    "BAAI/bge-small-en-v1.5": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    "BAAI/bge-base-en-v1.5": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    **{model: info["revision"] for model, info in PAPER_EMBEDDERS.items() if info["revision"]},
}

_CACHE_FILE = "doctosae_cache.json"
# Files whose size and modification time identify the weights of a local model folder.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".onnx")


def default_query_instruction(model: str) -> str:
    """Instruction prepended to queries by default for ``model`` (an id or an alias), ``""`` if none.

    The BGE v1.5 models get the instruction of their model cards (the one the paper used for BGE large);
    other models, MedEmbed included, get none.
    """
    return QUERY_INSTRUCTIONS.get(resolve_embedder(model), "")


def default_device() -> str:
    """``"cuda"`` if available, else ``"mps"`` (Apple silicon), else ``"cpu"``."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _is_gemini(model_id: str) -> bool:
    return model_id.lower().startswith("gemini")


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._]+", "_", text).strip("_") or "dataset"


def _as_texts(texts: Sequence[str] | str) -> list[str]:
    """A list of strings; a single string is one text, not a sequence of characters."""
    if isinstance(texts, str):
        return [texts]
    texts = list(texts)
    for position, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(f"texts must be strings, got {type(text).__name__} at position {position}")
    return texts


def _weights_fingerprint(source: str, revision: str | None, cache_folder: str | None) -> str | None:
    """What identifies the weights a model id or folder loads, without loading them; ``None`` if unknown.

    A local folder gives the size and modification time of its weight files, a Hub id its pinned revision or
    else the commit its main branch points to in the local Hub cache.
    """
    path = Path(source).expanduser()
    if path.is_dir():
        digest = hashlib.sha256()
        files = sorted(item for item in path.rglob("*") if item.is_file() and item.suffix in _WEIGHT_SUFFIXES)
        for item in files:
            stat = item.stat()
            digest.update(f"{item.relative_to(path)}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
        return f"local:{digest.hexdigest()[:16]}" if files else None
    if revision and re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    from huggingface_hub import try_to_load_from_cache

    for filename in ("config.json", "modules.json"):
        try:
            cached = try_to_load_from_cache(source, filename, cache_dir=cache_folder, revision=revision)
        except Exception:  # an id the Hub client refuses: nothing to identify
            return None
        if isinstance(cached, str):
            return Path(cached).parent.name  # .../snapshots/<commit>/config.json
    return None


class _BaseEmbedder(abc.ABC):
    """Dataset embedding with an on disk cache, shared by every embedder."""

    model_id: str
    query_instruction: str  # "" for none
    document_instruction: str  # "" for none

    @abc.abstractmethod
    def encode_queries(self, texts: Sequence[str] | str) -> np.ndarray:
        """``float32`` array ``[n, d]``, one row per query."""

    @abc.abstractmethod
    def encode_documents(self, texts: Sequence[str] | str) -> np.ndarray:
        """``float32`` array ``[n, d]``, one row per document."""

    @property
    @abc.abstractmethod
    def config(self) -> dict[str, Any]:
        """Every setting that changes the vectors (not the device or the batch size)."""

    @property
    def cache_name(self) -> str:
        """Folder name of this embedder's cached embeddings: model name plus a hash of :attr:`config`."""
        digest = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()[:12]
        return f"{_slug(self.model_id.rsplit('/', 1)[-1])}-{digest}"

    def weights_fingerprint(self) -> str | None:
        """What identifies the weights in use (a commit, the files of a local folder), ``None`` if unknown."""
        return None

    def _prepare(self) -> None:  # noqa: B027 (a no op unless overridden)
        """Load what embedding needs (the model of a local embedder); a no op for API embedders."""

    @classmethod
    def from_sae(cls, sae: DoctoSAE, **kwargs: Any) -> Embedder | GeminiEmbedder:
        """The embedder an SAE was trained on, as recorded in its config.

        ``sae.embedder`` gives the model, ``sae.query_instruction`` the instruction (``""`` for none) and
        ``sae.metadata["embedding"]``, written by the trainer, the other settings (revision, document
        instruction, normalization, truncation). An SAE that does not record its query instruction gets the
        default of the model, with a warning. Keyword arguments override the stored values or set the other
        options (``device``, ``batch_size``...). Returns a :class:`GeminiEmbedder` for an SAE trained on
        Gemini embeddings.
        """
        if not sae.embedder:
            raise ValueError(
                "This SAE does not record its embedder (sae.embedder is None): use Embedder(model)"
            )
        model = resolve_embedder(sae.embedder)
        if "query_instruction" not in kwargs:
            if sae.query_instruction is not None:
                kwargs["query_instruction"] = sae.query_instruction
            else:
                logger.warning(
                    "This SAE does not record how its training queries were embedded: queries get the "
                    "default instruction of %s (%r). Pass query_instruction=... if they were embedded "
                    "otherwise.",
                    model,
                    default_query_instruction(model),
                )
        gemini = _is_gemini(model)
        recorded = sae.metadata.get("embedding") or {}
        keys = (
            ("document_instruction",)
            if gemini
            else ("revision", "document_instruction", "normalize", "max_seq_length")
        )
        for key in keys:
            if key in recorded and key not in kwargs:
                kwargs[key] = recorded[key]
        return GeminiEmbedder(model, **kwargs) if gemini else Embedder(model, **kwargs)

    def embed_dataset(
        self,
        dataset: RetrievalDataset,
        splits: Sequence[str] | str | None = None,
        cache_dir: str | Path | None = None,
    ) -> EmbeddedDataset:
        """Embed the corpus and the queries of ``splits`` (every split by default).

        With ``cache_dir``, embeddings are stored in ``cache_dir/<dataset>/<cache_name>/`` with
        :meth:`EmbeddedDataset.save` and reused by later calls with the same settings, the same weights and
        the same texts; splits missing from the cache are embedded and added. The corpus is saved as soon as
        it is embedded and each file is replaced atomically, so an interrupted run does not lose it. When
        the model has to be loaded to embed a missing split and its weights differ from those of the cached
        vectors, everything is embedded again.
        """
        wanted = list(dataset.splits) if splits is None else _as_texts(splits)
        unknown = [split for split in wanted if split not in dataset.queries]
        if unknown:
            raise KeyError(f"{dataset.name} has no split {unknown}; available: {dataset.splits}")
        folder = None if cache_dir is None else Path(cache_dir) / _slug(dataset.name) / self.cache_name
        digests = {"corpus": texts_digest(dataset.corpus["doc_id"].tolist(), dataset.corpus["text"].tolist())}
        for split, frame in dataset.queries.items():
            digests[split] = texts_digest(frame["query_id"].tolist(), frame["text"].tolist())

        weights = self.weights_fingerprint() if folder is not None else None
        stored = self._load_cache(folder, dataset, digests, weights) if folder is not None else None
        missing = [split for split in wanted if stored is None or split not in stored.queries]
        if stored is not None and missing:
            self._prepare()  # the model about to embed the missing splits must have made the cached vectors
            loaded = self.weights_fingerprint()
            if weights is not None and loaded is not None and loaded != weights:
                logger.warning(
                    "The weights of %s changed since the embeddings cached in %s: all is embedded again.",
                    self.model_id,
                    folder,
                )
                stored, missing = None, list(wanted)
            weights = loaded or weights
        if stored is None:
            logger.info(
                "Embedding %d documents of %s with %s", len(dataset.corpus), dataset.name, self.model_id
            )
            corpus = self.encode_documents(dataset.corpus["text"].tolist())
            logger.info("Corpus embeddings: shape %s, mean L2 norm %.4f", corpus.shape, _mean_norm(corpus))
            stored = self._embedded(dataset, corpus, {})
            weights = self.weights_fingerprint() if folder is not None else None
            self._save(stored, folder, digests, weights, ["corpus"])
        if missing:
            queries = dict(stored.queries)
            for split in missing:
                logger.info("Embedding %d %s queries of %s", len(dataset.queries[split]), split, dataset.name)
                queries[split] = self.encode_queries(dataset.queries[split]["text"].tolist())
            stored = self._embedded(dataset, stored.corpus, queries)
            self._save(stored, folder, digests, weights, missing)
        return self._embedded(dataset, stored.corpus, {split: stored.queries[split] for split in wanted})

    def _embedded(
        self, dataset: RetrievalDataset, corpus: np.ndarray, queries: dict[str, np.ndarray]
    ) -> EmbeddedDataset:
        return EmbeddedDataset(
            dataset=dataset,
            embedder=self.model_id,
            corpus=corpus,
            queries=queries,
            query_instruction=self.query_instruction,
            embedding=dict(self.config),
        )

    def _load_cache(
        self, folder: Path, dataset: RetrievalDataset, digests: dict[str, str], weights: str | None
    ) -> EmbeddedDataset | None:
        """Cached embeddings of ``dataset``, keeping only the parts whose texts did not change."""
        record_path = folder / _CACHE_FILE
        if not record_path.is_file():
            return None
        try:
            record = json.loads(record_path.read_text())
            if (
                record.get("config") != self.config
                or record.get("texts", {}).get("corpus") != digests["corpus"]
            ):
                raise ValueError("other embedder settings or other documents")
            cached_weights = record.get("weights")
            if weights is not None and cached_weights is not None and weights != cached_weights:
                raise ValueError(f"other weights of {self.model_id}")
            # Query splits of other ids or texts (another split seed...) are left out; the corpus applies.
            stored = EmbeddedDataset.load(folder, dataset, strict=False)
        except (OSError, ValueError, KeyError) as error:
            logger.warning("Ignoring the embeddings cached in %s: %s", folder, error)
            return None
        if weights is None or cached_weights is None:
            logger.warning(
                "Cannot check that the embeddings cached in %s come from the current weights of %s: they are "
                "reused.",
                folder,
                self.model_id,
            )
        kept = {
            split: array
            for split, array in stored.queries.items()
            if record["texts"].get(split) == digests[split]
        }
        logger.info("Reusing embeddings cached in %s (query splits: %s)", folder, sorted(kept) or "none")
        return self._embedded(dataset, stored.corpus, kept)

    def _save(
        self,
        embedded: EmbeddedDataset,
        folder: Path | None,
        digests: dict[str, str],
        weights: str | None,
        parts: Sequence[str],
    ) -> None:
        """Write the arrays of ``parts``, then the record of every part whose file is in place."""
        if folder is None:
            return
        embedded.save(folder, parts=parts)
        texts = {part: digests[part] for part in ("corpus", *embedded.queries)}
        record = json.dumps({"config": self.config, "weights": weights, "texts": texts}, indent=2)
        temporary = folder / f".{_CACHE_FILE}.partial"
        temporary.write_text(record)
        os.replace(temporary, folder / _CACHE_FILE)


def _mean_norm(vectors: np.ndarray) -> float:
    return float(np.linalg.norm(vectors, axis=1).mean()) if len(vectors) else float("nan")


def _load_sentence_transformer(model_id: str, **kwargs: Any) -> Any:
    """Import sentence transformers lazily: it is slow to import and only needed to embed."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id, **kwargs)


class Embedder(_BaseEmbedder):
    """Sentence transformers embedder reproducing how the paper embedded queries and documents.

    Args:
        model: Hub id, local folder or alias (``"bge"``, ``"medembed"``) of a sentence transformers model.
        query_instruction: Text prepended to every query. ``None`` (default) picks the default of the model id
            (:data:`QUERY_INSTRUCTIONS`): the retrieval instruction for the BGE v1.5 models, none for other
            models. Pass ``""`` to embed queries without instruction.
        document_instruction: Text prepended to every document. The paper used none.
        device: ``"cuda"``, ``"mps"`` or ``"cpu"``; the first available, in that order, by default.
        batch_size: Texts per forward pass. The paper used 32; the batch size only changes float rounding.
        normalize: L2 normalize the vectors. The paper models end with a normalization layer, so for them
            this only changes rounding in the last bit.
        max_seq_length: Truncation length in tokens; ``None`` keeps the model's own (512 for the paper
            models, which the paper never changed).
        show_progress: Show progress bars.
        revision: Model revision. Known models default to a pinned commit (:data:`MODEL_REVISIONS`), the
            commit of the paper for the paper models.
        cache_folder: Folder where sentence transformers stores downloaded models.
        name: Hub id of the model held in the local folder ``model``. It is what embeddings and checkpoints
            record, and it picks the default query instruction, so that a local copy of
            ``BAAI/bge-large-en-v1.5`` behaves like the Hub model and no local path is ever shared.

    The model is loaded on first use, so building an ``Embedder`` is cheap and an embedding cache hit in
    :meth:`embed_dataset` never loads it.
    """

    def __init__(
        self,
        model: str = "BAAI/bge-large-en-v1.5",
        query_instruction: str | None = None,
        document_instruction: str | None = None,
        device: str | None = None,
        batch_size: int = 64,
        normalize: bool = True,
        max_seq_length: int | None = None,
        show_progress: bool = True,
        *,
        revision: str | None = None,
        cache_folder: str | Path | None = None,
        name: str | None = None,
    ) -> None:
        local = Path(model).expanduser().is_dir()
        if name is not None and not local:
            raise ValueError(f"name names the model of a local folder, and {model!r} is not a local folder")
        self.model_id = resolve_embedder(name if name is not None else model)
        if _is_gemini(self.model_id):
            raise ValueError(f"{self.model_id} is served through an API: use doctosae.embed.GeminiEmbedder")
        self.source = str(Path(model).expanduser()) if local else self.model_id
        self.query_instruction = (
            default_query_instruction(self.model_id) if query_instruction is None else query_instruction
        )
        self.document_instruction = document_instruction or ""
        # A local folder has no Hub revision: the pinned commit of its namesake would not describe it.
        self.revision = revision or (None if local else MODEL_REVISIONS.get(self.model_id))
        self.device = device or default_device()
        self.batch_size = int(batch_size)
        self.normalize = bool(normalize)
        self.max_seq_length = max_seq_length
        self.show_progress = show_progress
        self.cache_folder = None if cache_folder is None else str(cache_folder)
        self._model: Any = None

    def __repr__(self) -> str:
        return (
            f"Embedder(model={self.model_id!r}, query_instruction={self.query_instruction!r}, "
            f"device={self.device!r}, batch_size={self.batch_size})"
        )

    @property
    def config(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "revision": self.revision,
            "query_instruction": self.query_instruction,
            "document_instruction": self.document_instruction,
            "normalize": self.normalize,
            "max_seq_length": self.max_seq_length,
        }

    @property
    def model(self) -> Any:
        """The underlying ``SentenceTransformer``, loaded on first use."""
        if self._model is None:
            logger.info(
                "Loading %s (revision %s) on %s", self.model_id, self.revision or "latest", self.device
            )
            self._model = _load_sentence_transformer(
                self.source, device=self.device, revision=self.revision, cache_folder=self.cache_folder
            )
            if self.max_seq_length is not None:
                self._model.max_seq_length = self.max_seq_length
        return self._model

    def weights_fingerprint(self) -> str | None:
        return _weights_fingerprint(self.source, self.revision, self.cache_folder)

    def _prepare(self) -> None:
        self.model  # noqa: B018 (loads the model)

    @property
    def dim(self) -> int:
        """Embedding dimension ``d``."""
        model = self.model
        getter = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        return int(getter())

    def encode_queries(self, texts: Sequence[str] | str) -> np.ndarray:
        """Embed queries: ``instruction + text`` when an instruction is set, else ``encode_query``."""
        texts = _as_texts(texts)
        if self.query_instruction:
            return self._encode("encode", [self.query_instruction + text for text in texts])
        return self._encode("encode_query", texts)

    def encode_documents(self, texts: Sequence[str] | str) -> np.ndarray:
        """Embed documents with ``encode_document``, or ``encode`` on ``document_instruction + text``."""
        texts = _as_texts(texts)
        if self.document_instruction:
            return self._encode("encode", [self.document_instruction + text for text in texts])
        return self._encode("encode_document", texts)

    def _encode(self, method: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        model = self.model
        # sentence transformers before 5.0 has no encode_query or encode_document: plain encode, as the paper.
        encode = getattr(model, method, model.encode)
        vectors = encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


class GeminiEmbedder(_BaseEmbedder):
    """``gemini-embedding-001`` through the ``google-genai`` SDK, as ``paper/src/data_prep/embed_texts.py``.

    The paper sends queries and documents through the same call, without task type, title or output size
    (so 3,072 dimensions), 250 texts per request with a one second pause between requests, and keeps the raw
    vectors (no normalization). Failed requests are retried, which does not change the vectors.

    Credentials come from the environment. With ``vertexai=True`` (the paper setup): Application Default
    Credentials, ``project`` or ``GOOGLE_CLOUD_PROJECT``, ``location`` or ``GOOGLE_CLOUD_LOCATION``
    (``europe-west4`` by default, as in the paper code). With ``vertexai=False``: the Gemini Developer API
    with ``api_key`` or ``GEMINI_API_KEY``. Requires ``pip install "doctosae[gemini]"``. Check the Gemini API
    terms before publishing SAE weights trained on its embeddings.

    Args:
        model: Gemini embedding model.
        query_instruction: Text prepended to every query (none in the paper, the default).
        document_instruction: Text prepended to every document (none in the paper, the default).
        batch_size: Texts per request (lower it for long texts: a request is capped at 20k tokens).
        pause: Seconds between two requests.
        max_retries: Retries of a failed request, with exponential backoff.
        show_progress: Show a progress bar.
        vertexai: Use Vertex AI (paper) rather than the Gemini Developer API.
        project: Google Cloud project for Vertex AI.
        location: Google Cloud region for Vertex AI.
        api_key: Gemini Developer API key.
    """

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        query_instruction: str | None = None,
        document_instruction: str | None = None,
        batch_size: int = 250,
        pause: float = 1.0,
        max_retries: int = 3,
        show_progress: bool = True,
        *,
        vertexai: bool = True,
        project: str | None = None,
        location: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model_id = resolve_embedder(model)
        self.query_instruction = query_instruction or ""
        self.document_instruction = document_instruction or ""
        self.batch_size = int(batch_size)
        self.pause = float(pause)
        self.max_retries = int(max_retries)
        self.show_progress = show_progress
        self.vertexai = vertexai
        self.project = project
        self.location = location
        self.api_key = api_key
        self._client: Any = None
        self._dim = 0

    def __repr__(self) -> str:
        return (
            f"GeminiEmbedder(model={self.model_id!r}, vertexai={self.vertexai}, batch_size={self.batch_size})"
        )

    @property
    def config(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "query_instruction": self.query_instruction,
            "document_instruction": self.document_instruction,
            "normalize": False,
        }

    def weights_fingerprint(self) -> str | None:
        return self.model_id  # a versioned API model: the name identifies the weights

    def encode_queries(self, texts: Sequence[str] | str) -> np.ndarray:
        texts = _as_texts(texts)
        return self._embed([self.query_instruction + text for text in texts])

    def encode_documents(self, texts: Sequence[str] | str) -> np.ndarray:
        texts = _as_texts(texts)
        return self._embed([self.document_instruction + text for text in texts])

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai
            except ImportError as error:
                raise ImportError(
                    'GeminiEmbedder needs google-genai: pip install "doctosae[gemini]"'
                ) from error
            if self.vertexai:
                self._client = genai.Client(
                    vertexai=True,
                    project=self.project or os.environ.get("GOOGLE_CLOUD_PROJECT") or None,
                    location=self.location or os.environ.get("GOOGLE_CLOUD_LOCATION", "europe-west4"),
                )
            else:
                self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        client = self._get_client()
        vectors: list[np.ndarray] = []
        starts = range(0, len(texts), self.batch_size)
        for start in tqdm(starts, desc=self.model_id, unit="request", disable=not self.show_progress):
            batch = texts[start : start + self.batch_size]
            embeddings = self._request(client, batch).embeddings
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"{self.model_id} returned {len(embeddings)} embeddings for {len(batch)} texts"
                )
            vectors.extend(np.asarray(embedding.values, dtype=np.float32) for embedding in embeddings)
            if start + self.batch_size < len(texts):
                time.sleep(self.pause)
        result = np.vstack(vectors)
        self._dim = result.shape[1]
        return result

    def _request(self, client: Any, batch: list[str]) -> Any:
        for attempt in range(self.max_retries + 1):
            try:
                # No config, as in the paper: server default task type and output size.
                return client.models.embed_content(model=self.model_id, contents=batch)
            except Exception as error:  # the SDK raises transport, quota and server errors of several types
                if attempt == self.max_retries:
                    raise
                wait = max(self.pause, 1.0) * 2**attempt
                logger.warning("Gemini request failed (%s); retrying in %.0f s", error, wait)
                time.sleep(wait)
        raise AssertionError("unreachable")
