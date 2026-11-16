"""The :class:`Interpreter`: select, describe, score and explain the features of a MedSAE (Section 6)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from medsae._ids import resolve_embedder
from medsae.interpret.activations import (
    ActivationIndex,
    CodedTexts,
    article_keys,
    encode_sparse,
    frame_titles,
)
from medsae.interpret.cards import FeatureCard, load_cards, save_cards
from medsae.interpret.explain import RetrievalExplanation, explain_codes, unit
from medsae.interpret.labeling import label_cards
from medsae.interpret.llm import LLM
from medsae.interpret.prompts import METHODS
from medsae.interpret.scoring import (
    RANK_BAND,
    EvalPool,
    PoolSource,
    ScoreReport,
    exemplars,
    run_activation_prediction,
)
from medsae.interpret.selection import select_features
from medsae.interpret.tfidf import N_TERMS, TfidfTerms
from medsae.metrics import warn_on_other_settings

if TYPE_CHECKING:
    from medsae.data.dataset import EmbeddedDataset
    from medsae.embed import Embedder
    from medsae.sae import MedSAE

__all__ = ["Interpreter"]

logger = logging.getLogger(__name__)


def _other_embedder(sae: MedSAE, data: EmbeddedDataset) -> bool:
    """True when the SAE and the data both name their encoder and the names differ (aliases resolved)."""
    if not (sae.embedder and data.embedder):
        return False
    return resolve_embedder(sae.embedder) != resolve_embedder(data.embedder)


def _pool_source(name: str, frame: pd.DataFrame, codes: sp.csr_matrix) -> PoolSource:
    return PoolSource(
        name=name,
        codes=codes,
        texts=frame["text"].astype(str).tolist(),
        titles=frame_titles(frame),
        articles=article_keys(frame),
    )


class Interpreter:
    """Interpret the features of a :class:`~medsae.MedSAE` on an embedded retrieval dataset.

    The evidence shown for a feature comes from ``data``: the documents of its ``corpus_split`` corpus split
    (the whole corpus when the dataset defines no corpus split) and the queries of ``split``, both
    deduplicated by text. With the SAE trained on CURE and CURE embedded by the same encoder, this is the
    paper setting: the passages of the reconstruction training split and the training queries.

    Args:
        sae: The SAE to interpret.
        data: Embeddings of the dataset, computed by the encoder the SAE was trained on.
        split: Query split used as evidence (``"train"`` in the paper), or ``None`` for documents only.
        device: Device to run the SAE on (the model is moved there); its current device by default.
        corpus_split: Corpus split used as evidence documents when the dataset defines corpus splits; ``None``
            uses the whole corpus.
        batch_size: Embeddings encoded per SAE forward pass.
        show_progress: Show progress bars.

    Example::

        llm = TransformersLLM()  # google/medgemma-4b-it, as in the paper
        interpreter = Interpreter(sae, data)
        cards = interpreter.describe(interpreter.select_features(n=20), llm=llm)
        print(cards[0].to_markdown())
        print(interpreter.score(cards, llm).to_frame())

    The protocol of Table 3 of the paper scores every qualifying feature of the CURE SAE, in the order of
    ``select_features(theme_column="source_folder")`` (most theme specific first) with no ``n``: that order
    drives both seeded draws of :meth:`score`, so another order judges other texts.
    """

    def __init__(
        self,
        sae: MedSAE,
        data: EmbeddedDataset,
        split: str | None = "train",
        device: str | None = None,
        *,
        corpus_split: str | None = "train",
        batch_size: int = 1024,
        show_progress: bool = True,
    ) -> None:
        if data.dim != sae.d_in:
            raise ValueError(f"the SAE reads {sae.d_in} dimensional embeddings, the dataset has {data.dim}")
        if _other_embedder(sae, data):
            logger.warning(
                "the SAE was trained on %s embeddings, the dataset was embedded by %s",
                sae.embedder,
                data.embedder,
            )
        else:
            warn_on_other_settings(sae, data)
        if split is not None and split not in data.queries:
            raise ValueError(
                f"no {split!r} query embeddings (available: {sorted(data.queries)}); "
                "pass split=None to use documents only"
            )
        splits = data.dataset.corpus_splits or {}
        if corpus_split is not None and splits and corpus_split not in splits:
            raise ValueError(f"no {corpus_split!r} corpus split (available: {sorted(splits)})")
        self.sae = sae.to(device) if device is not None else sae
        self.device = device
        self.data = data
        self.split = split
        self.corpus_split = corpus_split
        self.batch_size = int(batch_size)
        self.show_progress = show_progress
        self._embedder: Embedder | None = None

    def __repr__(self) -> str:
        return (
            f"Interpreter(dataset={self.data.dataset.name!r}, n_latents={self.sae.n_latents}, "
            f"split={self.split!r}, corpus_split={self.corpus_split!r})"
        )

    def _encode(self, embeddings: np.ndarray) -> sp.csr_matrix:
        return encode_sparse(self.sae, embeddings, self.batch_size, self.show_progress)

    @cached_property
    def corpus_codes(self) -> sp.csr_matrix:
        """SAE codes of every corpus document of ``data``, in corpus order."""
        return self._encode(self.data.corpus)

    @cached_property
    def index(self) -> ActivationIndex:
        """Evidence documents and queries with their codes."""
        dataset = self.data.dataset
        splits = dataset.corpus_splits or {}
        rows = splits[self.corpus_split] if self.corpus_split is not None and splits else None
        documents = CodedTexts.from_frame(dataset.corpus, self.corpus_codes, rows=rows)
        queries = None
        if self.split is not None:
            query_codes = self._encode(self.data.queries[self.split])
            queries = CodedTexts.from_frame(dataset.queries[self.split], query_codes, id_column="query_id")
        n_queries = 0 if queries is None else len(queries)
        logger.info("evidence: %d documents, %d queries", len(documents), n_queries)
        return ActivationIndex(documents, queries)

    @cached_property
    def tfidf(self) -> TfidfTerms:
        """TF IDF space over the evidence documents then the evidence queries."""
        index = self.index
        return TfidfTerms.fit(index.documents.texts, () if index.queries is None else index.queries.texts)

    def select_features(
        self,
        n: int | None = None,
        min_documents: int = 5,
        min_queries: int = 5,
        firing_rate_band: tuple[float, float] = (0.0, 1.0),
        theme_column: str | None = None,
    ) -> list[int]:
        """Features worth interpreting (Appendix D.1).

        Args:
            n: Return the first ``n`` features (all by default).
            min_documents: The feature must fire on at least this many evidence documents.
            min_queries: And on at least this many evidence queries (when there are queries).
            firing_rate_band: Inclusive band on the document firing rate; the paper kept every rate.
            theme_column: Corpus column holding a theme per document (``source_folder`` for CURE). Features
                are then ordered by ascending theme entropy, most theme specific first; otherwise by index.
        """
        themes = None
        if theme_column is not None:
            corpus = self.data.dataset.corpus
            if theme_column not in corpus.columns:
                raise ValueError(f"no {theme_column!r} column in the corpus (columns {list(corpus.columns)})")
            themes = corpus[theme_column].to_numpy(dtype=object)[self.index.documents.rows].tolist()
        return select_features(self.index, n, min_documents, min_queries, firing_rate_band, themes)

    def describe(
        self,
        features: Sequence[int],
        llm: LLM | None = None,
        n_documents: int = 5,
        n_queries: int = 5,
        n_negatives: int = 2,
        n_terms: int = N_TERMS,
        n_keywords: int = 10,
        max_chars: int | None = 1000,
        batch_size: int = 20,
    ) -> list[FeatureCard]:
        """Build a card per feature: evidence, firing statistics and TF IDF terms, plus an Interpreter label
        when ``llm`` is given.

        Args:
            features: Feature ids, for example from :meth:`select_features`.
            llm: Interpreter LLM; cards stay unlabelled without one.
            n_documents: Top documents shown (at most two per article).
            n_queries: Top queries shown.
            n_negatives: Documents the feature does not fire on, shown last.
            n_terms: TF IDF terms kept.
            n_keywords: Keyword budget asked from the Interpreter LLM.
            max_chars: Characters shown per document body or query in the Interpreter prompt.
            batch_size: Features per Interpreter LLM call.
        """
        index, tfidf = self.index, self.tfidf
        cards = []
        for feature in features:
            feature = int(feature)
            if not 0 <= feature < index.n_latents:
                raise ValueError(f"feature {feature} is outside [0, {index.n_latents})")
            document_rows = index.top_documents(feature, n_documents)
            query_rows = index.top_queries(feature, n_queries)
            cards.append(
                FeatureCard(
                    feature=feature,
                    n_documents=int(index.document_counts[feature]),
                    n_queries=0 if index.query_counts is None else int(index.query_counts[feature]),
                    firing_rate=float(index.firing_rates[feature]),
                    top_documents=index.documents.snippets(feature, document_rows),
                    top_queries=[] if index.queries is None else index.queries.snippets(feature, query_rows),
                    negatives=index.documents.snippets(
                        feature, index.negatives(feature, n_negatives), with_activation=False
                    ),
                    tfidf_terms=tfidf.terms(document_rows, query_rows, n_terms),
                )
            )
        if llm is not None:
            cards = self.label(cards, llm, n_keywords=n_keywords, max_chars=max_chars, batch_size=batch_size)
        return cards

    def label(
        self,
        cards: Sequence[FeatureCard],
        llm: LLM,
        n_keywords: int = 10,
        max_chars: int | None = 1000,
        batch_size: int = 20,
    ) -> list[FeatureCard]:
        """Labelled copies of ``cards`` (topic, description, keywords) from the Interpreter LLM."""
        return label_cards(cards, llm, n_keywords, max_chars, batch_size, self.show_progress)

    def _in_domain_sources(self) -> list[PoolSource]:
        """The whole corpus of ``data``: train split rows, then validation rows, then any other row."""
        corpus, codes = self.data.dataset.corpus, self.corpus_codes
        splits = self.data.dataset.corpus_splits or {}
        if not splits:
            return [_pool_source("corpus", corpus, codes)]
        names = [name for name in ("train", "val") if name in splits] + sorted(set(splits) - {"train", "val"})
        used = np.zeros(len(corpus), dtype=bool)
        sources = []
        for name in names:
            rows = np.asarray(splits[name], dtype=np.int64)
            sources.append(_pool_source(name, corpus.iloc[rows], codes[rows]))
            used[rows] = True
        rest = np.flatnonzero(~used)
        if len(rest):
            sources.append(_pool_source("other", corpus.iloc[rest], codes[rest]))
        return sources

    def score(
        self,
        cards: Sequence[FeatureCard],
        llm: LLM,
        data: EmbeddedDataset | None = None,
        methods: Sequence[str] = METHODS,
        n_positive: int = 3,
        n_negative: int = 3,
        rank_band: tuple[int, int] = RANK_BAND,
        pool_positives: int = 5,
        pool_negatives: int = 5,
        seed: int = 42,
        max_chars: int | None = 1000,
        batch_size: int = 8,
    ) -> ScoreReport:
        """Activation Prediction scores of the interpretations of ``cards`` (Tables 3, 9 and 10).

        Args:
            cards: Cards from :meth:`describe` (labelled for the ``"llm"`` method).
            llm: Activation Prediction LLM.
            data: Out of domain dataset embedded by the same encoder: the pool is its corpus and the prompt
                says the text comes from another corpus. ``None``, or a dataset with the name of the
                interpreter's own (another object of the same data), scores in domain, on the whole corpus of
                the interpreter's dataset: the setting is decided by name, as the paper decided it by source.
            methods: Interpretations to score among ``"tfidf"``, ``"llm"`` and ``"snippets"``.
            n_positive: Positive texts per feature (the feature ranks within ``rank_band`` on the text).
            n_negative: Negative texts per feature (the feature is inactive on the text).
            rank_band: Ranks of the feature among the text's active features that make a positive.
            pool_positives: Positive rows drawn per feature into the evaluation rows.
            pool_negatives: Rows firing none of the features added to the evaluation rows.
            seed: Seed of both draws (rows, then pairs).
            max_chars: Characters of the judged text, and of each example of the ``"snippets"`` description.
            batch_size: Prompts per LLM call.
        """
        if data is not None and data is not self.data and data.dataset.name == self.data.dataset.name:
            logger.warning(
                "%s is the dataset of the interpreter: it is scored in domain, on the embeddings it holds",
                data.dataset.name,
            )
        out_of_domain = data is not None and data.dataset.name != self.data.dataset.name
        if out_of_domain:
            if data.dim != self.sae.d_in:
                raise ValueError(
                    f"the SAE reads {self.sae.d_in} dimensional embeddings, {data.dataset.name} "
                    f"has {data.dim}"
                )
            if _other_embedder(self.sae, data):
                logger.warning(
                    "the SAE was trained on %s embeddings, %s was embedded by %s",
                    self.sae.embedder,
                    data.dataset.name,
                    data.embedder,
                )
            else:
                warn_on_other_settings(self.sae, data)
            sources = [_pool_source(data.dataset.name, data.dataset.corpus, self._encode(data.corpus))]
            name = data.dataset.name
        else:
            sources = self._in_domain_sources()
            name = self.data.dataset.name
        texts, titles = exemplars(cards)
        pool = EvalPool.build(sources, texts, titles)
        logger.info("%s evaluation pool: %d texts", name, len(pool))
        return run_activation_prediction(
            cards,
            llm,
            pool,
            dataset=name,
            out_of_domain=out_of_domain,
            methods=methods,
            n_positive=n_positive,
            n_negative=n_negative,
            rank_band=rank_band,
            pool_positives=pool_positives,
            pool_negatives=pool_negatives,
            seed=seed,
            max_chars=max_chars,
            batch_size=batch_size,
            show_progress=self.show_progress,
        )

    @cached_property
    def _document_rows(self) -> dict[str, int]:
        rows: dict[str, int] = {}
        for row, text in enumerate(self.data.dataset.corpus["text"].astype(str).tolist()):
            rows.setdefault(text, row)
        return rows

    @cached_property
    def _query_embeddings(self) -> dict[str, np.ndarray]:
        stored: dict[str, np.ndarray] = {}
        for split, embeddings in self.data.queries.items():
            texts = self.data.dataset.queries[split]["text"].astype(str).tolist()
            for text, vector in zip(texts, embeddings, strict=True):
                stored.setdefault(text, vector)
        return stored

    @cached_property
    def _unit_corpus(self) -> sp.csr_matrix:
        codes = self.corpus_codes.astype(np.float64)
        norms = np.sqrt(np.asarray(codes.multiply(codes).sum(axis=1)).ravel())
        inverse = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
        return sp.csr_matrix(sp.diags(inverse) @ codes)

    def _embed(self, text: str, query: bool, embedder: Embedder | None) -> np.ndarray:
        if embedder is None:
            if self._embedder is None:
                from medsae.embed import Embedder

                # The encoder runs where the interpreter was asked to run (an API encoder has no device);
                # it embeds one text at a time, so without a progress bar.
                api = resolve_embedder(self.sae.embedder or "").lower().startswith("gemini")
                options = {} if self.device is None or api else {"device": str(self.device)}
                self._embedder = Embedder.from_sae(self.sae, show_progress=False, **options)
            embedder = self._embedder
        encode = embedder.encode_queries if query else embedder.encode_documents
        return np.asarray(encode([text]), dtype=np.float32)

    def _query_code(self, query: str | np.ndarray, embedder: Embedder | None) -> sp.csr_matrix:
        if isinstance(query, str):
            stored = self._query_embeddings.get(query)
            vector = stored[None, :] if stored is not None else self._embed(query, True, embedder)
        else:
            vector = np.asarray(query, dtype=np.float32).reshape(1, -1)
        return self._encode(vector)

    def _passage_code(
        self, passage: str | np.ndarray, embedder: Embedder | None
    ) -> tuple[sp.csr_matrix, int | None]:
        if isinstance(passage, str):
            row = self._document_rows.get(passage)
            if row is not None:
                return self.corpus_codes[row], row
            return self._encode(self._embed(passage, False, embedder)), None
        return self._encode(np.asarray(passage, dtype=np.float32).reshape(1, -1)), None

    def explain(
        self,
        query: str | np.ndarray,
        doc_a: str | np.ndarray,
        doc_b: str | np.ndarray,
        embedder: Embedder | None = None,
        n_features: int = 4,
        cards: Sequence[FeatureCard] | None = None,
        rank: bool = True,
    ) -> RetrievalExplanation:
        """Which features make one passage outrank the other for a query (Section 6.3, Figure 2).

        Texts of the dataset reuse their stored embeddings; other texts are embedded with ``embedder`` (by
        default the encoder recorded in the SAE). Dense embeddings can be passed instead of texts.

        Args:
            query: Query text or embedding.
            doc_a: First passage, text or embedding.
            doc_b: Second passage, text or embedding.
            embedder: Encoder for texts that are not in the dataset.
            n_features: Features shown (4 in the paper).
            cards: Cards whose Interpreter topics name the features.
            rank: Rank both passages against the whole corpus by sparse cosine and make A the better ranked.
        """
        query_code = self._query_code(query, embedder)
        code_a, row_a = self._passage_code(doc_a, embedder)
        code_b, row_b = self._passage_code(doc_b, embedder)
        rank_a = rank_b = None
        if rank:
            q = unit(query_code)
            scores = self._unit_corpus @ q

            def passage_rank(code: sp.csr_matrix, row: int | None) -> int:
                score = scores[row] if row is not None else float(unit(code) @ q)
                return 1 + int((scores > score).sum())

            rank_a, rank_b = passage_rank(code_a, row_a), passage_rank(code_b, row_b)
        swapped = rank_a is not None and rank_b < rank_a
        if swapped:
            code_a, code_b, doc_a, doc_b, rank_a, rank_b = code_b, code_a, doc_b, doc_a, rank_b, rank_a
        explanation = explain_codes(query_code, code_a, code_b, n_features)
        explanation.query = query if isinstance(query, str) else None
        explanation.passage_a = doc_a if isinstance(doc_a, str) else None
        explanation.passage_b = doc_b if isinstance(doc_b, str) else None
        explanation.rank_a, explanation.rank_b, explanation.swapped = rank_a, rank_b, swapped
        if cards:
            topics = {card.feature: card.topic for card in cards if card.has_label}
            explanation = explanation.with_names(topics)
        return explanation

    def metadata(self) -> dict[str, Any]:
        """Description of the SAE and of the dataset, stored with saved cards."""
        return {
            "embedder": self.sae.embedder,
            "d_in": self.sae.d_in,
            "n_latents": self.sae.n_latents,
            "k": self.sae.k,
            "dataset": self.data.dataset.name,
            "query_split": self.split,
            "corpus_split": self.corpus_split,
        }

    def save_cards(self, cards: Sequence[FeatureCard], path: str | Path) -> None:
        """Write cards to a JSON file together with :meth:`metadata`."""
        save_cards(cards, path, metadata=self.metadata())

    @staticmethod
    def load_cards(path: str | Path) -> list[FeatureCard]:
        """Read cards written by :meth:`save_cards`."""
        return load_cards(path)
