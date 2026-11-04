"""Activation Prediction: how well an interpretation predicts where its feature fires (Section 6.2, D.3).

An LLM reads the interpretation of one feature and one text, and writes the probability that the feature is
active on the text. A text is a positive when the feature is among its 20 strongest active features, a
negative when the feature is inactive on it. An interpretation method scores the Pearson correlation between
probability and ground truth, per feature, averaged over features. The steps follow
``build_interpret_eval.py``, ``predict_neurons_binary.py`` and the table scripts of the paper code:

1. Evaluation pool: a corpus deduplicated by text, without the texts shown to the Interpreter and without
   other passages of their articles, at most two passages per article.
2. Evaluation rows: 5 positives per feature and 5 rows firing none of the features,
   ``np.random.default_rng(42)``.
3. Pairs: 3 positives and 3 negatives per feature, one ``random.Random(42)`` shared across features.
4. Prompts in chunks of 8; the probability is the ``"score"`` of the last flat JSON object of the reply,
   clamped to [0, 1]. A reply without one is a parse error, kept with score 0.5 and left out of every metric.
   A prompt whose request fails, even sent again alone, is an error record, kept and left out the same way:
   one failed request never costs the other predictions.
5. Metrics per feature (Pearson r, AUROC), averaged over the features that every method scored. The paper
   reports Pearson r (Table 3); AUROC is an extra metric.
"""

from __future__ import annotations

import json
import logging
import random
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from medsae.interpret.activations import MAX_PER_ARTICLE, dedup_first
from medsae.interpret.cards import FeatureCard
from medsae.interpret.llm import LLM, generate_isolated
from medsae.interpret.prompts import (
    METHODS,
    cap_text,
    feature_description,
    judge_system_prompt,
    judge_user_prompt,
)

__all__ = [
    "MAX_FAILED_CHUNKS",
    "RANK_BAND",
    "RANK_BANDS",
    "EvalPool",
    "PoolSource",
    "ScoreReport",
    "build_eval_rows",
    "entry_ranks",
    "exemplars",
    "feature_aurocs",
    "feature_metrics",
    "parse_score",
    "pearson_r",
    "rank_band_stats",
    "run_activation_prediction",
    "sample_pairs",
    "sample_pool_rows",
]

logger = logging.getLogger(__name__)

RANK_BAND = (1, 20)
RANK_BANDS = ((1, 5), (6, 10), (11, 15), (16, 20))
# Consecutive chunks whose every prompt fails, even sent alone, after which the LLM is taken as down.
MAX_FAILED_CHUNKS = 3
_PREVIEW_CHARS = 300
_FLAT_OBJECT = re.compile(r"\{[^{}]*\}", re.DOTALL)


class LLMUnavailableError(RuntimeError):
    """The Activation Prediction LLM failed on every prompt of several chunks in a row."""


def _usable(prediction: Mapping[str, Any]) -> bool:
    """A prediction that enters the metrics: parsed, and not an error record."""
    return not prediction["parse_error"] and not prediction.get("error")


@dataclass
class PoolSource:
    """One block of candidate evaluation texts with their codes.

    Attributes:
        name: Source name recorded on every row (corpus split or dataset name).
        codes: Codes ``[n, n_latents]``.
        texts: Embedded strings.
        titles: Stripped titles, ``""`` when missing: rows sharing a shown passage's title are excluded.
        articles: Article keys, ``""`` when unknown: at most two rows per article are kept.
    """

    name: str
    codes: sp.spmatrix
    texts: Sequence[str]
    titles: Sequence[str] | None = None
    articles: Sequence[str] | None = None


def exemplars(cards: Iterable[FeatureCard]) -> tuple[set[str], set[str]]:
    """Texts and titles the Interpreter has seen: every shown document, query and negative of every card."""
    snippets = [s for card in cards for s in (*card.top_documents, *card.top_queries, *card.negatives)]
    return {s["text"] for s in snippets}, {s["title"] for s in snippets if s.get("title")}


@dataclass
class EvalPool:
    """Texts eligible for Activation Prediction, with their codes (``build_interpret_eval.load_pool``)."""

    codes: sp.csr_matrix
    texts: np.ndarray
    sources: np.ndarray

    @classmethod
    def build(
        cls,
        sources: Sequence[PoolSource],
        exemplar_texts: set[str] = frozenset(),
        exemplar_titles: set[str] = frozenset(),
        cap_articles: bool = True,
    ) -> EvalPool:
        """Pool the sources, deduplicate by text, drop what the Interpreter saw, cap rows per article.

        A text shown to the Interpreter (any card's document, query or negative) is excluded, and so is every
        row whose title is the title of a shown passage. The article cap keeps the first two rows of each
        article in pooled order, counted before the exclusions, as in the paper.
        """
        if not sources:
            raise ValueError("the evaluation pool needs at least one source")

        def column(source: PoolSource, values: Sequence[str] | None) -> list[str]:
            return [""] * len(source.texts) if values is None else [str(v) for v in values]

        codes = sp.vstack([sp.csr_matrix(source.codes) for source in sources], format="csr")
        texts = np.asarray([str(t) for source in sources for t in source.texts], dtype=object)
        titles = np.asarray([t for source in sources for t in column(source, source.titles)], dtype=object)
        articles = np.asarray(
            [a for source in sources for a in column(source, source.articles)], dtype=object
        )
        names = np.asarray([source.name for source in sources for _ in source.texts], dtype=object)
        if codes.shape[0] != len(texts):
            raise ValueError(f"{codes.shape[0]} code rows for {len(texts)} texts")
        keep = dedup_first(texts)
        codes, texts, titles = codes[keep], texts[keep], titles[keep]
        articles, names = articles[keep], names[keep]
        leaked = np.fromiter(
            (
                text in exemplar_texts or (title != "" and title in exemplar_titles)
                for text, title in zip(texts, titles, strict=True)
            ),
            dtype=bool,
            count=len(texts),
        )
        first = np.ones(len(texts), dtype=bool)
        if cap_articles:
            seen: dict[str, int] = {}
            for position, article in enumerate(articles):
                if article:
                    first[position] = seen.get(article, 0) < MAX_PER_ARTICLE
                    seen[article] = seen.get(article, 0) + 1
        selected = np.flatnonzero(~leaked & first)
        pool_codes = codes[selected]
        pool_codes.sort_indices()
        return cls(codes=pool_codes, texts=texts[selected], sources=names[selected])

    def __len__(self) -> int:
        return len(self.texts)


def entry_ranks(codes: sp.csr_matrix) -> np.ndarray:
    """Rank of every stored entry within its row, 1 for the largest value (``row_ranks``).

    Aligned with ``codes.data``. Ties follow numpy's default sort reversed, as in the paper code.
    """
    ranks = np.zeros(len(codes.data), dtype=np.int64)
    for row in range(codes.shape[0]):
        start, end = codes.indptr[row], codes.indptr[row + 1]
        order = np.argsort(codes.data[start:end])[::-1]
        ranks[start + order] = np.arange(1, end - start + 1)
    return ranks


def _entry_rows(codes: sp.csr_matrix) -> np.ndarray:
    return np.repeat(np.arange(codes.shape[0]), np.diff(codes.indptr))


def _feature_positions(features: Sequence[int], n_latents: int) -> np.ndarray:
    position = np.full(n_latents, -1, dtype=np.int64)
    position[np.asarray(features, dtype=np.int64)] = np.arange(len(features))
    return position


def positive_matrix(
    codes: sp.csr_matrix,
    features: Sequence[int],
    rank_band: tuple[int, int] = RANK_BAND,
    ranks: np.ndarray | None = None,
) -> sp.csc_matrix:
    """Boolean ``[n_rows, len(features)]``: the feature ranks ``low..high`` among the row's features.

    ``ranks`` are the :func:`entry_ranks` of ``codes`` when already computed.
    """
    low, high = rank_band
    ranks = entry_ranks(codes) if ranks is None else ranks
    columns = _feature_positions(features, codes.shape[1])[codes.indices]
    ok = (columns >= 0) & (ranks >= low) & (ranks <= high)
    data = np.ones(int(ok.sum()), dtype=bool)
    shape = (codes.shape[0], len(features))
    positives = sp.csc_matrix((data, (_entry_rows(codes)[ok], columns[ok])), shape=shape)
    positives.sort_indices()  # rows in increasing order within each feature, like np.where on a dense mask
    return positives


def sample_pool_rows(
    codes: sp.csr_matrix,
    features: Sequence[int],
    positives: sp.csc_matrix,
    per_feature: int = 5,
    n_negatives: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Evaluation rows: ``per_feature`` positives per feature, then ``n_negatives`` rows firing none of the
    features, one generator for the whole loop (``build_interpret_eval.sample_indices``)."""
    rng = np.random.default_rng(seed)
    chosen = []
    for j in range(len(features)):
        strong = positives.indices[positives.indptr[j] : positives.indptr[j + 1]].astype(np.int64)
        chosen.append(rng.choice(strong, min(per_feature, len(strong)), replace=False))
    if n_negatives:
        hits = (_feature_positions(features, codes.shape[1])[codes.indices] >= 0) & (codes.data > 0)
        fires_any = np.zeros(codes.shape[0], dtype=bool)
        fires_any[_entry_rows(codes)[hits]] = True
        none_fire = np.flatnonzero(~fires_any)
        chosen.append(rng.choice(none_fire, min(n_negatives, len(none_fire)), replace=False))
    if not chosen:
        return np.array([], dtype=np.int64)
    return np.unique(np.concatenate(chosen))


def build_eval_rows(
    pool: EvalPool,
    features: Sequence[int],
    rank_band: tuple[int, int] = RANK_BAND,
    per_feature: int = 5,
    n_negatives: int = 5,
    seed: int = 42,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Sampled evaluation rows and the features kept (those with at least one positive row).

    Each row is ``{"row", "source", "text", "activations", "positive_for", "ranks"}`` with ``activations``
    (code values above zero) and ``ranks`` keyed by the feature id as a string, like the paper's files.
    """
    codes = pool.codes
    features = [int(f) for f in features]
    if len(set(features)) != len(features):
        raise ValueError("features must be unique")
    if any(not 0 <= f < codes.shape[1] for f in features):
        raise ValueError(f"features must be in [0, {codes.shape[1]})")
    ranks = entry_ranks(codes)
    positives = positive_matrix(codes, features, rank_band, ranks)
    sampled = sample_pool_rows(codes, features, positives, per_feature, n_negatives, seed)
    sampled_positives = positives.tocsr()[sampled]
    kept_positions = np.flatnonzero(np.asarray(sampled_positives.sum(axis=0)).ravel() > 0)
    kept = [features[j] for j in kept_positions]
    kept_set = set(kept)
    is_positive = sampled_positives[:, kept_positions].toarray()
    rows = []
    for i, row in enumerate(sampled):
        start, end = codes.indptr[row], codes.indptr[row + 1]
        entries = [
            (int(f), float(v), int(r))
            for f, v, r in zip(codes.indices[start:end], codes.data[start:end], ranks[start:end], strict=True)
            if int(f) in kept_set
        ]
        rows.append(
            {
                "row": int(row),
                "source": str(pool.sources[row]),
                "text": str(pool.texts[row]),
                "activations": {str(f): v for f, v, _ in entries if v > 0},
                "positive_for": [kept[j] for j in np.flatnonzero(is_positive[i])],
                "ranks": {str(f): r for f, _, r in entries},
            }
        )
    return kept, rows


def sample_pairs(
    rows: Sequence[Mapping[str, Any]],
    features: Sequence[int],
    n_positive: int = 3,
    n_negative: int = 3,
    seed: int = 42,
) -> list[tuple[int, int, bool]]:
    """``(feature, row index, is_positive)`` pairs, feature by feature, positives then negatives.

    One ``random.Random(seed)`` is shared across features in order (``predict_neurons_binary.sample_pairs``),
    so skipping a feature changes the draws of every later one.
    """
    rng = random.Random(seed)
    positive_rows: dict[int, list[int]] = {f: [] for f in features}
    active_rows: dict[int, list[int]] = {f: [] for f in features}
    for i, row in enumerate(rows):
        for f in row["positive_for"]:
            if f in positive_rows:
                positive_rows[f].append(i)
        for key in row["activations"]:
            if int(key) in active_rows:
                active_rows[int(key)].append(i)
    every_row = np.arange(len(rows))
    pairs: list[tuple[int, int, bool]] = []
    for f in features:
        positive = positive_rows[f]
        negative = np.setdiff1d(every_row, active_rows[f]).tolist()
        pairs += [(f, i, True) for i in rng.sample(positive, min(n_positive, len(positive)))]
        pairs += [(f, i, False) for i in rng.sample(negative, min(n_negative, len(negative)))]
    return pairs


def parse_score(text: str) -> tuple[float | None, float | None]:
    """``(probability clamped to [0, 1], value as written)`` from the last flat ``{...}`` of a reply, or
    ``(None, None)`` when it does not hold a numeric ``"score"`` (``predict_neurons_binary._parse_json``)."""
    matches = _FLAT_OBJECT.findall(text)
    if not matches:
        return None, None
    try:
        raw = float(json.loads(matches[-1])["score"])
    except (ValueError, KeyError, TypeError, OverflowError, RecursionError):
        return None, None
    return max(0.0, min(1.0, raw)), raw


def pearson_r(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Point biserial correlation of scores with ±1 labels; 0 for constant scores, NaN for a single class."""
    s = np.asarray(scores, dtype=float)
    y = np.array([1.0 if label else -1.0 for label in labels])
    if not s.std():
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.corrcoef(s, y)[0, 1])


def feature_metrics(
    predictions: Sequence[Mapping[str, Any]], features: Sequence[int]
) -> dict[int, dict[str, Any]]:
    """Per feature ``{"support", "n_pairs", "pearson"}`` over predictions without parse errors or request
    errors (``score_pairs``).

    ``pearson`` is NaN for a feature left without valid predictions.
    """
    by_feature: dict[int, list[Mapping[str, Any]]] = {f: [] for f in features}
    for p in predictions:
        if _usable(p) and p["feature"] in by_feature:
            by_feature[p["feature"]].append(p)
    out = {}
    for f, sub in by_feature.items():
        scores, labels = [p["score"] for p in sub], [p["ground_truth"] for p in sub]
        out[f] = {
            "support": sum(labels),
            "n_pairs": len(sub),
            "pearson": pearson_r(scores, labels) if sub else float("nan"),
        }
    return out


def feature_aurocs(predictions: Sequence[Mapping[str, Any]], features: Iterable[int]) -> dict[int, float]:
    """Per feature AUROC of the scores against the ground truth, for features with both classes."""
    keep = set(features)
    by_feature: dict[int, tuple[list[bool], list[float]]] = {}
    for p in predictions:
        if _usable(p) and p["feature"] in keep:
            labels, scores = by_feature.setdefault(p["feature"], ([], []))
            labels.append(bool(p["ground_truth"]))
            scores.append(p["score"])
    return {
        f: float(roc_auc_score(labels, scores))
        for f, (labels, scores) in by_feature.items()
        if 0 < sum(labels) < len(labels)
    }


def rank_band_stats(
    predictions: Sequence[Mapping[str, Any]], bands: Sequence[tuple[int, int]] = RANK_BANDS
) -> dict[str, float]:
    """Mean probability on positives per rank band, mean on negatives (``"inactive"``) and the Spearman
    correlation between rank and probability on positives (``table_probability_by_rank.band_stats``)."""
    valid = [p for p in predictions if _usable(p)]
    ranks = np.array([p["rank"] for p in valid if p["ground_truth"]], dtype=float)
    scores = np.array([p["score"] for p in valid if p["ground_truth"]], dtype=float)
    out = {}
    for low, high in bands:
        inside = (ranks >= low) & (ranks <= high)
        out[f"{low}-{high}"] = float(scores[inside].mean()) if inside.any() else float("nan")
    negatives = [p["score"] for p in valid if not p["ground_truth"]]
    out["inactive"] = float(np.mean(negatives)) if negatives else float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["spearman"] = float(spearmanr(ranks, scores).statistic) if len(ranks) > 1 else float("nan")
    return out


def _nanmean(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or np.isnan(values).all():
        return float("nan")
    return float(np.nanmean(values))


@dataclass
class ScoreReport:
    """Activation Prediction results of one evaluation pool.

    Attributes:
        dataset: Name of the dataset the pool comes from.
        out_of_domain: Whether the pool is out of domain (the judge prompt then says so).
        methods: Interpretation methods scored, among ``"tfidf"``, ``"llm"`` and ``"snippets"``.
        features: Features with at least one positive evaluation row.
        scored: Features each method scored (a feature without a usable label is skipped by ``"llm"``).
        predictions: One record per scored pair and method: ``feature``, ``row``, ``source``, ``text`` (first
            300 characters), ``ground_truth``, ``activation``, ``rank``, ``parse_error``, ``score``,
            ``fires``, plus ``raw_reply`` after a parse error, ``score_raw`` when the score was clamped and
            ``error`` when the request failed (such records are left out of the metrics, like parse errors).
        rows: The evaluation rows (full texts).
        llm: Description of the model that predicted.
        failed: Methods stopped because the LLM failed on every prompt of several chunks in a row, with the
            error. They are left out of ``methods``; the other methods are complete.
    """

    dataset: str
    out_of_domain: bool
    methods: list[str]
    features: list[int]
    scored: dict[str, list[int]]
    predictions: dict[str, list[dict[str, Any]]]
    rows: list[dict[str, Any]] = field(repr=False)
    llm: str = ""
    failed: dict[str, str] = field(default_factory=dict)

    @cached_property
    def shared_features(self) -> list[int]:
        """Features every method scored: the set every average is taken over, as in the paper tables."""
        if not self.methods:
            return []
        shared = set.intersection(*(set(self.scored[m]) for m in self.methods))
        return [f for f in self.features if f in shared]

    @cached_property
    def per_feature(self) -> dict[str, dict[int, dict[str, Any]]]:
        """Per method and feature: ``support``, ``n_pairs``, ``pearson``, ``auroc`` (NaN with one class)."""
        out = {}
        for method in self.methods:
            metrics = feature_metrics(self.predictions[method], self.scored[method])
            aurocs = feature_aurocs(self.predictions[method], self.scored[method])
            for f, values in metrics.items():
                values["auroc"] = aurocs.get(f, float("nan"))
            out[method] = metrics
        return out

    @cached_property
    def summary(self) -> dict[str, dict[str, Any]]:
        """Per method: mean Pearson r and mean AUROC over :attr:`shared_features`, counts, parse errors,
        request errors and the share of exact 0 scores (``abstained``)."""
        shared = self.shared_features
        shared_set = set(shared)
        out = {}
        for method in self.methods:
            metrics = self.per_feature[method]
            aurocs = [metrics[f]["auroc"] for f in shared if not np.isnan(metrics[f]["auroc"])]
            predictions = self.predictions[method]
            on_shared = [p for p in predictions if p["feature"] in shared_set and not p.get("error")]
            out[method] = {
                "pearson_r": _nanmean([metrics[f]["pearson"] for f in shared]),
                "auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
                "n_features": len(shared),
                "n_calls": len(predictions),
                "n_parse_errors": sum(1 for p in predictions if p["parse_error"]),
                "n_errors": sum(1 for p in predictions if p.get("error")),
                "abstained": (sum(p["score"] == 0.0 for p in on_shared) / len(on_shared))
                if on_shared
                else float("nan"),
            }
        return out

    def rank_bands(self, bands: Sequence[tuple[int, int]] = RANK_BANDS) -> dict[str, dict[str, float]]:
        """Per method: mean probability per rank band of the feature on positive texts, on negatives and the
        Spearman correlation between rank and probability (Tables 11 to 13)."""
        return {method: rank_band_stats(self.predictions[method], bands) for method in self.methods}

    def to_frame(self) -> pd.DataFrame:
        """One row per method: ``pearson_r``, ``auroc``, ``n_features``, ``n_calls``, ``n_parse_errors``,
        ``n_errors``, ``abstained``."""
        frame = pd.DataFrame.from_dict(self.summary, orient="index")
        frame.index.name = "method"
        return frame

    def feature_frame(self) -> pd.DataFrame:
        """One row per (method, feature) with the per feature metrics."""
        records = [
            {"method": method, "feature": f, **values}
            for method, metrics in self.per_feature.items()
            for f, values in metrics.items()
        ]
        columns = ["method", "feature", "support", "n_pairs", "pearson", "auroc"]
        return pd.DataFrame.from_records(records, columns=columns)

    def rank_frame(self, bands: Sequence[tuple[int, int]] = RANK_BANDS) -> pd.DataFrame:
        """:meth:`rank_bands` as a frame, one row per method."""
        frame = pd.DataFrame.from_dict(self.rank_bands(bands), orient="index")
        frame.index.name = "method"
        return frame

    def to_dict(self) -> dict[str, Any]:
        """JSON compatible summary, per feature metrics and predictions (rows are left out)."""
        return {
            "dataset": self.dataset,
            "out_of_domain": self.out_of_domain,
            "llm": self.llm,
            "methods": list(self.methods),
            "failed": dict(self.failed),
            "features": list(self.features),
            "shared_features": list(self.shared_features),
            "n_rows": len(self.rows),
            "summary": self.summary,
            "rank_bands": self.rank_bands(),
            "per_feature": {
                method: {str(f): values for f, values in metrics.items()}
                for method, metrics in self.per_feature.items()
            },
            "predictions": self.predictions,
        }


def _predict(
    llm: LLM,
    method: str,
    pairs: Sequence[tuple[int, int, bool]],
    rows: Sequence[Mapping[str, Any]],
    descriptions: Mapping[int, str],
    out_of_domain: bool,
    max_chars: int | None,
    batch_size: int,
    show_progress: bool,
) -> list[dict[str, Any]]:
    """One record per pair. Failed requests become error records; :class:`LLMUnavailableError` is raised
    once every prompt of :data:`MAX_FAILED_CHUNKS` chunks in a row has failed."""
    system = judge_system_prompt(method)
    texts = [cap_text(row["text"], max_chars) for row in rows]
    predictions = []
    size = max(1, batch_size)
    failed_chunks = 0
    progress = tqdm(
        range(0, len(pairs), size),
        desc=f"Activation Prediction ({method})",
        disable=not show_progress,
        leave=False,
    )
    for start in progress:
        chunk = pairs[start : start + size]
        users = [judge_user_prompt(f, descriptions[f], texts[i], out_of_domain) for f, i, _ in chunk]
        replies = generate_isolated(llm, system, users)
        failed_chunks = failed_chunks + 1 if all(isinstance(reply, Exception) for reply in replies) else 0
        if failed_chunks == MAX_FAILED_CHUNKS:
            raise LLMUnavailableError(
                f"the LLM failed on every prompt of {MAX_FAILED_CHUNKS} chunks in a row, last: "
                f"{type(replies[-1]).__name__}: {replies[-1]}"
            )
        for (f, i, truth), reply in zip(chunk, replies, strict=True):
            row = rows[i]
            record = {
                "feature": f,
                "row": row["row"],
                "source": row["source"],
                "text": texts[i][:_PREVIEW_CHARS],
                "ground_truth": truth,
                "activation": row["activations"].get(str(f)),
                "rank": row["ranks"].get(str(f)),
            }
            if isinstance(reply, Exception):
                record |= {"parse_error": False, "error": f"{type(reply).__name__}: {reply}"}
                score = None
            else:
                score, raw = parse_score(reply)
                record["parse_error"] = score is None
                if score is None:
                    record["raw_reply"] = reply
                elif raw != score:
                    record["score_raw"] = raw
            record["score"] = 0.5 if score is None else score
            record["fires"] = score is not None and score > 0.5
            predictions.append(record)
    errors = sum(1 for record in predictions if record.get("error"))
    if errors:
        logger.warning(
            "%d of %d %s predictions failed and are left out of the metrics", errors, len(predictions), method
        )
    return predictions


def run_activation_prediction(
    cards: Sequence[FeatureCard],
    llm: LLM,
    pool: EvalPool,
    *,
    dataset: str,
    out_of_domain: bool = False,
    methods: Sequence[str] = METHODS,
    n_positive: int = 3,
    n_negative: int = 3,
    rank_band: tuple[int, int] = RANK_BAND,
    pool_positives: int = 5,
    pool_negatives: int = 5,
    seed: int = 42,
    max_chars: int | None = 1000,
    batch_size: int = 8,
    show_progress: bool = True,
) -> ScoreReport:
    """Score the interpretations of ``cards`` on ``pool`` with the Activation Prediction LLM.

    Every default is the paper's: rank band 1 to 20, 5 positives per feature and 5 rows firing none in the
    pool, 3 positive and 3 negative pairs per feature, seed 42, texts cut to 1000 characters, chunks of 8
    prompts. A method stopped because the LLM is unavailable is reported in :attr:`ScoreReport.failed` and
    the methods scored before it are kept.
    """
    methods = list(methods)
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown methods {unknown}; choose among {METHODS}")
    if "llm" in methods and not any(card.has_label for card in cards):
        raise ValueError("method 'llm' needs labelled cards: describe the features with an llm first")
    if "tfidf" in methods and not any(card.tfidf_terms for card in cards):
        raise ValueError(
            "method 'tfidf' needs TF IDF terms, and no card has any (TF IDF is disabled on a corpus too "
            "small for its min_df): leave 'tfidf' out of methods"
        )
    by_feature = {card.feature: card for card in cards}
    features, rows = build_eval_rows(pool, list(by_feature), rank_band, pool_positives, pool_negatives, seed)
    scored: dict[str, list[int]] = {}
    predictions: dict[str, list[dict[str, Any]]] = {}
    failed: dict[str, str] = {}
    for method in methods:
        descriptions = {}
        for f in features:
            try:
                descriptions[f] = feature_description(by_feature[f], method, max_chars)
            except ValueError:
                continue
        pairs = sample_pairs(rows, list(descriptions), n_positive, n_negative, seed)
        try:
            predictions[method] = _predict(
                llm, method, pairs, rows, descriptions, out_of_domain, max_chars, batch_size, show_progress
            )
        except LLMUnavailableError as error:
            logger.error("Activation Prediction of %s stopped: %s", method, error)
            failed[method] = str(error)
            continue
        scored[method] = list(descriptions)
    return ScoreReport(
        dataset=dataset,
        out_of_domain=out_of_domain,
        methods=[method for method in methods if method not in failed],
        features=features,
        scored=scored,
        predictions=predictions,
        rows=rows,
        llm=repr(llm),
        failed=failed,
    )
