"""Sparse codes of the evidence texts, firing statistics and the evidence pulled for each feature.

Follows ``select_neurons.py`` and ``interpretability_utils.py`` of the paper code (Section 6.1, Appendix D.1):

* a feature fires on a text when its code value is strictly positive, everywhere (counts, ranks, negatives);
* texts are deduplicated by exact string, keeping the first occurrence and the original order;
* the top documents of a feature are its strongest activations, with at most two passages per article;
* negatives are documents it does not fire on, drawn by a fresh ``np.random.default_rng(0)`` per feature.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from medsae.sae import MedSAE

__all__ = [
    "MAX_PER_ARTICLE",
    "ActivationIndex",
    "CodedTexts",
    "article_keys",
    "dedup_first",
    "encode_sparse",
    "firing_counts",
    "frame_titles",
    "negative_rows",
    "snippet_titles",
    "top_activating",
    "top_per_article",
]

logger = logging.getLogger(__name__)

# MAX_CHUNKS_PER_DOC of the paper: CURE passages are chunks of articles, and a feature's evidence (and the
# evaluation pool) keeps at most two passages of one article.
MAX_PER_ARTICLE = 2


def encode_sparse(
    sae: MedSAE, embeddings: np.ndarray, batch_size: int = 1024, show_progress: bool = False
) -> sp.csr_matrix:
    """SAE codes of dense embeddings as a CSR matrix ``[n, n_latents]``, encoded in chunks on the SAE device.

    Each row stores its non zero entries (at most ``k``) by increasing feature index, the ``np.flatnonzero``
    storage of the paper's ``convert_embeddings.py`` that its interpretability scripts read: ties between
    equal activations then break the same way.
    """
    import torch

    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != sae.d_in:
        raise ValueError(f"expected embeddings of shape [n, {sae.d_in}], got {x.shape}")
    blocks = []
    with torch.inference_mode():
        starts = range(0, len(x), batch_size)
        for start in tqdm(starts, desc="SAE codes", disable=not show_progress, leave=False):
            batch = torch.from_numpy(x[start : start + batch_size]).to(sae.device)
            blocks.append(sp.csr_matrix(sae.encode(batch).float().cpu().numpy()))
    if not blocks:
        return sp.csr_matrix((0, sae.n_latents), dtype=np.float32)
    codes = sp.vstack(blocks, format="csr")
    codes.sort_indices()
    return codes


def dedup_first(texts: Sequence[str]) -> np.ndarray:
    """Positions of the first occurrence of every distinct string, in original order (``dedup_by_text``)."""
    seen: set[str] = set()
    keep = []
    for position, text in enumerate(texts):
        if text not in seen:
            seen.add(text)
            keep.append(position)
    return np.asarray(keep, dtype=np.int64)


def firing_counts(codes: sp.spmatrix | np.ndarray) -> np.ndarray:
    """Per feature number of rows with a strictly positive code value."""
    return np.asarray((codes > 0).sum(axis=0)).ravel().astype(np.int64)


def _column(codes_csc: sp.spmatrix | np.ndarray, feature: int) -> np.ndarray:
    """One feature's activations as a dense vector (pass a CSC matrix when looping over features)."""
    if sp.issparse(codes_csc):
        return np.asarray(codes_csc[:, feature].todense()).ravel()
    return np.asarray(codes_csc[:, feature])


def top_activating(codes_csc: sp.spmatrix | np.ndarray, feature: int, n: int) -> np.ndarray:
    """Rows of the ``n`` strongest activations of a feature, strongest first, among rows where it fires.

    Same calls as the paper's ``neuron_top_activating``, so ties break the same way.
    """
    col = _column(codes_csc, feature)
    k = min(n, int((col > 0).sum()))
    if k <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-col, k - 1)[:k]
    return idx[np.argsort(-col[idx])]


def top_per_article(
    codes_csc: sp.spmatrix | np.ndarray, feature: int, n: int, articles: Sequence[str] | None = None
) -> np.ndarray:
    """Like :func:`top_activating`, keeping at most :data:`MAX_PER_ARTICLE` rows per article key.

    Rows whose article key is empty belong to no article and are never capped. The paper code put every row
    without a title in one shared bucket (``"nan"`` or ``""``); every CURE passage has a title, so the paper
    numbers are unaffected, and a corpus without titles is not reduced to two passages per feature here.
    """
    if articles is None:
        return top_activating(codes_csc, feature, n)
    if n <= 0:
        return np.array([], dtype=int)
    seen: dict[str, int] = {}
    keep: list[int] = []
    for row in top_activating(codes_csc, feature, codes_csc.shape[0]):
        key = articles[int(row)]
        if key:
            if seen.get(key, 0) >= MAX_PER_ARTICLE:
                continue
            seen[key] = seen.get(key, 0) + 1
        keep.append(int(row))
        if len(keep) == n:
            break
    return np.asarray(keep, dtype=int)


def negative_rows(codes_csc: sp.spmatrix | np.ndarray, feature: int, n: int) -> np.ndarray:
    """``n`` rows the feature does not fire on, in increasing order.

    A fresh ``np.random.default_rng(0)`` per feature, as the paper's ``_negative_snippets``: the draw of one
    feature does not depend on which other features are described.
    """
    rows = np.flatnonzero(_column(codes_csc, feature) <= 0)
    if len(rows) == 0 or n <= 0:
        return np.array([], dtype=int)
    return np.sort(np.random.default_rng(0).choice(rows, min(n, len(rows)), replace=False))


def _clean(value: Any) -> str:
    """Stripped string of a frame cell; missing values (``None``, NaN) become the empty string."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def snippet_titles(frame: pd.DataFrame, texts: Sequence[str]) -> np.ndarray:
    """Per row, the title when the embedded text is exactly ``title + " " + body``, else ``None``.

    Mirrors ``select_neurons._title_of``: prompts then show ``TITLE: ... | TEXT: ...``. Without a ``body``
    column the body is read back from the text.
    """
    titles = np.full(len(frame), None, dtype=object)
    if "title" not in frame.columns:
        return titles
    raw_titles = frame["title"].tolist()
    raw_bodies = frame["body"].tolist() if "body" in frame.columns else None
    for position, (text, raw_title) in enumerate(zip(texts, raw_titles, strict=True)):
        title = _clean(raw_title)
        if not title:
            continue
        if raw_bodies is not None:
            body = _clean(raw_bodies[position])
        else:
            body = text[len(title) + 1 :].strip() if text.startswith(title + " ") else ""
        if body and text == f"{title} {body}":
            titles[position] = title
    return titles


def _cleaned_column(frame: pd.DataFrame, column: str | None) -> np.ndarray:
    if column is None or column not in frame.columns:
        return np.full(len(frame), "", dtype=object)
    return np.asarray([_clean(value) for value in frame[column].tolist()], dtype=object)


def frame_titles(frame: pd.DataFrame) -> np.ndarray:
    """Stripped title of every row, ``""`` when missing (missing titles are not grouped together)."""
    return _cleaned_column(frame, "title")


def article_keys(frame: pd.DataFrame) -> np.ndarray:
    """Article of every row: the ``group`` column if any, else the stripped title; ``""`` when unknown."""
    return _cleaned_column(frame, "group" if "group" in frame.columns else "title")


@dataclass
class CodedTexts:
    """Texts with their sparse codes, deduplicated by exact text.

    Attributes:
        codes: CSR matrix ``[n_texts, n_latents]``.
        texts: Embedded strings.
        ids: Document or query ids.
        titles: Title of each text when it rebuilds the text (``title + " " + body``), else ``None``.
        articles: Article key of each text, ``""`` when unknown. Passages of one article share it.
        rows: Position of each text in the frame it was built from.
    """

    codes: sp.csr_matrix
    texts: np.ndarray
    ids: np.ndarray
    titles: np.ndarray
    articles: np.ndarray
    rows: np.ndarray

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        codes: sp.spmatrix,
        rows: Sequence[int] | np.ndarray | None = None,
        id_column: str = "doc_id",
    ) -> CodedTexts:
        """Build from a corpus or query frame (column ``text``) and the codes of all its rows.

        Args:
            frame: Frame with a ``text`` column (the embedded strings); ``title``, ``body`` and ``group`` are
                used when present.
            codes: Codes of every row of ``frame``, in frame order.
            rows: Frame rows to keep, in this order (the paper's ``corpus_train`` file order); all by default.
            id_column: Column holding the ids; positions are used when it is missing.
        """
        if codes.shape[0] != len(frame):
            raise ValueError(f"{codes.shape[0]} code rows for a frame of {len(frame)} rows")
        rows = np.arange(len(frame), dtype=np.int64) if rows is None else np.asarray(rows, dtype=np.int64)
        sub = frame.iloc[rows]
        texts = np.asarray(sub["text"].astype(str).tolist(), dtype=object)
        keep = dedup_first(texts)
        if len(keep) < len(texts):
            logger.info("%d texts, %d after dropping exact duplicates", len(texts), len(keep))
        ids = sub[id_column].astype(str).tolist() if id_column in sub.columns else [str(r) for r in rows]
        return cls(
            codes=sp.csr_matrix(codes)[rows[keep]],
            texts=texts[keep],
            ids=np.asarray(ids, dtype=object)[keep],
            titles=snippet_titles(sub, texts)[keep],
            articles=article_keys(sub)[keep],
            rows=rows[keep],
        )

    def __len__(self) -> int:
        return len(self.texts)

    @cached_property
    def columns(self) -> sp.csc_matrix:
        """The codes in column major form, for repeated per feature access."""
        return self.codes.tocsc()

    def snippets(
        self, feature: int, rows: Sequence[int], with_activation: bool = True
    ) -> list[dict[str, Any]]:
        """``{"id", "text", "title"}`` (plus ``"activation"``, the raw code value) for the given rows."""
        values = _column(self.columns, feature) if with_activation else None
        out = []
        for row in rows:
            snippet = {"id": str(self.ids[row]), "text": str(self.texts[row]), "title": self.titles[row]}
            if values is not None:
                snippet["activation"] = float(values[row])
            out.append(snippet)
        return out


class ActivationIndex:
    """Evidence documents and queries with their codes: firing statistics and per feature evidence.

    Args:
        documents: The evidence corpus (the paper uses the reconstruction training split of CURE).
        queries: The evidence queries (the CURE training queries in the paper), or ``None``.
    """

    def __init__(self, documents: CodedTexts, queries: CodedTexts | None = None) -> None:
        if queries is not None and queries.codes.shape[1] != documents.codes.shape[1]:
            raise ValueError("documents and queries must be coded by the same SAE")
        self.documents = documents
        self.queries = queries

    @property
    def n_latents(self) -> int:
        return int(self.documents.codes.shape[1])

    @cached_property
    def document_counts(self) -> np.ndarray:
        """Number of evidence documents each feature fires on."""
        return firing_counts(self.documents.codes)

    @cached_property
    def query_counts(self) -> np.ndarray | None:
        """Number of evidence queries each feature fires on, ``None`` without queries."""
        return None if self.queries is None else firing_counts(self.queries.codes)

    @property
    def firing_rates(self) -> np.ndarray:
        """Share of evidence documents each feature fires on."""
        return self.document_counts / max(len(self.documents), 1)

    def top_documents(self, feature: int, n: int = 5) -> np.ndarray:
        """Rows of the feature's strongest documents, at most two per article."""
        return top_per_article(self.documents.columns, feature, n, self.documents.articles)

    def top_queries(self, feature: int, n: int = 5) -> np.ndarray:
        """Rows of the feature's strongest queries (no article cap)."""
        if self.queries is None:
            return np.array([], dtype=int)
        return top_activating(self.queries.columns, feature, n)

    def negatives(self, feature: int, n: int = 2) -> np.ndarray:
        """Rows of ``n`` documents the feature does not fire on."""
        return negative_rows(self.documents.columns, feature, n)
