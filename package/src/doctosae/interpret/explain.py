"""Retrieval explanations: which features make one passage outrank another for a query (Figure 2).

Codes are L2 normalised, as the retrieval cosine does, so the cosine of a query and a passage is the sum over
features of ``q[j] * p[j]``. For two passages A and B, the features whose contributions differ most,
``abs(q[j] * (a[j] - b[j]))``, explain the gap between the two scores. This rule reproduces the order and the
favoured passage of every panel printed in Figures 2, 11 and 12 of the paper. The share of the cosine it
reports is the quantity the paper prints, but the printed shares cannot be checked without the full codes,
which are not released.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = ["RetrievalExplanation", "explain_codes", "unit"]


def unit(code: np.ndarray | sp.spmatrix) -> np.ndarray:
    """Dense float64 copy of a code scaled to unit L2 norm (a zero code stays zero)."""
    vector = np.asarray(code.todense() if sp.issparse(code) else code, dtype=np.float64).ravel()
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


@dataclass
class RetrievalExplanation:
    """The features that separate the scores of two passages for one query.

    Attributes:
        features: Feature ids, largest contribution difference first.
        query_values: L2 normalised activation of each feature on the query.
        passage_a_values: Same on passage A, the better ranked passage when ranks are known.
        passage_b_values: Same on passage B.
        delta: Contribution difference ``q[j] * (a[j] - b[j])``; positive favours A.
        favours: ``"A"`` or ``"B"`` per feature.
        cosine_a: Cosine of the query and passage A codes (the sum of all contributions).
        cosine_b: Same for passage B.
        share_a: Share of ``cosine_a`` made of the shown features' contributions.
        share_b: Same for passage B.
        names: Interpreter topic of each feature when cards were given.
        query: Query text, when known.
        passage_a: Text of passage A, when known.
        passage_b: Text of passage B, when known.
        rank_a: Rank of passage A for the query over the corpus, when computed.
        rank_b: Same for passage B.
        swapped: True when the passages were swapped so that A is the better ranked one.
    """

    features: list[int]
    query_values: list[float]
    passage_a_values: list[float]
    passage_b_values: list[float]
    delta: list[float]
    favours: list[str]
    cosine_a: float
    cosine_b: float
    share_a: float
    share_b: float
    names: list[str | None] = field(default_factory=list)
    query: str | None = None
    passage_a: str | None = None
    passage_b: str | None = None
    rank_a: int | None = None
    rank_b: int | None = None
    swapped: bool = False

    @property
    def contribution_a(self) -> list[float]:
        """Contribution of each shown feature to ``cosine_a``."""
        return [q * a for q, a in zip(self.query_values, self.passage_a_values, strict=True)]

    @property
    def contribution_b(self) -> list[float]:
        """Contribution of each shown feature to ``cosine_b``."""
        return [q * b for q, b in zip(self.query_values, self.passage_b_values, strict=True)]

    def with_names(self, names: Mapping[int, str]) -> RetrievalExplanation:
        """Copy with feature names looked up in ``names`` (feature id to topic)."""
        values = asdict(self)
        values["names"] = [names.get(feature) for feature in self.features]
        return RetrievalExplanation(**values)

    def to_frame(self) -> pd.DataFrame:
        """One row per shown feature: name, activations, contributions, difference and favoured passage."""
        names = self.names or [None] * len(self.features)
        return pd.DataFrame(
            {
                "feature": self.features,
                "name": names,
                "query": self.query_values,
                "passage_a": self.passage_a_values,
                "passage_b": self.passage_b_values,
                "contribution_a": self.contribution_a,
                "contribution_b": self.contribution_b,
                "delta": self.delta,
                "favours": self.favours,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON compatible dictionary."""
        return asdict(self)

    def to_text(self, max_chars: int | None = 200) -> str:
        """Plain text table in the layout of Figure 2, passages cut to ``max_chars`` characters."""

        def excerpt(text: str) -> str:
            text = " ".join(text.split())
            return text if max_chars is None or len(text) <= max_chars else text[:max_chars].rstrip() + "..."

        def ranked(label: str, rank: int | None) -> str:
            return label if rank is None else f"{label} (rank {rank})"

        lines = []
        if self.query is not None:
            lines.append(f"query: {excerpt(self.query)}")
        if self.passage_a is not None:
            lines.append(f"{ranked('passage A', self.rank_a)}: {excerpt(self.passage_a)}")
        if self.passage_b is not None:
            lines.append(f"{ranked('passage B', self.rank_b)}: {excerpt(self.passage_b)}")
        if lines:
            lines.append("")
        table = [["feature", *map(str, self.features)]]
        if any(self.names):
            table.append(["topic", *[(name or "")[:28] for name in self.names]])
        table += [
            ["query", *[f"{v:.3f}" for v in self.query_values]],
            ["passage A", *[f"{v:.3f}" for v in self.passage_a_values]],
            ["passage B", *[f"{v:.3f}" for v in self.passage_b_values]],
            ["favours", *self.favours],
        ]
        widths = [max(len(row[col]) for row in table) for col in range(len(table[0]))]
        lines += [
            "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
            for row in table
        ]
        lines += [
            "",
            f"These {len(self.features)} features make up {self.share_a:.0%} of the cosine of passage A "
            f"({self.cosine_a:.3f}) and {self.share_b:.0%} of the cosine of passage B ({self.cosine_b:.3f}).",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text()


def explain_codes(
    query_code: np.ndarray | sp.spmatrix,
    a_code: np.ndarray | sp.spmatrix,
    b_code: np.ndarray | sp.spmatrix,
    n_features: int = 4,
) -> RetrievalExplanation:
    """Explain why passage A scores higher or lower than passage B for a query, from their SAE codes.

    Only features active on the query can contribute. Features are ordered by decreasing ``abs(delta)`` (a
    stable sort, so ties keep the feature order), features with ``delta == 0`` are left out, and the first
    ``n_features`` are kept (4 in the paper).
    """
    q, a, b = unit(query_code), unit(a_code), unit(b_code)
    if not q.shape == a.shape == b.shape:
        raise ValueError(f"codes of different sizes: {q.shape}, {a.shape}, {b.shape}")
    contribution_a, contribution_b = q * a, q * b
    delta = contribution_a - contribution_b
    order = [int(j) for j in np.argsort(-np.abs(delta), kind="stable") if delta[j] != 0][:n_features]
    cosine_a, cosine_b = float(contribution_a.sum()), float(contribution_b.sum())
    return RetrievalExplanation(
        features=order,
        query_values=[float(q[j]) for j in order],
        passage_a_values=[float(a[j]) for j in order],
        passage_b_values=[float(b[j]) for j in order],
        delta=[float(delta[j]) for j in order],
        favours=["A" if delta[j] > 0 else "B" for j in order],
        cosine_a=cosine_a,
        cosine_b=cosine_b,
        share_a=float(contribution_a[order].sum() / cosine_a) if cosine_a else 0.0,
        share_b=float(contribution_b[order].sum() / cosine_b) if cosine_b else 0.0,
    )
