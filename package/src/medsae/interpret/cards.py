"""Feature cards: what is known about one SAE feature, as a JSON serialisable record.

A card holds the evidence the Interpreter LLM reads (top documents, top queries and documents the feature does
not fire on), the TF IDF terms of that evidence and, once labelled, the Interpreter topic, description and
keywords.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from medsae.interpret.prompts import split_title

__all__ = ["FeatureCard", "LABEL_ERROR", "PARSE_ERROR", "load_cards", "save_cards"]

# Topics recorded when labelling fails, as in the paper's labels files: the LLM call raised (LABEL_ERROR) or
# its reply held no usable JSON label (PARSE_ERROR, the raw reply is kept as the description).
LABEL_ERROR = "LABEL_ERROR"
PARSE_ERROR = "PARSE_ERROR"
_FAILED_TOPICS = (LABEL_ERROR, PARSE_ERROR)

_CARDS_FORMAT = "medsae.feature_cards"
_CARDS_VERSION = 1


@dataclass
class FeatureCard:
    """Evidence and interpretation of one SAE feature.

    Attributes:
        feature: Feature index in ``[0, n_latents)``.
        n_documents: Number of evidence documents the feature fires on (code value strictly positive).
        n_queries: Number of evidence queries it fires on.
        firing_rate: ``n_documents`` divided by the number of evidence documents.
        top_documents: Strongest documents, at most two per article, as
            ``{"id", "text", "title", "activation"}``. ``text`` is the exact embedded string; ``title`` is set
            when the text is the title, a space and the body, so that prompts can show
            ``TITLE: ... | TEXT: ...``.
        top_queries: Strongest queries, as ``{"id", "text", "title", "activation"}`` (``title`` is ``None``).
        negatives: Documents the feature does not fire on, as ``{"id", "text", "title"}``.
        tfidf_terms: TF IDF terms of the top documents and queries, most weighted first.
        topic: Interpreter LLM topic, or :data:`LABEL_ERROR` / :data:`PARSE_ERROR` when labelling failed.
        description: Interpreter LLM description (the raw reply after a parse error).
        keywords: Interpreter LLM keywords, most useful first.
    """

    feature: int
    n_documents: int
    n_queries: int
    firing_rate: float
    top_documents: list[dict[str, Any]]
    top_queries: list[dict[str, Any]]
    negatives: list[dict[str, Any]]
    tfidf_terms: list[str]
    topic: str | None = None
    description: str | None = None
    keywords: list[str] = field(default_factory=list)

    @property
    def has_label(self) -> bool:
        """True when the Interpreter LLM produced a usable label."""
        return self.topic is not None and self.topic not in _FAILED_TOPICS

    @property
    def interpretation(self) -> str | None:
        """``"topic: description (keywords: k1, k2)"``: the label as the Activation Prediction LLM and the
        clinician annotation study read it, or ``None`` without a usable label."""
        if not self.has_label:
            return None
        return f"{self.topic}: {self.description or ''} (keywords: {', '.join(self.keywords)})"

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FeatureCard:
        """Inverse of :meth:`to_dict`; unknown keys are ignored."""
        names = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in names})

    def to_markdown(self, max_chars: int | None = 300) -> str:
        """Readable summary: label, firing counts, TF IDF terms and evidence, texts cut to ``max_chars``."""
        heading = f"### Feature {self.feature}"
        if self.has_label:
            heading += f": {self.topic}"
        lines = [heading, ""]
        if self.has_label and self.description:
            lines += [_one_line(self.description), ""]
        elif self.topic in _FAILED_TOPICS:
            lines += [f"Labelling failed ({self.topic}).", ""]
        lines.append(
            f"* Fires on {self.n_documents:,} documents ({self.firing_rate:.2%}) "
            f"and {self.n_queries:,} queries"
        )
        if self.has_label and self.keywords:
            lines.append(f"* Keywords: {', '.join(self.keywords)}")
        if self.tfidf_terms:
            lines.append(f"* TF IDF terms: {', '.join(self.tfidf_terms)}")
        sections = (
            ("Top documents", self.top_documents),
            ("Top queries", self.top_queries),
            ("Documents it does not fire on", self.negatives),
        )
        for name, snippets in sections:
            if snippets:
                lines += ["", f"**{name}**", ""]
                lines += [f"{rank}. {_snippet_markdown(s, max_chars)}" for rank, s in enumerate(snippets, 1)]
        return "\n".join(lines)


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _cut(text: str, max_chars: int | None) -> str:
    text = _one_line(text)
    if max_chars is None or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _snippet_markdown(snippet: Mapping[str, Any], max_chars: int | None) -> str:
    title, body = split_title(snippet)
    if body is not None:
        shown = f"**{_one_line(title)}**: {_cut(body, max_chars)}"
    else:
        shown = _cut(snippet["text"], max_chars)
    activation = snippet.get("activation")
    return f"`[{activation:.2f}]` {shown}" if activation is not None else shown


def save_cards(
    cards: Iterable[FeatureCard], path: str | Path, metadata: Mapping[str, Any] | None = None
) -> None:
    """Write cards to a JSON file, with optional metadata (SAE and dataset names, never paths)."""
    payload = {
        "format": _CARDS_FORMAT,
        "version": _CARDS_VERSION,
        "metadata": dict(metadata or {}),
        "cards": [card.to_dict() for card in cards],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_cards(path: str | Path) -> list[FeatureCard]:
    """Read cards written by :func:`save_cards`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != _CARDS_FORMAT:
        raise ValueError(f"{path} is not a medsae feature cards file")
    return [FeatureCard.from_dict(card) for card in payload["cards"]]
