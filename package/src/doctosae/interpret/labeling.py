"""Interpreter LLM labels: a topic, a description and keywords for each feature (Section 6.1, Appendix D.2).

The prompts are the paper's (:mod:`doctosae.interpret.prompts`) and replies are parsed exactly as its local
backend did (``label_neurons._parse_json_label``): the model reasons briefly, then closes with a JSON object.
The last balanced ``{...}`` group that validates wins; a reply cut by the token budget is repaired when it
still yields a topic; anything else becomes a ``PARSE_ERROR`` label that keeps the raw reply.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from tqdm.auto import tqdm

from doctosae.interpret.cards import LABEL_ERROR, PARSE_ERROR, FeatureCard
from doctosae.interpret.llm import LLM, generate_isolated
from doctosae.interpret.prompts import interpreter_system_prompt, interpreter_user_prompt

__all__ = ["MAX_FAILED_CHUNKS", "MAX_KEYWORDS", "Label", "label_cards", "label_from_json", "parse_label"]

logger = logging.getLogger(__name__)

# Keyword ceiling applied while parsing, whatever the requested budget (NeuronLabel.MAX_KEYWORDS).
MAX_KEYWORDS = 10

# Consecutive chunks whose every prompt fails, even sent alone, after which the LLM is taken as down.
MAX_FAILED_CHUNKS = 3


@dataclass(frozen=True)
class Label:
    """One Interpreter LLM label."""

    topic: str
    description: str = ""
    keywords: list[str] = field(default_factory=list)


def _coerce_text(value: Any) -> Any:
    """``None`` becomes ``""`` and a list is joined with ``"; "``, as the paper's validators do."""
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return value


def _coerce_keywords(value: Any) -> Any:
    """Normalise the keyword shapes small models produce: no blanks, no repeats (any case), at most 10."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value] if value.strip() else []
    if not isinstance(value, list):
        return value
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        key = text.strip().casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) == MAX_KEYWORDS:
            break
    return out


def label_from_json(obj: Any) -> Label | None:
    """The label held by a decoded JSON value, or ``None`` when it does not validate.

    Same acceptance rules as the paper's pydantic model: ``topic`` is required, ``topic`` and ``description``
    must end up strings and ``keywords`` a list of strings once coerced; other keys are ignored.
    """
    if not isinstance(obj, dict) or "topic" not in obj:
        return None
    topic = _coerce_text(obj["topic"])
    description = _coerce_text(obj["description"]) if "description" in obj else ""
    keywords = _coerce_keywords(obj["keywords"]) if "keywords" in obj else []
    if not isinstance(topic, str) or not isinstance(description, str):
        return None
    if not isinstance(keywords, list) or not all(isinstance(keyword, str) for keyword in keywords):
        return None
    return Label(topic=topic, description=description, keywords=keywords)


def _last_comma_outside_string(text: str) -> int | None:
    """Index of the last ``,`` of ``text`` outside any JSON string, or ``None``."""
    in_string, escape, last = False, False, None
    for position, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == ",":
            last = position
    return last


def _close_truncated_json(text: str) -> dict | None:
    """Recover an object from a reply cut in the middle of its JSON.

    From the first ``{``, up to 6 attempts: close an open string, drop a trailing comma, close open brackets
    and braces, and parse; on failure cut at the last comma outside a string and try again.
    """
    start = text.find("{")
    if start == -1:
        return None
    candidate = text[start:]
    for _ in range(6):
        stack: list[str] = []
        in_string, escape = False, False
        for char in candidate:
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char in "{[":
                stack.append(char)
            elif char in "}]" and stack:
                stack.pop()
        repaired = (candidate + '"' if in_string else candidate).rstrip()
        if repaired.endswith(","):
            repaired = repaired[:-1]
        repaired += "".join("}" if opener == "{" else "]" for opener in reversed(stack))
        try:
            obj = json.loads(repaired)
            return obj if isinstance(obj, dict) else None
        except ValueError:
            pass
        cut = _last_comma_outside_string(candidate)
        if cut is None:
            return None
        candidate = candidate[:cut]
    return None


def _brace_groups(text: str) -> tuple[list[str], int | None]:
    """Every balanced top level ``{...}`` group (braces inside JSON strings skipped), and the start of an
    unclosed group if one remains."""
    groups: list[str] = []
    stack: list[int] = []
    in_string, escape = False, False
    for position, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            stack.append(position)
        elif char == "}" and stack:
            start = stack.pop()
            if not stack:
                groups.append(text[start : position + 1])
    return groups, stack[0] if stack else None


def parse_label(text: str) -> Label:
    """Parse an Interpreter reply; never raises (``PARSE_ERROR`` label with the raw reply as description)."""
    groups, unclosed = _brace_groups(text)
    for group in reversed(groups):
        try:
            label = label_from_json(json.loads(group))
        except (ValueError, RecursionError):
            continue
        if label is not None:
            return label
    recovered = _close_truncated_json(text if unclosed is None else text[unclosed:])
    if recovered and recovered.get("topic"):
        label = label_from_json(recovered)
        if label is not None:
            return label
    return Label(topic=PARSE_ERROR, description=text, keywords=[])


def label_cards(
    cards: Iterable[FeatureCard],
    llm: LLM,
    n_keywords: int = 10,
    max_chars: int | None = 1000,
    batch_size: int = 20,
    show_progress: bool = True,
) -> list[FeatureCard]:
    """Copies of ``cards`` labelled by the Interpreter LLM.

    Cards are sent in consecutive chunks of ``batch_size`` (20 in the paper), each card showing its top
    documents, top queries and negatives with bodies cut to ``max_chars`` characters. When a chunk call
    fails, its cards without a reply are sent again one at a time, and only a card that still fails gets a
    ``LABEL_ERROR`` label, as the paper's labeller isolated each feature. After :data:`MAX_FAILED_CHUNKS`
    consecutive chunks where every card failed, the LLM is taken as unavailable: the remaining cards get
    ``LABEL_ERROR`` without being sent, and an error is logged.
    """
    cards = list(cards)
    system = interpreter_system_prompt(n_keywords)
    labelled: list[FeatureCard] = []
    size = max(1, batch_size)
    failed_chunks = 0
    for start in tqdm(
        range(0, len(cards), size), desc="Interpreter LLM", disable=not show_progress, leave=False
    ):
        chunk = cards[start : start + size]
        if failed_chunks >= MAX_FAILED_CHUNKS:
            unsent = RuntimeError("not sent: the LLM had failed on every prompt of the previous chunks")
            outcomes: list[str | Exception] = [unsent] * len(chunk)
        else:
            users = [interpreter_user_prompt(card, n_keywords, max_chars) for card in chunk]
            outcomes = generate_isolated(llm, system, users)
            for card, outcome in zip(chunk, outcomes, strict=True):
                if isinstance(outcome, Exception):
                    logger.warning("labelling feature %s failed: %s", card.feature, outcome)
            failed_chunks = failed_chunks + 1 if all(isinstance(o, Exception) for o in outcomes) else 0
            if failed_chunks == MAX_FAILED_CHUNKS:
                logger.error(
                    "The Interpreter LLM failed on every prompt of %d chunks in a row: the remaining "
                    "features are not sent and get LABEL_ERROR labels.",
                    MAX_FAILED_CHUNKS,
                )
        labels = [
            Label(LABEL_ERROR, f"{type(outcome).__name__}: {outcome}")
            if isinstance(outcome, Exception)
            else parse_label(outcome)
            for outcome in outcomes
        ]
        for card, label in zip(chunk, labels, strict=True):
            keywords = list(label.keywords)[:n_keywords]
            labelled.append(
                dataclasses.replace(card, topic=label.topic, description=label.description, keywords=keywords)
            )
    failed = Counter(card.topic for card in labelled if not card.has_label)
    if failed:
        logger.warning("%d of %d labels failed: %s", sum(failed.values()), len(labelled), dict(failed))
    return labelled
