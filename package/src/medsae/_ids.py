"""Model identifiers and embedding settings: what embeddings and checkpoints record, and checks on it.

Embeddings and checkpoints record the encoder that produced the vectors and the settings that change them
(revision, instructions, normalization, truncation), so that queries are always embedded the way the
training queries were. Whatever leaves the machine (a model card, a pushed ``config.json``) names a Hub id,
never a local path, which could reveal a user name or a folder layout.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from huggingface_hub.utils import HFValidationError, validate_repo_id

__all__ = [
    "EMBEDDER_ALIASES",
    "EMBEDDING_KEYS",
    "embedding_differences",
    "embedding_settings",
    "is_hub_id",
    "is_local_path",
    "require_shareable_embedder",
    "resolve_embedder",
    "shareable_model_id",
]

# Short names for the three embedders of the paper.
EMBEDDER_ALIASES: dict[str, str] = {
    "bge": "BAAI/bge-large-en-v1.5",
    "medembed": "abhinand/MedEmbed-large-v0.1",
    "gemini": "gemini-embedding-001",
}

# Models whose id is written in its canonical case whatever the case it is given in: the Hub resolves ids
# without regard to case, so "baai/bge-large-en-v1.5" loads the same weights and must get the same defaults.
_KNOWN_MODELS = (*EMBEDDER_ALIASES.values(), "BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5")
_CANONICAL = {model.lower(): model for model in _KNOWN_MODELS}

#: Settings of an embedder that change its vectors, in the order they are reported.
EMBEDDING_KEYS = (
    "model",
    "revision",
    "query_instruction",
    "document_instruction",
    "normalize",
    "max_seq_length",
)

_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


def resolve_embedder(embedder: str) -> str:
    """Full model id of an alias such as ``"bge"``, canonical case of a known model id, else ``embedder``."""
    key = embedder.lower()
    return EMBEDDER_ALIASES.get(key) or _CANONICAL.get(key, embedder)


def is_local_path(value: str) -> bool:
    """True when ``value`` names a file or folder of this machine rather than a Hub id."""
    if value.startswith(("/", ".", "~")) or "\\" in value or _WINDOWS_DRIVE.match(value):
        return True
    try:
        return Path(value).expanduser().exists()
    except OSError:
        return False


def is_hub_id(value: str) -> bool:
    """True for a valid Hub repository id (``org/name`` or ``name``) that is not also a local path."""
    if not value or is_local_path(value):
        return False
    try:
        validate_repo_id(value)
    except HFValidationError:
        return False
    return True


def shareable_model_id(value: str | None) -> str | None:
    """``value`` when it can be shared (a Hub id or an API model name), else the last part of the local path.

    A local path would publish a folder layout, and often a user name, in metadata and model cards.
    """
    if value is None or not is_local_path(value):
        return value
    return re.split(r"[\\/]", value.rstrip("\\/"))[-1] or "local_model"


def require_shareable_embedder(embedder: str | None) -> None:
    """Refuse to upload a checkpoint whose embedder is a local path: ``sae.embedder`` must be a Hub id."""
    if embedder is not None and is_local_path(embedder):
        raise ValueError(
            f"the embedder of this checkpoint is a local path ({shareable_model_id(embedder)!r}): set "
            "sae.embedder to the Hub id of that model before uploading, so that the repository names no "
            "local folder and other users can rebuild the encoder"
        )


def embedding_settings(
    embedder: str | None, query_instruction: str | None, extra: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Known settings of embeddings: ``extra`` (a recorded embedder configuration), the model and the query
    instruction. ``None`` for the model or the instruction means unknown, so the key is left out."""
    settings = dict(extra or {})
    if embedder:
        settings["model"] = embedder
    if query_instruction is not None:
        settings["query_instruction"] = query_instruction
    return settings


def embedding_differences(expected: Mapping[str, Any], found: Mapping[str, Any]) -> list[str]:
    """Settings recorded on both sides that differ, as readable strings (empty when they agree).

    A setting missing on either side is unknown and never differs. Model ids are compared after alias and
    case resolution, and a revision is compared only when both sides pin one.
    """
    differences = []
    for key in EMBEDDING_KEYS:
        if key not in expected or key not in found:
            continue
        a, b = expected[key], found[key]
        if key == "model":
            a, b = resolve_embedder(str(a)), resolve_embedder(str(b))
        elif key == "revision" and (a is None or b is None):
            continue
        if a != b:
            differences.append(f"{key} {a!r} against {b!r}")
    return differences
