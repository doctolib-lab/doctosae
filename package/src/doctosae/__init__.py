"""DoctoSAE: sparse autoencoders for interpretable and transferable medical information retrieval.

Reference implementation of "Explanations that Transfer: Interpreting Medical Retrieval with Sparse
AutoEncoders" (ML4H 2026). A TopK sparse autoencoder is trained on the frozen embeddings of a text
encoder, with a reconstruction term and an InfoNCE term on the sparse codes; retrieval then runs on the
sparse codes and every feature can be described in natural language.

The public names below are imported on first access, so ``import doctosae`` (and the ``doctosae`` command
line) stays fast: PyTorch, pandas and sentence transformers load only when a model or a dataset is
used.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

# Public name to the module that defines it.
_EXPORTS: dict[str, str] = {
    "DoctoSAE": "doctosae.sae",
    "load_pretrained": "doctosae.hub",
    "list_pretrained": "doctosae.hub",
    "pretrained_repo_id": "doctosae.hub",
    "pretrained_checkpoint_id": "doctosae.hub",
    "from_paper_checkpoint": "doctosae.hub",
    "RetrievalDataset": "doctosae.data",
    "load_dataset": "doctosae.data",
    "list_datasets": "doctosae.data",
    "load_beir_folder": "doctosae.data",
    "Embedder": "doctosae.embed",
    "EmbeddedDataset": "doctosae.embed",
    "Trainer": "doctosae.training",
    "TrainingConfig": "doctosae.training",
    "evaluate": "doctosae.metrics",
    "encode_corpus": "doctosae.retrieval",
    "search": "doctosae.retrieval",
    "Interpreter": "doctosae.interpret",
    "FeatureCard": "doctosae.interpret",
    "OpenAICompatibleLLM": "doctosae.interpret",
    "TransformersLLM": "doctosae.interpret",
}

__all__ = [
    "__version__",
    "DoctoSAE",
    "load_pretrained",
    "list_pretrained",
    "pretrained_repo_id",
    "pretrained_checkpoint_id",
    "from_paper_checkpoint",
    "RetrievalDataset",
    "load_dataset",
    "list_datasets",
    "load_beir_folder",
    "Embedder",
    "EmbeddedDataset",
    "Trainer",
    "TrainingConfig",
    "evaluate",
    "encode_corpus",
    "search",
    "Interpreter",
    "FeatureCard",
    "OpenAICompatibleLLM",
    "TransformersLLM",
]

if TYPE_CHECKING:  # static analysers and IDEs see the real objects
    from doctosae.data import RetrievalDataset, list_datasets, load_beir_folder, load_dataset
    from doctosae.embed import EmbeddedDataset, Embedder
    from doctosae.hub import (
        from_paper_checkpoint,
        list_pretrained,
        load_pretrained,
        pretrained_checkpoint_id,
        pretrained_repo_id,
    )
    from doctosae.interpret import FeatureCard, Interpreter, OpenAICompatibleLLM, TransformersLLM
    from doctosae.metrics import evaluate
    from doctosae.retrieval import encode_corpus, search
    from doctosae.sae import DoctoSAE
    from doctosae.training import Trainer, TrainingConfig


def __getattr__(name: str) -> Any:
    """Import a public name from its module on first access (PEP 562), then cache it on the package."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'doctosae' has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
