"""MedSAE: sparse autoencoders for interpretable and transferable medical information retrieval.

Reference implementation of "Explanations that Transfer: Interpreting Medical Retrieval with Sparse
AutoEncoders" (ML4H 2026). A TopK sparse autoencoder is trained on the frozen embeddings of a text
encoder, with a reconstruction term and an InfoNCE term on the sparse codes; retrieval then runs on the
sparse codes and every feature can be described in natural language.

The public names below are imported on first access, so ``import medsae`` (and the ``medsae`` command
line) stays fast: PyTorch, pandas and sentence transformers load only when a model or a dataset is
used.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

# Public name to the module that defines it.
_EXPORTS: dict[str, str] = {
    "MedSAE": "medsae.sae",
    "load_pretrained": "medsae.hub",
    "list_pretrained": "medsae.hub",
    "pretrained_repo_id": "medsae.hub",
    "from_paper_checkpoint": "medsae.hub",
    "RetrievalDataset": "medsae.data",
    "load_dataset": "medsae.data",
    "list_datasets": "medsae.data",
    "load_beir_folder": "medsae.data",
    "Embedder": "medsae.embed",
    "EmbeddedDataset": "medsae.embed",
    "Trainer": "medsae.training",
    "TrainingConfig": "medsae.training",
    "evaluate": "medsae.metrics",
    "encode_corpus": "medsae.retrieval",
    "search": "medsae.retrieval",
    "Interpreter": "medsae.interpret",
    "FeatureCard": "medsae.interpret",
    "OpenAICompatibleLLM": "medsae.interpret",
    "TransformersLLM": "medsae.interpret",
}

__all__ = [
    "__version__",
    "MedSAE",
    "load_pretrained",
    "list_pretrained",
    "pretrained_repo_id",
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
    from medsae.data import RetrievalDataset, list_datasets, load_beir_folder, load_dataset
    from medsae.embed import EmbeddedDataset, Embedder
    from medsae.hub import from_paper_checkpoint, list_pretrained, load_pretrained, pretrained_repo_id
    from medsae.interpret import FeatureCard, Interpreter, OpenAICompatibleLLM, TransformersLLM
    from medsae.metrics import evaluate
    from medsae.retrieval import encode_corpus, search
    from medsae.sae import MedSAE
    from medsae.training import Trainer, TrainingConfig


def __getattr__(name: str) -> Any:
    """Import a public name from its module on first access (PEP 562), then cache it on the package."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'medsae' has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
