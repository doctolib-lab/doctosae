"""Pretrained MedSAE checkpoints on the Hugging Face Hub, and conversion of paper checkpoints.

Official checkpoints live under the ``doctolib-lab`` organisation, one repository per (embedder, k) pair::

    doctolib-lab/medsae-bge-large-en-v1.5-k64
    doctolib-lab/medsae-medembed-large-v0.1-k128

so that choosing a sparsity level is choosing a repository: ``load_pretrained("bge", k=64)``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError

from medsae._ids import EMBEDDER_ALIASES, require_shareable_embedder, resolve_embedder
from medsae.sae import MedSAE

__all__ = [
    "HUB_ORG",
    "EMBEDDER_ALIASES",
    "resolve_embedder",
    "pretrained_repo_id",
    "list_pretrained",
    "load_pretrained",
    "push_to_hub",
    "from_paper_checkpoint",
]

logger = logging.getLogger(__name__)

HUB_ORG = "doctolib-lab"

# Sparse code types of the paper code that MedSAE reproduces exactly. With a ReLU, keeping the k largest
# activations ("topk") and activating the k largest pre activations ("topknorelu") give the same code.
_SUPPORTED_SPARSITY = ("topknorelu", "topk")


def pretrained_repo_id(embedder: str = "bge", k: int = 64, org: str = HUB_ORG) -> str:
    """Hub repository of the official checkpoint trained on CURE for ``embedder`` at sparsity ``k``."""
    short_name = resolve_embedder(embedder).split("/")[-1].lower()
    return f"{org}/medsae-{short_name}-k{int(k)}"


def list_pretrained(author: str = HUB_ORG, token: str | None = None) -> list[str]:
    """Ids of the MedSAE checkpoints published by ``author`` (models tagged with the ``medsae`` library)."""
    models = HfApi(token=token).list_models(author=author, filter="medsae")
    return sorted(model.id for model in models)


def load_pretrained(
    embedder: str = "bge",
    k: int = 64,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
    **kwargs: Any,
) -> MedSAE:
    """Download and load a checkpoint from the Hub.

    Args:
        embedder: Alias (``"bge"``, ``"medembed"``, ``"gemini"``) or full id of the embedder the SAE was
            trained on.
        k: Sparsity level the checkpoint was trained with; it picks the repository.
        repo_id: Explicit repository id, overriding ``embedder`` and ``k``.
        revision, token, cache_dir: Forwarded to :func:`huggingface_hub.hf_hub_download`.
        **kwargs: Overrides of the stored configuration passed to :meth:`MedSAE.from_pretrained`, for example
            ``query_instruction=...``. To encode with fewer active features than the checkpoint was trained
            with, set ``sae.k`` after loading or pass ``k`` to :meth:`MedSAE.encode`.
    """
    repo_id = repo_id or pretrained_repo_id(embedder, k)
    common = {"revision": revision, "token": token, "cache_dir": cache_dir, "library_name": "medsae"}
    try:
        config_path = hf_hub_download(repo_id, "config.json", **common)
        hf_hub_download(repo_id, "model.safetensors", **common)
    except GatedRepoError as error:  # subclass of RepositoryNotFoundError, so it is caught first
        raise PermissionError(f"{repo_id!r} is gated: accept its terms on the Hub and log in.") from error
    except RepositoryNotFoundError as error:
        raise FileNotFoundError(
            f"No MedSAE checkpoint at {repo_id!r} (or it is private). Published checkpoints: "
            f"{_published_or_unknown()}. You can also train your own with medsae.Trainer."
        ) from error
    except RevisionNotFoundError as error:
        raise FileNotFoundError(f"Revision {revision!r} does not exist in {repo_id!r}.") from error
    # Loading from the downloaded folder avoids a second (counted) download of config.json.
    return MedSAE.from_pretrained(Path(config_path).parent, **kwargs)


def _published_or_unknown() -> str:
    try:
        return ", ".join(list_pretrained()) or "none yet"
    except Exception:  # only used to enrich an error message
        return "unknown (could not reach the Hub)"


def push_to_hub(
    sae: MedSAE,
    repo_id: str,
    *,
    private: bool = True,
    license: str | None = None,
    datasets: list[str] | None = None,
    token: str | None = None,
    commit_message: str = "Upload MedSAE checkpoint",
) -> str:
    """Upload ``sae`` to the Hub, as a private repository unless ``private=False``.

    ``license`` and ``datasets`` end up in the model card metadata. When omitted they come from
    ``sae.metadata`` (filled by the trainer from the dataset registry): weights inherit the terms of their
    training data, for example ``cc-by-nc-4.0`` for CURE. Only push SAEs trained on data you may share:
    ``b_pre`` and the decoder carry aggregate information about the training embeddings. A checkpoint whose
    embedder is a local path is refused: set ``sae.embedder`` to the Hub id of the model first.
    """
    require_shareable_embedder(sae.embedder)
    if license:
        sae.metadata["license"] = license
    if datasets:
        sae.metadata["datasets"] = list(datasets)
    return sae.push_to_hub(repo_id, private=private, token=token, commit_message=commit_message)


def from_paper_checkpoint(
    experiment_folder: str | Path,
    checkpoint: str = "best_sae.pth",
    embedder: str | None = None,
    query_instruction: str | None = None,
    map_location: str | torch.device = "cpu",
) -> MedSAE:
    """Build a :class:`MedSAE` from a run folder written by ``paper/src/training/train_sae.py``.

    The folder holds ``sae_config.json`` and the checkpoint (``best_sae.pth`` is the best validation NDCG@10,
    ``best_sae_loss.pth`` the best validation loss). The paper hyperparameters are kept in
    ``metadata["paper_config"]``; every paper checkpoint is trained on CURE, hence its license and dataset.
    """
    folder = Path(experiment_folder)
    config = json.loads((folder / "sae_config.json").read_text())
    sparsity = config.get("sparsity_type", "l1")
    if sparsity not in _SUPPORTED_SPARSITY:
        raise ValueError(
            f"{folder} holds a {sparsity!r} SAE; MedSAE reproduces the TopK models of the paper "
            f"(sparsity_type in {_SUPPORTED_SPARSITY})."
        )
    path = folder / checkpoint
    if not path.exists():
        path = folder / "checkpoints" / checkpoint
    if not path.exists():
        raise FileNotFoundError(f"No {checkpoint} in {folder} or {folder / 'checkpoints'}")
    state = torch.load(path, map_location=map_location, weights_only=True)
    # Checkpoints saved from torch.nn.DataParallel prefix every key with "module.".
    state = {key.removeprefix("module."): value for key, value in state.items()}

    sae = MedSAE(
        d_in=int(config.get("input_dim") or state["b_pre"].shape[0]),
        n_latents=int(config["hidden"]),
        k=int(config["k"]),
        activation=config.get("activation_fn", "relu"),
        embedder=resolve_embedder(embedder) if embedder else None,
        query_instruction=query_instruction,
        metadata={
            "paper_config": config,
            "paper_checkpoint": checkpoint,
            "train_dataset": "cure",
            "datasets": ["clinia/CUREv1"],
            "license": "cc-by-nc-4.0",
        },
    )
    sae.load_state_dict(state, strict=True)
    logger.info("Loaded %s from %s", sae.extra_repr(), path)
    return sae
