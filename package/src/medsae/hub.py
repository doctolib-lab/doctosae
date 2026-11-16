"""Pretrained MedSAE checkpoints on the Hugging Face Hub, and conversion of paper checkpoints.

The released checkpoints live in the ``doctolib-lab`` organisation (collection
https://huggingface.co/collections/doctolib-lab/medsae): one repository per encoder, and in it one folder per
configuration, named after the number of active features k and the number of features m::

    doctolib-lab/medsae-bge-large-en-v1.5/k64_m3072/{sae.safetensors,cfg.json}
    doctolib-lab/medsae-bge-large-en-v1.5/k256_m3072/...
    doctolib-lab/medsae-gemini-embedding-001/k128_m9216/...

A checkpoint id is the repository followed by the folder, ``doctolib-lab/medsae-bge-large-en-v1.5/k64_m3072``,
and ``load_pretrained("bge", k=64)`` finds it. A repository written by ``MedSAE.push_to_hub`` (``config.json``
and ``model.safetensors`` at its root) holds a single checkpoint, whose id is the repository id.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from medsae._ids import EMBEDDER_ALIASES, require_shareable_embedder, resolve_embedder
from medsae.sae import MedSAE

__all__ = [
    "HUB_ORG",
    "EMBEDDER_ALIASES",
    "resolve_embedder",
    "pretrained_repo_id",
    "pretrained_checkpoint_id",
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

# Files of a released checkpoint folder, and of a repository written by MedSAE.push_to_hub.
_RELEASED_CONFIG = "cfg.json"
_RELEASED_WEIGHTS = "sae.safetensors"
_MIXIN_FILES = {"config.json", "model.safetensors"}
_FOLDER = re.compile(r"k(\d+)_m(\d+)")


# Every paper checkpoint, released ones included, is trained on CURE: its weights inherit the CURE terms.
def _cure_terms() -> dict[str, Any]:
    return {"train_dataset": "cure", "datasets": ["clinia/CUREv1"], "license": "cc-by-nc-4.0"}


def pretrained_repo_id(embedder: str = "bge", org: str = HUB_ORG) -> str:
    """Hub repository of the released checkpoints trained on CURE for ``embedder``."""
    short_name = resolve_embedder(embedder).split("/")[-1].lower()
    return f"{org}/medsae-{short_name}"


def _split_checkpoint_id(checkpoint_id: str) -> tuple[str, str | None]:
    """``org/name/k64_m3072`` to (``org/name``, ``k64_m3072``); a repository id has no folder."""
    parts = checkpoint_id.strip("/").split("/")
    if len(parts) == 3:
        return f"{parts[0]}/{parts[1]}", parts[2]
    return checkpoint_id, None


def _folder_key(folder: str) -> tuple[int, int]:
    k, m = _FOLDER.fullmatch(folder).groups()
    return int(k), int(m)


def _released_folders(files: Iterable[str]) -> list[str]:
    """Folders of a repository that hold a released checkpoint, by increasing k then m."""
    files = set(files)
    folders = {name.split("/")[0] for name in files if name.count("/") == 1}
    released = [
        folder
        for folder in folders
        if _FOLDER.fullmatch(folder)
        and {f"{folder}/{_RELEASED_CONFIG}", f"{folder}/{_RELEASED_WEIGHTS}"} <= files
    ]
    return sorted(released, key=_folder_key)


@contextmanager
def _hub_errors(repo_id: str, revision: str | None) -> Iterator[None]:
    """Turn Hub errors into the built in exceptions a user expects, with what to do next."""
    try:
        yield
    except GatedRepoError as error:  # subclass of RepositoryNotFoundError, so it is caught first
        raise PermissionError(f"{repo_id!r} is gated: accept its terms on the Hub and log in.") from error
    except RepositoryNotFoundError as error:
        raise FileNotFoundError(
            f"No MedSAE checkpoint at {repo_id!r} (or it is private). Published checkpoints: "
            f"{_published_or_unknown()}. You can also train your own with medsae.Trainer."
        ) from error
    except RevisionNotFoundError as error:
        raise FileNotFoundError(f"Revision {revision!r} does not exist in {repo_id!r}.") from error
    except EntryNotFoundError as error:
        raise FileNotFoundError(
            f"{repo_id!r} has no such checkpoint ({error}). Published checkpoints: {_published_or_unknown()}."
        ) from error


def list_pretrained(author: str = HUB_ORG, token: str | None = None) -> list[str]:
    """Ids of the MedSAE checkpoints published by ``author``, ready for :func:`load_pretrained`.

    A released checkpoint reads ``repository/kK_mM``; a repository written by ``MedSAE.push_to_hub`` holds
    one checkpoint and reads as its repository id. Only repositories named with ``medsae`` are listed.
    """
    models = HfApi(token=token).list_models(author=author, search="medsae", expand=["siblings"])
    ids = []
    for model in sorted(models, key=lambda model: model.id):
        files = {sibling.rfilename for sibling in model.siblings or []}
        if files >= _MIXIN_FILES:
            ids.append(model.id)
        ids.extend(f"{model.id}/{folder}" for folder in _released_folders(files))
    return ids


def pretrained_checkpoint_id(
    embedder: str = "bge",
    k: int | None = None,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
) -> str:
    """Id of the checkpoint that :func:`load_pretrained` loads for these arguments.

    Lists the files of the repository, unless ``repo_id`` is already a full checkpoint id.

    ``k`` picks the folder of a repository of released checkpoints; ``None`` picks the smallest k, the
    configuration that the paper interprets. Among folders with the same k, the smallest m wins.
    """
    repo, folder = _split_checkpoint_id(repo_id) if repo_id else (pretrained_repo_id(embedder), None)
    if folder is not None:
        return f"{repo}/{folder}"
    with _hub_errors(repo, revision):
        files = set(HfApi(token=token).list_repo_files(repo, revision=revision))
    if files >= _MIXIN_FILES:
        return repo
    folders = _released_folders(files)
    if not folders:
        raise FileNotFoundError(f"{repo!r} holds no MedSAE checkpoint.")
    if k is None:
        return f"{repo}/{folders[0]}"
    matching = [folder for folder in folders if _folder_key(folder)[0] == int(k)]
    if not matching:
        available = ", ".join(f"{repo}/{folder}" for folder in folders)
        raise FileNotFoundError(f"No checkpoint with k={k} in {repo!r}. Available: {available}.")
    return f"{repo}/{matching[0]}"


def load_pretrained(
    embedder: str = "bge",
    k: int | None = None,
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
            trained on; it picks the repository.
        k: Active features of the released checkpoint: 64 or 256 for BGE and MedEmbed, 128 or 256 for
            Gemini. ``None`` takes the smallest, the configuration that the paper interprets.
        repo_id: Checkpoint id as :func:`list_pretrained` prints it
            (``"doctolib-lab/medsae-bge-large-en-v1.5/k64_m3072"``), or a repository id, instead of
            ``embedder``. A full checkpoint id is downloaded without listing the repository first.
        revision, token, cache_dir: Forwarded to :func:`huggingface_hub.hf_hub_download`.
        **kwargs: Overrides of the stored configuration, for example ``query_instruction=...``. To encode
            with fewer active features than the checkpoint was trained with, set ``sae.k`` after loading or
            pass ``k`` to :meth:`MedSAE.encode`.
    """
    checkpoint_id = pretrained_checkpoint_id(embedder, k, repo_id=repo_id, revision=revision, token=token)
    repo, folder = _split_checkpoint_id(checkpoint_id)
    common = {"revision": revision, "token": token, "cache_dir": cache_dir, "library_name": "medsae"}
    with _hub_errors(repo, revision):
        if folder is None:
            config_path = hf_hub_download(repo, "config.json", **common)
            hf_hub_download(repo, "model.safetensors", **common)
        else:
            config_path = hf_hub_download(repo, f"{folder}/{_RELEASED_CONFIG}", **common)
            weights_path = hf_hub_download(repo, f"{folder}/{_RELEASED_WEIGHTS}", **common)
    if folder is not None:
        sae = _from_released_checkpoint(config_path, weights_path, checkpoint_id, **kwargs)
    else:
        # Loading from the downloaded folder avoids a second (counted) download of config.json.
        sae = MedSAE.from_pretrained(Path(config_path).parent, **kwargs)
    if k is not None and sae.k != int(k):
        raise ValueError(
            f"{checkpoint_id!r} is trained with k={sae.k}, not k={k}: set sae.k to encode sparser."
        )
    return sae


def _published_or_unknown() -> str:
    try:
        return ", ".join(list_pretrained()) or "none yet"
    except Exception:  # only used to enrich an error message
        return "unknown (could not reach the Hub)"


def _require_topk(config: dict[str, Any], source: Any) -> None:
    sparsity = config.get("sparsity_type", "l1")
    if sparsity not in _SUPPORTED_SPARSITY:
        raise ValueError(
            f"{source} holds a {sparsity!r} SAE; MedSAE reproduces the TopK models of the paper "
            f"(sparsity_type in {_SUPPORTED_SPARSITY})."
        )


def _from_released_checkpoint(
    config_path: str | Path, weights_path: str | Path, checkpoint_id: str, **overrides: Any
) -> MedSAE:
    """A :class:`MedSAE` from a released folder: ``cfg.json`` and ``sae.safetensors`` in the paper format.

    The weights carry the parameter names of the paper code, which are those of MedSAE. Queries get the
    instruction the paper embedded them with, the default one of the encoder.
    """
    from safetensors.torch import load_model

    from medsae.embed import default_query_instruction

    config = json.loads(Path(config_path).read_text())
    _require_topk(config, checkpoint_id)
    embedder = resolve_embedder(config["backbone"])
    map_location = overrides.pop("map_location", "cpu")
    strict = overrides.pop("strict", True)
    arguments = {
        "d_in": int(config["input_dim"]),
        "n_latents": int(config["hidden_dim"]),
        "k": int(config["k"]),
        "activation": config.get("activation_fn", "relu"),
        "embedder": embedder,
        "query_instruction": default_query_instruction(embedder),
        "metadata": {"hub_checkpoint": checkpoint_id, "paper_config": config, **_cure_terms()},
    } | overrides
    sae = MedSAE(**arguments)
    load_model(sae, str(weights_path), strict=strict, device=str(map_location))
    logger.info("Loaded %s from %s", sae.extra_repr(), checkpoint_id)
    return sae.eval()


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
    _require_topk(config, folder)
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
        metadata={"paper_config": config, "paper_checkpoint": checkpoint, **_cure_terms()},
    )
    sae.load_state_dict(state, strict=True)
    logger.info("Loaded %s from %s", sae.extra_repr(), path)
    return sae
