"""The DoctoSAE model: a TopK sparse autoencoder over frozen dense text embeddings.

The architecture follows the paper (Section 3.1, Equations 1 to 3) and its released code
(``TopkNoReluSparseAutoencoder`` in ``paper/src/utils/sae_architectures.py``)::

    u = W_enc (x - b_pre) + b_enc
    z = ReLU(TopK_k(u))
    x_hat = W_dec z + b_pre

Parameter names (``encoder``, ``decoder``, ``b_pre``) match the paper code, so checkpoints written by
``paper/src/training/train_sae.py`` load without any key renaming (see
:func:`doctosae.hub.from_paper_checkpoint`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import ModelCard, PyTorchModelHubMixin

from doctosae._ids import is_hub_id, require_shareable_embedder, resolve_embedder

__all__ = ["DoctoSAE", "geometric_median"]

_ACTIVATIONS = ("relu", "none")

# Model card of the checkpoints: the metadata block, then the training data and its terms, which the weights
# inherit, and the responsible use paragraph of the README. A paper link is added once the paper is public.
MODEL_CARD_TEMPLATE = """
---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

# DoctoSAE sparse autoencoder

A TopK sparse autoencoder over the frozen embeddings of {{ embedder }}: {{ n_latents }} features, at most
{{ k }} active per embedding. Load it with the `doctosae` package, `DoctoSAE.from_pretrained(<repository
id>)`, and embed queries with `doctosae.Embedder.from_sae(sae)` so that they get the instruction of the
training queries.

* Code: {{ repo_url }}
* Docs: {{ docs_url }}
{%- if paper_url %}
* Paper: {{ paper_url }}
{%- endif %}

## Training data

{{ training_data }}

## Responsible use

DoctoSAE is a research tool. Feature descriptions are written by an LLM and can be wrong, and retrieval
quality on medical questions is far from perfect. Do not use these weights, their codes or their explanations
for diagnosis, treatment or any other clinical decision.
"""


def _jsonable(value: Any) -> Any:
    """Recursively convert numpy scalars and arrays, tuples and paths into values ``json.dumps`` accepts."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _registry_terms(name: str | None) -> str | None:
    """Usage terms of a registered dataset, for checkpoints that record its name but not its terms."""
    if not name:
        return None
    from doctosae.data.registry import DATASETS, resolve_dataset_name  # pandas only loads to write a card

    key = resolve_dataset_name(name)
    return DATASETS[key].terms if key else None


def _dataset_line(record: dict[str, Any]) -> str:
    """One training dataset of a checkpoint: name, Hub ids, license and terms."""
    name = record.get("train_dataset") or "an unrecorded dataset"
    ids = ", ".join(f"`{dataset}`" for dataset in record.get("datasets") or [])
    license = record.get("license")
    if license == "other":
        license = record.get("license_name") or "other"
    parts = [f"{name} ({ids})" if ids else name]
    parts.append(f"license `{license}`" if license else "no recorded license")
    line = ", ".join(parts) + "."
    terms = record.get("terms") or _registry_terms(record.get("train_dataset"))
    return f"{line} {terms}" if terms else line


def _training_data_text(metadata: dict[str, Any]) -> str:
    """Card paragraph on the training data, whose terms the weights inherit, earliest training first."""
    records = [*metadata.get("parents", []), metadata] if metadata.get("train_dataset") else []
    if not records:
        return "No training data is recorded for these weights: check their origin before using them."
    lines = [f"* {_dataset_line(record)}" for record in records]
    intro = "These weights were trained on the data below, and their use follows its terms:"
    if len(records) > 1:
        intro = "These weights were trained, then fine tuned, on the data below, and follow all its terms:"
    return "\n".join([intro, "", *lines])


def geometric_median(
    x: np.ndarray | torch.Tensor,
    eps: float = 1e-5,
    max_iter: int = 200,
    max_samples: int = 100_000,
    seed: int = 0,
) -> np.ndarray:
    """Geometric median of the rows of ``x`` with Weiszfeld's algorithm.

    Used to initialise ``b_pre`` (Bricken et al., 2023; Gao et al., 2024). Identical to the paper code:
    at most ``max_samples`` rows are drawn without replacement, the iteration starts from the mean and stops
    when the relative update is below ``eps``.
    """
    x = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    x = x.astype(np.float32, copy=False)
    if len(x) > max_samples:
        rng = np.random.RandomState(seed)
        x = x[rng.choice(len(x), max_samples, replace=False)]
    y = x.mean(axis=0)
    for _ in range(max_iter):
        distances = np.maximum(np.linalg.norm(x - y, axis=1), 1e-8)
        weights = 1.0 / distances
        y_new = (weights[:, None] * x).sum(0) / weights.sum()
        if np.linalg.norm(y_new - y) / (np.linalg.norm(y) + 1e-8) < eps:
            return y_new
        y = y_new
    return y


class DoctoSAE(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="doctosae",
    repo_url="https://github.com/doctolib-lab/doctosae",
    docs_url="https://github.com/doctolib-lab/doctosae/tree/main/package",
    license="apache-2.0",
    pipeline_tag="feature-extraction",
    tags=["sparse-autoencoder", "interpretability", "medical", "information-retrieval"],
    model_card_template=MODEL_CARD_TEMPLATE,
):
    """TopK sparse autoencoder trained on the embeddings of a frozen text encoder.

    Args:
        d_in: Dimension ``d`` of the dense embeddings (1024 for ``BAAI/bge-large-en-v1.5``).
        n_latents: Number of SAE features ``m`` (the paper uses ``m = 3d``, ``9d`` or ``21d``).
        k: Number of active features per embedding. The paper reports ``k`` in {64, 128, 256}, and
            "dense" models where ``k = d``.
        activation: ``"relu"`` (paper default, Gao et al. 2024 Equation 2) or ``"none"`` for the linear TopK
            of Makhzani and Frey (2013).
        embedder: Identifier of the frozen embedding model the SAE was trained on. Stored in ``config.json``
            so that :meth:`doctosae.Embedder.from_sae` can rebuild the matching encoder.
        query_instruction: Instruction prepended to queries before embedding, if the embedder needs one.
        metadata: Free form, JSON serializable training metadata (dataset, loss weights, scores...).

    The model is a :class:`huggingface_hub.PyTorchModelHubMixin`, so ``save_pretrained``, ``push_to_hub``
    and ``from_pretrained`` work out of the box. Keyword arguments given to ``from_pretrained`` override the
    stored configuration, for example ``DoctoSAE.from_pretrained(repo_id, k=32)``.
    """

    def __init__(
        self,
        d_in: int,
        n_latents: int,
        k: int = 64,
        activation: str = "relu",
        embedder: str | None = None,
        query_instruction: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {_ACTIVATIONS}, got {activation!r}")
        if not 0 < k <= n_latents:
            raise ValueError(f"k must be in [1, n_latents={n_latents}], got {k}")
        self.k = int(k)
        self.activation = activation
        self.embedder = embedder
        self.query_instruction = query_instruction
        self.metadata = _jsonable(dict(metadata or {}))

        self.encoder = nn.Linear(d_in, n_latents, bias=True)
        self.decoder = nn.Linear(n_latents, d_in, bias=False)
        self.b_pre = nn.Parameter(torch.zeros(d_in))
        # Paper default init: the decoder starts as the encoder transpose with unit norm columns.
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())
        self.normalize_decoder()

    @property
    def d_in(self) -> int:
        return self.encoder.in_features

    @property
    def n_latents(self) -> int:
        return self.encoder.out_features

    @property
    def device(self) -> torch.device:
        return self.b_pre.device

    def extra_repr(self) -> str:
        return f"d_in={self.d_in}, n_latents={self.n_latents}, k={self.k}, activation={self.activation!r}"

    def get_config(self) -> dict[str, Any]:
        """Current constructor arguments, as written to ``config.json``.

        Read from the live attributes rather than the values captured at construction, so that a changed ``k``
        or metadata added during training is what gets saved.
        """
        return {
            "d_in": self.d_in,
            "n_latents": self.n_latents,
            "k": self.k,
            "activation": self.activation,
            "embedder": self.embedder,
            "query_instruction": self.query_instruction,
            "metadata": _jsonable(self.metadata),
        }

    def save_pretrained(self, save_directory, *, config=None, **kwargs):  # noqa: D102 (documented by the mixin)
        return super().save_pretrained(save_directory, config=config or self.get_config(), **kwargs)

    def push_to_hub(self, repo_id: str, *, config=None, **kwargs):  # noqa: D102 (documented by the mixin)
        require_shareable_embedder(self.embedder)
        return super().push_to_hub(repo_id, config=config or self.get_config(), **kwargs)

    @property
    def is_trained(self) -> bool:
        """True once the weights have been trained: ``b_pre`` was initialised or a training is recorded."""
        return bool(self.b_pre.detach().any()) or "train_dataset" in self.metadata

    def generate_model_card(self, *args: Any, **kwargs: Any) -> ModelCard:
        """Model card whose metadata and text follow the training data of this checkpoint.

        The weights inherit the terms of the data they were trained on, so ``metadata["license"]`` and
        ``metadata["datasets"]`` (set by :class:`doctosae.Trainer` from the dataset registry) override the
        package license, and the card text repeats the terms of the training data. For example a checkpoint
        trained on CURE is ``cc-by-nc-4.0``. Trained weights that record no license get ``license: other``
        with ``license_name: unknown``, never the license of the code. ``base_model`` is set, as an adapter of
        the encoder, only when the embedder is a Hub id.
        """
        embedder = resolve_embedder(self.embedder) if self.embedder else None
        text = {
            "embedder": f"`{embedder}`" if embedder and is_hub_id(embedder) else "a text encoder",
            "n_latents": f"{self.n_latents:,}",
            "k": self.k,
            "training_data": _training_data_text(self.metadata),
        }
        card = super().generate_model_card(*args, **(text | kwargs))
        license = self.metadata.get("license")
        if license:
            card.data.license = license
            if license == "other":
                card.data.license_name = self.metadata.get("license_name") or "unknown"
                if self.metadata.get("license_link"):
                    card.data.license_link = self.metadata["license_link"]
        elif self.is_trained:
            card.data.license, card.data.license_name = "other", "unknown"
        if self.metadata.get("datasets"):
            card.data.datasets = list(self.metadata["datasets"])
        if embedder and "/" in embedder and is_hub_id(embedder):
            # The SAE reads the frozen outputs of the encoder: an adapter, not a fine tune of it.
            card.data.base_model = embedder
            card.data.base_model_relation = "adapter"
        card.data.tags = sorted(set(card.data.tags or []) | {f"k{self.k}"})
        return card

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None,
        cache_dir: str | Path | None,
        force_download: bool,
        local_files_only: bool,
        token: str | bool | None,
        map_location: str = "cpu",
        strict: bool = True,
        **model_kwargs: Any,
    ) -> DoctoSAE:
        # The mixin silently swallows Hub errors (missing, gated or private repository, wrong revision) and
        # would then fail with an obscure TypeError. Fail loudly instead, and load weights strictly by
        # default.
        missing = [name for name in ("d_in", "n_latents") if name not in model_kwargs]
        if missing:
            raise FileNotFoundError(
                f"No usable config.json for {model_id!r}: missing repository, gated or private repository "
                f"without a token, wrong revision, or not a DoctoSAE checkpoint (missing {missing})."
            )
        if Path(model_id).is_dir():
            # What the mixin does for a local folder, without the line it prints to stdout.
            from safetensors.torch import load_model

            model = cls(**model_kwargs)
            load_model(model, str(Path(model_id) / "model.safetensors"), strict=strict, device=map_location)
            return model.eval()
        return super()._from_pretrained(
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            map_location=map_location,
            strict=strict,
            **model_kwargs,
        )

    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder pre activations ``W_enc (x - b_pre) + b_enc``, before any sparsity."""
        return self.encoder(x - self.b_pre)

    def _activate(self, values: torch.Tensor) -> torch.Tensor:
        return torch.relu(values) if self.activation == "relu" else values

    def topk(
        self, pre: torch.Tensor, k: int | None = None, allowed: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Keep the ``k`` largest pre activations of each row (optionally among ``allowed`` features only),
        apply the activation and zero everything else."""
        k = self.k if k is None else int(k)
        scores = pre if allowed is None else pre.masked_fill(~allowed, float("-inf"))
        indices = torch.topk(scores, min(k, pre.shape[-1]), dim=-1).indices
        values = self._activate(torch.gather(pre, -1, indices))
        return torch.zeros_like(pre).scatter(-1, indices, values)

    def encode(self, x: torch.Tensor, k: int | None = None) -> torch.Tensor:
        """Sparse code ``z`` of ``x``: at most ``k`` non zero entries per row (``self.k`` by default)."""
        return self.topk(self.pre_acts(x), k)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruction ``W_dec z + b_pre``."""
        return self.decoder(z) + self.b_pre

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x_hat, z)``."""
        z = self.encode(x)
        return self.decode(z), z

    def aux_recon(self, x: torch.Tensor, dead: torch.Tensor, k_aux: int) -> torch.Tensor:
        """Reconstruction from the ``k_aux`` largest dead features only (auxiliary loss of Gao et al. 2024).

        ``b_pre`` is left out on purpose: the target of this term is the residual ``x - x_hat``, which already
        excludes it. ``dead`` is a boolean mask over features and must not be empty.
        """
        n_dead = int(dead.sum())
        if n_dead == 0:
            raise ValueError("aux_recon needs at least one dead feature")
        return self.decoder(self.topk(self.pre_acts(x), min(k_aux, n_dead), allowed=dead))

    @torch.no_grad()
    def init_b_pre(self, x: np.ndarray | torch.Tensor, **kwargs: Any) -> None:
        """Set ``b_pre`` to the geometric median of ``x`` (rows are embeddings)."""
        median = geometric_median(x, **kwargs)
        self.b_pre.copy_(torch.as_tensor(median, dtype=self.b_pre.dtype, device=self.b_pre.device))

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Rescale every decoder column (one per feature) to unit L2 norm."""
        weight = self.decoder.weight
        weight.div_(weight.norm(dim=0, keepdim=True).clamp(min=1e-8))

    @torch.no_grad()
    def project_decoder_gradient(self) -> None:
        """Remove the gradient component parallel to each decoder column, so that a step followed by
        :meth:`normalize_decoder` moves along the unit sphere (Bricken et al., 2023)."""
        weight = self.decoder.weight
        if weight.grad is None:
            return
        squared_norms = (weight * weight).sum(dim=0, keepdim=True).clamp(min=1e-8)
        weight.grad.sub_((weight.grad * weight).sum(dim=0, keepdim=True) / squared_norms * weight)
