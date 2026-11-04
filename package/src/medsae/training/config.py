"""Hyperparameters of :class:`medsae.training.Trainer`, with the values of the paper runs as defaults."""

from __future__ import annotations

import operator
from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["TrainingConfig"]

# Fields that decide where and how fast training runs, not what it learns: kept out of the saved metadata.
_RUNTIME_FIELDS = ("eval_batch_size", "device", "show_progress")

# Fields converted to Python numbers, so that values taken from numpy arrays (np.int64, np.float32...) reach
# the JSON files of a run as plain numbers. Integers go through operator.index, which accepts numpy integers
# and refuses 512.0 instead of truncating it.
_INT_FIELDS = ("batch_size", "epochs", "eval_every", "eval_batch_size", "seed")
_FLOAT_FIELDS = (
    "lr",
    "lambda_nce",
    "temperature",
    "alpha_q",
    "weight_decay",
    "aux_coef",
    "corpus_val_fraction",
)
_BOOL_FIELDS = ("compute_val_loss", "show_progress")


@dataclass
class TrainingConfig:
    """Training hyperparameters.

    The defaults reproduce the paper runs (``paper/scripts/sweep_cure_*.sh``): Adam with learning rate
    ``5e-4``, batches of 512, 100 epochs, query reconstruction weight ``0.1`` and a validation NDCG@10
    every 100 steps. The paper sweeps ``lambda_nce`` in {``0``, ``5e-6``, ``1e-5``} and ``temperature`` in
    {``0.01``, ``0.05``, ``0.1``} and selects them per k on the validation NDCG@10; ``1e-5`` and ``0.05``,
    one point of that grid, are the defaults here.

    Numeric fields are converted to Python numbers, so values taken from numpy arrays work as well.

    Attributes:
        lr: Adam learning rate.
        batch_size: Rows per batch of each of the three loaders (training documents, training queries and
            (query, positive document) pairs). A loader with fewer rows gets a smaller batch, with a warning.
        epochs: Passes over the training documents. One step consumes one batch of each loader.
        lambda_nce: Weight of the InfoNCE term; ``0`` trains on reconstruction only.
        temperature: InfoNCE temperature, applied to cosine similarities between sparse codes.
        alpha_q: Weight of query reconstruction relative to document reconstruction.
        weight_decay: Coupled L2 penalty of Adam on the encoder (never on the decoder or ``b_pre``); 0 in
            the paper.
        k_aux: Number of dead features used by the auxiliary loss of Gao et al. (2024); ``None`` disables it,
            as in every run of the paper tables.
        aux_coef: Weight of the auxiliary loss when ``k_aux`` is set.
        eval_every: Steps between two validation NDCG@10 evaluations; the last step is always evaluated.
        corpus_val_fraction: Share of the documents that are not positives of a training query held out of
            training, used only when the dataset carries no corpus split of its own.
        seed: Seed of the loader shuffles, of the pair sampler and of the ``b_pre`` initialisation. The
            SAE weights are drawn when the model is built: call ``torch.manual_seed`` before
            ``MedSAE(...)`` to fix them too.
        compute_val_loss: Also compute the validation loss of the paper at every evaluation. It is monitoring
            only (the checkpoint is selected on NDCG@10) and costs one more pass over the corpus.
        eval_batch_size: Embeddings encoded at once during evaluation.
        device: ``"cuda"``, ``"mps"``, ``"cpu"``, or ``None`` for the first available in that order.
        show_progress: Show a progress bar.
    """

    lr: float = 5e-4
    batch_size: int = 512
    epochs: int = 100
    lambda_nce: float = 1e-5
    temperature: float = 0.05
    alpha_q: float = 0.1
    weight_decay: float = 0.0
    k_aux: int | None = None
    aux_coef: float = 1 / 32
    eval_every: int = 100
    corpus_val_fraction: float = 0.1
    seed: int = 42
    compute_val_loss: bool = False
    eval_batch_size: int = 4096
    device: str | None = None
    show_progress: bool = True

    def __post_init__(self) -> None:
        self._coerce()
        for name in ("batch_size", "epochs", "eval_every", "eval_batch_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")
        for name in ("lr", "temperature"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        for name in ("lambda_nce", "alpha_q", "weight_decay", "aux_coef"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative, got {getattr(self, name)}")
        if self.k_aux is not None and self.k_aux < 1:
            raise ValueError(f"k_aux must be None or at least 1, got {self.k_aux}")
        if not 0 <= self.corpus_val_fraction < 1:
            raise ValueError(f"corpus_val_fraction must be in [0, 1), got {self.corpus_val_fraction}")

    def _coerce(self) -> None:
        """Python ``int``, ``float`` and ``bool`` values, whatever numeric types were given."""
        for name in (*_INT_FIELDS, *(("k_aux",) if self.k_aux is not None else ())):
            value = getattr(self, name)
            try:
                setattr(self, name, operator.index(value))
            except TypeError:
                raise TypeError(f"{name} must be an integer, got {value!r}") from None
        for name in _FLOAT_FIELDS:
            setattr(self, name, float(getattr(self, name)))
        for name in _BOOL_FIELDS:
            setattr(self, name, bool(getattr(self, name)))

    def hyperparameters(self) -> dict[str, Any]:
        """Fields that define what is learned, as JSON native values (saved in the checkpoint metadata)."""
        return {name: value for name, value in asdict(self).items() if name not in _RUNTIME_FIELDS}
