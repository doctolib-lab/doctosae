"""Single source of truth for SAE model classes and the ``return_sae`` factory.

Architectures: ``L1SparseAutoencoderUntied`` (primary), ``L1TwoLayerSparseAutoencoder``,
``KLSparseAutoencoder``, ``TopkSparseAutoencoder``. Plus shared bits: ``geometric_median``
(Anthropic b_pre init), ``kl_divergence``, and the ``SoftThreshold`` activation."""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def kl_divergence(rho: float, rho_hat: torch.Tensor) -> torch.Tensor:
    rho_hat = torch.clamp(rho_hat, 1e-10, 1 - 1e-10)
    return torch.sum(
        rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    )


def geometric_median(
    X,
    eps: float = 1e-5,
    max_iter: int = 200,
    max_samples: int = 100_000,
    seed: int = 0,
) -> np.ndarray:
    """Geometric median of X via Weiszfeld's algorithm; used to initialise b_pre."""
    X = np.asarray(X, dtype=np.float32)
    if len(X) > max_samples:
        rng = np.random.RandomState(seed)
        X = X[rng.choice(len(X), max_samples, replace=False)]
    y = X.mean(axis=0)
    for _ in range(max_iter):
        d = np.maximum(np.linalg.norm(X - y, axis=1), 1e-8)
        w = 1.0 / d
        y_new = (w[:, None] * X).sum(0) / w.sum()
        if np.linalg.norm(y_new - y) / (np.linalg.norm(y) + 1e-8) < eps:
            return y_new
        y = y_new
    return y


def gao_init(encoder: nn.Linear, decoder: nn.Linear) -> None:
    """Gao et al. 2024 tied init: decoder := encoder's transpose, decoder columns unit-normed,
    encoder bias zeroed."""
    with torch.no_grad():
        decoder.weight.copy_(encoder.weight.t())
        decoder.weight.div_(decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8))
        encoder.bias.zero_()

def _make_act(name: str, dim: int = None) -> nn.Module:
    name = name.lower()
    if name == "relu":        return nn.ReLU()
    if name == "leaky_relu":  return nn.LeakyReLU()
    if name == "sigmoid":     return nn.Sigmoid()
    if name == "none":        return nn.Identity()
    if name == "shrinkage":   return SoftThreshold()
    raise ValueError(
        f"Unknown activation_fn {name!r}. Choose: relu, leaky_relu, sigmoid, none, shrinkage"
    )


class SoftThreshold(nn.Module):
    """Learnable soft-thresholding activation: sign(x) * max(|x| - λ, 0)."""

    def __init__(self):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * F.relu(torch.abs(x) - self.threshold.abs())


# ---------------------------------------------------------------------------
# Architecture classes
# ---------------------------------------------------------------------------

class L1SparseAutoencoderUntied(nn.Module):
    """1-layer SAE with untied decoder, learned b_pre, L1 sparsity (Anthropic style)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation_fn: str = "relu",
        init: str = "default",
    ):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.b_pre = nn.Parameter(torch.zeros(input_dim), requires_grad=True)
        self._encoder_act = _make_act(activation_fn, hidden_dim)
        if init == "gao":
            gao_init(self.encoder, self.decoder)
        else:
            # FastAutoencoder init: decoder tied to the encoder transpose, then unit-normed.
            with torch.no_grad():
                self.decoder.weight.copy_(self.encoder.weight.t())
                self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8))

    def set_b_pre(self, mean: torch.Tensor):
        with torch.no_grad():
            self.b_pre.copy_(mean)

    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder pre-activations, before any sparsity."""
        return self.encoder(x - self.b_pre)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._encoder_act(self.pre_acts(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) + self.b_pre

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z), z

    def sparsity_loss(self, z: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(z))

    @torch.no_grad()
    def normalize_decoder(self):
        W = self.decoder.weight
        self.decoder.weight.div_(W.norm(dim=0, keepdim=True).clamp(min=1e-8))

    @torch.no_grad()
    def project_decoder_gradient(self):
        """Remove the gradient component parallel to each decoder column (Anthropic eq. 3)."""
        W = self.decoder.weight
        if W.grad is None:
            return
        w_sq = (W * W).sum(dim=0, keepdim=True).clamp(min=1e-8)
        W.grad.sub_((W.grad * W).sum(dim=0, keepdim=True) / w_sq * W)


class L1TwoLayerSparseAutoencoder(nn.Module):
    """2-layer SAE: input → intermediate → hidden (sparse) → intermediate → input."""

    def __init__(
        self,
        input_dim: int,
        intermediate_dim: int,
        hidden_dim: int,
        activation_fn: str = "relu",
    ):
        super().__init__()
        self.encoder1 = nn.Linear(input_dim, intermediate_dim, bias=True)
        self.encoder2 = nn.Linear(intermediate_dim, hidden_dim, bias=True)
        self.decoder1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.decoder2 = nn.Linear(intermediate_dim, input_dim, bias=False)
        self._act = _make_act(activation_fn, intermediate_dim)
        self._encoder_act = _make_act(activation_fn, hidden_dim)
        self.b_pre = nn.Parameter(torch.zeros(input_dim), requires_grad=True)

    def set_b_pre(self, mean: torch.Tensor):
        with torch.no_grad():
            self.b_pre.copy_(mean)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self._act(self.encoder1(x - self.b_pre))
        return self._encoder_act(self.encoder2(h))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self._act(self.decoder1(z))
        return self.decoder2(h) + self.b_pre

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z), z

    def sparsity_loss(self, z: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(z))

    @torch.no_grad()
    def normalize_decoder(self):
        W = self.decoder2.weight
        self.decoder2.weight.div_(W.norm(dim=0, keepdim=True).clamp(min=1e-8))

    @torch.no_grad()
    def project_decoder_gradient(self):
        W = self.decoder2.weight
        if W.grad is None:
            return
        w_sq = (W * W).sum(dim=0, keepdim=True).clamp(min=1e-8)
        W.grad.sub_((W.grad * W).sum(dim=0, keepdim=True) / w_sq * W)


class KLSparseAutoencoder(nn.Module):
    """SAE with KL-divergence sparsity toward a target activation rate rho."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z), z

    def sparsity_loss(self, z: torch.Tensor, rho: float = 0.05) -> torch.Tensor:
        return kl_divergence(rho, z.mean(dim=0))


class TopkSparseAutoencoder(L1SparseAutoencoderUntied):
    """SAE with hard top-k activation per forward pass (no L1 penalty needed)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        k: int = 128,
        activation_fn: str = "relu",
        init: str = "default",
    ):
        super().__init__(input_dim, hidden_dim, activation_fn=activation_fn, init=init)
        self.k = k

    def _topk(self, pre: torch.Tensor, k: int, allowed: torch.Tensor | None = None) -> torch.Tensor:
        """This class's TopK encoding of `pre`, restricted to the latents in `allowed`."""
        z = self._encoder_act(pre)
        scores = z if allowed is None else z.masked_fill(~allowed, float("-inf"))
        mask = torch.zeros_like(z)
        mask.scatter_(1, torch.topk(scores, k, dim=1)[1], 1.0)
        return z * mask

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._topk(self.pre_acts(x), self.k)

    def aux_recon(self, x: torch.Tensor, dead: torch.Tensor, k_aux: int) -> torch.Tensor:
        """Reconstruction from the top-k_aux dead latents only (Gao et al. 2024's auxiliary loss),
        for a non-empty `dead`. b_pre is omitted: decode() adds it, so it is already in the residual."""
        return self.decoder(self._topk(self.pre_acts(x), min(k_aux, int(dead.sum())), allowed=dead))

    def sparsity_loss(self, z: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=z.device)


class TopkNoReluSparseAutoencoder(TopkSparseAutoencoder):
    """TopK selects support from raw pre-activations, then encoder_act is applied (Gao et al. 2024,
    eq. 2: z = ReLU(TopK(pre))); --activation-fn none gives the linear TopK of Makhzani & Frey
    2013."""

    def _topk(self, pre: torch.Tensor, k: int, allowed: torch.Tensor | None = None) -> torch.Tensor:
        scores = pre if allowed is None else pre.masked_fill(~allowed, float("-inf"))
        mask = torch.zeros_like(pre)
        mask.scatter_(1, torch.topk(scores, k, dim=1)[1], 1.0)
        return self._encoder_act(pre * mask)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def return_sae(
    input_dim: int,
    sparsity_type: str,
    hidden_dim: int,
    k: int = 128,
    intermediate_dim: Optional[int] = None,
    activation_fn: str = "relu",
    init: str = "default",
) -> nn.Module:
    """Instantiate the right SAE class for the given sparsity_type. ``init='gao'`` uses
    ``gao_init`` (single encoder/decoder pair only, so not l1_2layer/kl)."""
    if sparsity_type in ("l1_2layer", "kl") and init != "default":
        raise ValueError(f"init={init!r} unsupported for sparsity_type={sparsity_type!r}")
    if sparsity_type == "l1_2layer":
        if intermediate_dim is None:
            raise ValueError("intermediate_dim required for sparsity_type='l1_2layer'")
        return L1TwoLayerSparseAutoencoder(
            input_dim, intermediate_dim, hidden_dim,
            activation_fn=activation_fn,
        )
    if sparsity_type == "topk":
        return TopkSparseAutoencoder(
            input_dim, hidden_dim, k=k,
            activation_fn=activation_fn, init=init,
        )
    if sparsity_type == "topknorelu":
        return TopkNoReluSparseAutoencoder(
            input_dim, hidden_dim, k=k,
            activation_fn=activation_fn, init=init,
        )
    if sparsity_type == "kl":
        return KLSparseAutoencoder(input_dim, hidden_dim)
    if sparsity_type in ("l1", None):
        if sparsity_type is None:
            warnings.warn("sparsity_type not set, defaulting to 'l1'")
        return L1SparseAutoencoderUntied(
            input_dim, hidden_dim,
            activation_fn=activation_fn, init=init,
        )
    raise ValueError(f"Unknown sparsity_type {sparsity_type!r}")


def load_sae_checkpoint(experiment_folder: str, checkpoint: str = "best_sae.pth",
                        device: torch.device | None = None, input_dim: int | None = None) -> nn.Module:
    """Load a trained SAE from an experiment folder's sae_config.json + checkpoint .pth."""
    import json
    import os

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = os.path.join(experiment_folder, "sae_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"sae_config.json not found in {experiment_folder}")
    with open(config_path) as f:
        cfg = json.load(f)

    model = return_sae(
        input_dim=input_dim or cfg["input_dim"],
        sparsity_type=cfg.get("sparsity_type", "l1"),
        hidden_dim=cfg["hidden"],
        k=cfg.get("k", 128),
        intermediate_dim=cfg.get("intermediate"),
        activation_fn=cfg.get("activation_fn", "relu"),
    ).to(device)

    checkpoint_path = os.path.join(experiment_folder, checkpoint)
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(experiment_folder, "checkpoints", checkpoint)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    else:
        warnings.warn(f"checkpoint {checkpoint!r} not found in {experiment_folder}, using randomly initialised model")
    return model


def _hidden_dim(model: nn.Module) -> int:
    """Return the SAE bottleneck dimension regardless of architecture."""
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "encoder2"):
        return m.encoder2.out_features
    return m.encoder.out_features
