"""Shared SAE training helpers: reconstruction loss, query/corpus loss weighting, and
weight-decay-aware optimizer construction."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reconstruction loss
# ---------------------------------------------------------------------------

def recon_loss(x: torch.Tensor, x_hat: torch.Tensor, mode: str = "mse") -> torch.Tensor:
    """Reconstruction loss: MSE by default, or cosine distance when mode='cosine'."""
    if mode == "cosine":
        x     = F.normalize(x,     dim=-1)
        x_hat = F.normalize(x_hat, dim=-1)
    return F.mse_loss(x_hat, x, reduction="mean")


def squared_error(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """||x - x_hat||^2 per row, averaged over the batch -- summed over dimensions, not averaged."""
    return (x - x_hat).pow(2).sum(dim=-1).mean()


def weighted_loss(value_c, value_q, N_c: int, N_q: int, alpha_q: float = 1.0):
    """(value_c · N_c + α_q · value_q · N_q) / (N_c + α_q · N_q); α_q=0 ⇒ corpus only, α_q=1 ⇒ raw sample-count weighting."""
    return (value_c * N_c + alpha_q * value_q * N_q) / (N_c + alpha_q * N_q)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, lr: float, wd: float) -> torch.optim.Adam:
    """Adam optimiser with weight decay excluded from the decoder weights and b_pre."""
    m = model.module if hasattr(model, "module") else model
    no_decay = set()
    for attr in ("decoder", "decoder2"):
        if hasattr(m, attr):
            no_decay.add(id(getattr(m, attr).weight))
    if hasattr(m, "b_pre"):
        no_decay.add(id(m.b_pre))
    param_groups = [
        {"params": [p for p in model.parameters() if id(p) not in no_decay], "weight_decay": wd},
        {"params": [p for p in model.parameters() if id(p) in no_decay],     "weight_decay": 0.0},
    ]
    return torch.optim.Adam(param_groups, lr=lr)
