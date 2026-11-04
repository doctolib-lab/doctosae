"""Batched SAE encoding: map dense embeddings to the sparse latent space without gradients."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


@torch.no_grad()
def encode_in_batches(model, X: np.ndarray, device, batch: int = 256) -> np.ndarray:
    """Encode every row of X in mini-batches (no grad); returns float32 (N, hidden)."""
    m = model.module if hasattr(model, "module") else model
    m.eval()
    out = []
    for i in range(0, len(X), batch):
        x = torch.from_numpy(X[i:i+batch].astype(np.float32)).to(device)
        out.append(m.encode(x).cpu().numpy().astype(np.float32))
    return np.vstack(out)


@torch.no_grad()
def encode_in_batches_sparse(model, X: np.ndarray, device, batch: int = 256) -> sp.csr_matrix:
    """Encode every row of X in mini-batches (no grad); returns an (N, hidden) csr_matrix. Top-k
    indices/values are selected on GPU, and rows carrying fewer than k nonzeros drop those slots to
    match dense np.flatnonzero semantics.
    """
    m = model.module if hasattr(model, "module") else model
    m.eval()
    k = m.k
    hidden = m.encoder.out_features if hasattr(m, "encoder") else m.encoder2.out_features
    n = len(X)
    values  = np.empty(n * k, dtype=np.float32)
    indices = np.empty(n * k, dtype=np.int32)
    indptr  = np.zeros(n + 1, dtype=np.int64)
    cursor = 0
    for i in range(0, n, batch):
        x = torch.from_numpy(X[i:i+batch].astype(np.float32)).to(device)
        z = m.encode(x)
        topk_idx = torch.topk(z.abs(), k, dim=1).indices
        topk_val = torch.gather(z, 1, topk_idx)
        idx_np = topk_idx.cpu().numpy().astype(np.int32)
        val_np = topk_val.cpu().numpy().astype(np.float32)
        rows = val_np.shape[0]
        keep = val_np != 0
        if keep.all():  # no ReLU (or no negatives): every row keeps exactly k, write as one block
            sel_v, sel_i = val_np.reshape(-1), idx_np.reshape(-1)
            per_row = np.full(rows, k, dtype=np.int64)
        else:
            flat = keep.reshape(-1)
            sel_v, sel_i = val_np.reshape(-1)[flat], idx_np.reshape(-1)[flat]
            per_row = keep.sum(axis=1).astype(np.int64)
        values[cursor:cursor + sel_v.size]  = sel_v
        indices[cursor:cursor + sel_i.size] = sel_i
        indptr[i + 1:i + 1 + rows] = cursor + np.cumsum(per_row)
        cursor += sel_v.size
    return sp.csr_matrix(
        (values[:cursor], indices[:cursor], indptr),
        shape=(n, hidden), dtype=np.float32,
    )
