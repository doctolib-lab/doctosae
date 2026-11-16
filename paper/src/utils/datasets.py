"""Torch Datasets for SAE training: ``EmbDataset`` (per-row embedding + id, for the corpus and
query loaders) and ``PairDataset`` (query→positive-doc pairs, for the InfoNCE contrastive loader)."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class EmbDataset(Dataset):
    """Per-row embedding + integer row id; used by the corpus and queries loaders."""
    def __init__(self, X: np.ndarray, ids: np.ndarray):
        self.X = X.astype(np.float32)
        self.ids = np.asarray(ids, dtype=np.int64)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), int(self.ids[i])


class PairDataset(Dataset):
    """Yields (xq, xd, q_idx, c_idx, score) for each (query, doc+) pair; score is the qrels
    relevance value for that pair, used to weight the positive term in InfoNCE."""
    def __init__(self, pairs: np.ndarray, scores: np.ndarray, Xq: np.ndarray, Xc: np.ndarray) -> None:
        self.pairs = np.asarray(pairs, dtype=np.int64)
        self.scores = np.asarray(scores, dtype=np.float32)
        self.Xq = Xq
        self.Xc = Xc

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        q, d = int(self.pairs[idx, 0]), int(self.pairs[idx, 1])
        return (
            torch.tensor(self.Xq[q], dtype=torch.float32),
            torch.tensor(self.Xc[d], dtype=torch.float32),
            torch.tensor(q, dtype=torch.long),
            torch.tensor(d, dtype=torch.long),
            torch.tensor(self.scores[idx], dtype=torch.float32),
        )
