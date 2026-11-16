"""InfoNCE contrastive helpers for SAE training: load query→positive pairs from qrels, the
``info_nce`` loss, a sampler that keeps each query unique within a batch, and false-negative
masking (``build_pos_dict`` / ``build_pos_mask``) so a query's other positives aren't used as negatives."""
from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def load_pairs(queries_df: pd.DataFrame, corpus_df: pd.DataFrame, qrels_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ((query_idx, doc_idx) positive pairs, raw qrels score per pair) from a BeIR qrels TSV."""
    qr = pd.read_csv(qrels_path, sep="\t")
    if qr.shape[1] < 3:
        qr = pd.read_csv(qrels_path, sep=r"\s+", engine="python")
    qr["query-id"] = qr["query-id"].astype(str)
    qr["corpus-id"] = qr["corpus-id"].astype(str)
    score_col = "score" if "score" in qr.columns else "relevance"
    qr = qr[qr[score_col] > 0]
    q2i = {qid: i for i, qid in enumerate(queries_df["id"].astype(str).values)}
    d2i = {did: i for i, did in enumerate(corpus_df.index.values)}
    pairs, scores = [], []
    for q, d, s in zip(qr["query-id"], qr["corpus-id"], qr[score_col]):
        if q in q2i and d in d2i:
            pairs.append((q2i[q], d2i[d]))
            scores.append(s)
    return np.array(pairs, dtype=np.int64), np.array(scores, dtype=np.float32)


def info_nce(
    z_q: torch.Tensor,
    z_d: torch.Tensor,
    z_neg: torch.Tensor | None = None,
    pos_mask: torch.Tensor | None = None,
    pos_mask_neg: torch.Tensor | None = None,
    scaled_scores: torch.Tensor | None = None,
    temperature: float = 0.05,
    symmetric: bool = False,
) -> torch.Tensor:
    """InfoNCE loss. The q->d direction uses corpus negatives, d->q is in-batch only; symmetric=True
    averages both. ``scaled_scores`` (shape [B]) multiplies each pair's cross-entropy loss before
    the batch mean."""
    zq = F.normalize(z_q, dim=1)
    zd = F.normalize(z_d, dim=1)
    targets = torch.arange(z_q.shape[0], device=z_q.device)

    def _loss(logits: torch.Tensor) -> torch.Tensor:
        per_pair = F.cross_entropy(logits, targets, reduction="none")
        if scaled_scores is None:
            return per_pair.mean()
        return (scaled_scores * per_pair).mean()

    # q -> d direction (in-batch positives + optional corpus negatives)
    logits_qd = zq @ zd.t() / temperature
    if pos_mask is not None:
        logits_qd = logits_qd.masked_fill(pos_mask, float("-inf"))
    if z_neg is not None:
        zn = F.normalize(z_neg, dim=1)
        logits_qd_neg = zq @ zn.t() / temperature
        if pos_mask_neg is not None:
            logits_qd_neg = logits_qd_neg.masked_fill(pos_mask_neg, float("-inf"))
        logits_qd = torch.cat([logits_qd, logits_qd_neg], dim=1)

    loss_qd = _loss(logits_qd)

    if not symmetric:
        return loss_qd

    # d -> q direction (in-batch only; no query external pool)
    logits_dq = zd @ zq.t() / temperature
    if pos_mask is not None:
        logits_dq = logits_dq.masked_fill(pos_mask.t(), float("-inf"))

    return 0.5 * (loss_qd + _loss(logits_dq))


class UniqueQueryBatchSampler:
    """Batch sampler that guarantees no query appears twice in the same batch."""

    def __init__(self, pairs: np.ndarray, batch_size: int, shuffle: bool = True,
                 seed: int = 42, drop_last: bool = True) -> None:
        self.pairs = np.asarray(pairs, dtype=np.int64)
        self.B = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._rng = random.Random(seed)
        self.q_to_pair_idx: dict[int, list[int]] = defaultdict(list)
        for i, (q, _) in enumerate(self.pairs):
            self.q_to_pair_idx[int(q)].append(i)
        self.queries = list(self.q_to_pair_idx.keys())
        n_full_per_pass = max(len(self.queries) // self.B, 1)
        self.n_passes = max(
            1,
            (len(self.pairs) + n_full_per_pass * self.B - 1) // (n_full_per_pass * self.B),
        )

    def __iter__(self):
        for _ in range(self.n_passes):
            queries = list(self.queries)
            if self.shuffle:
                self._rng.shuffle(queries)
            pair_idxs = [
                self._rng.choice(self.q_to_pair_idx[q]) if self.shuffle
                else self.q_to_pair_idx[q][0]
                for q in queries
            ]
            for k in range(0, len(pair_idxs), self.B):
                batch = pair_idxs[k:k + self.B]
                if len(batch) == self.B or not self.drop_last:
                    yield batch

    def __len__(self) -> int:
        return (len(self.queries) // self.B) * self.n_passes


def build_pos_dict(pairs: np.ndarray) -> dict[int, frozenset[int]]:
    """Map q_idx → frozenset(c_idx) of positive docs for each query."""
    bucket: dict[int, set[int]] = defaultdict(set)
    for q, d in pairs:
        bucket[int(q)].add(int(d))
    return {q: frozenset(s) for q, s in bucket.items()}


def build_pos_mask(q_idx, c_idx, pos_dict, device, skip_diagonal: bool = False) -> torch.Tensor:
    """(B, N) bool: True at (i, n) where (q_i, c_n) is a known positive; optional diagonal skip.
    Work scales with the number of in-batch intersections, not B*N.
    """
    B, N = len(q_idx), len(c_idx)
    q_np = q_idx.cpu().numpy() if torch.is_tensor(q_idx) else np.asarray(q_idx)
    c_np = c_idx.cpu().numpy() if torch.is_tensor(c_idx) else np.asarray(c_idx)

    c_positions: dict[int, list[int]] = defaultdict(list)
    for n_pos, c_val in enumerate(c_np):
        c_positions[int(c_val)].append(n_pos)

    rows: list[int] = []
    cols: list[int] = []
    for i, q_val in enumerate(q_np):
        positives = pos_dict.get(int(q_val))
        if not positives:
            continue
        for c_val in positives:
            positions = c_positions.get(c_val)
            if positions:
                rows.extend([i] * len(positions))
                cols.extend(positions)

    out = torch.zeros(B, N, dtype=torch.bool, device=device)
    if rows:
        out[torch.tensor(rows, device=device), torch.tensor(cols, device=device)] = True
    if skip_diagonal:
        out.fill_diagonal_(False)
    return out
