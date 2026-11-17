"""Loss terms of the paper (Section 3.1 and Appendix A) with the reductions of its released code.

The released code (``paper/src/utils/training_utils.py`` and ``contrastive_utils.py``) differs from
Equation 5 of the text in two ways, which together set the scale the InfoNCE weights of the paper
(``5e-6`` and ``1e-5``) were tuned for. The squared error is averaged over dimensions as well as rows,
and the document and query terms are combined as an average weighted by ``alpha_q``. The functions below
keep those reductions.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence, Set

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["reconstruction_loss", "squared_error", "weighted_loss", "positive_mask", "info_nce_loss"]


def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Squared error averaged over rows and dimensions (``F.mse_loss`` with ``reduction="mean"``)."""
    return F.mse_loss(x_hat, x, reduction="mean")


def squared_error(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Squared L2 error summed over dimensions, averaged over rows: the scale of the auxiliary loss."""
    return (x - x_hat).pow(2).sum(dim=-1).mean()


def weighted_loss(
    corpus_value: torch.Tensor, query_value: torch.Tensor, n_corpus: int, n_queries: int, alpha_q: float
) -> torch.Tensor:
    """Average of a document and a query term, weighted by batch size and by ``alpha_q`` for queries.

    ``alpha_q = 0`` keeps documents only and ``alpha_q = 1`` weights every row equally. The paper uses ``0.1``
    because each step reconstructs as many queries as documents while queries are far fewer (1,400 training
    queries against 223,425 training documents on CURE).
    """
    return (corpus_value * n_corpus + alpha_q * query_value * n_queries) / (n_corpus + alpha_q * n_queries)


def _as_rows(values: Sequence[int] | np.ndarray | torch.Tensor) -> list[int]:
    if isinstance(values, torch.Tensor):
        values = values.cpu().numpy()
    return np.asarray(values).tolist()


def positive_mask(
    query_rows: Sequence[int] | np.ndarray | torch.Tensor,
    doc_rows: Sequence[int] | np.ndarray | torch.Tensor,
    positives: Mapping[int, Set[int]],
    device: torch.device | str | None = None,
    skip_diagonal: bool = False,
) -> torch.Tensor:
    """Boolean ``[len(query_rows), len(doc_rows)]`` mask, True where a document is a positive of the query.

    Masked logits are set to minus infinity in :func:`info_nce_loss`, so that a query's other relevant
    documents never act as its negatives. With ``skip_diagonal`` the diagonal (each anchor's own positive
    in the pair batch) stays unmasked. Same result as ``build_pos_mask`` of the paper code; the work
    scales with the number of positives of the batch queries, not with the size of the mask.
    """
    query_list, doc_list = _as_rows(query_rows), _as_rows(doc_rows)
    columns_of: dict[int, list[int]] = defaultdict(list)
    for column, doc in enumerate(doc_list):
        columns_of[doc].append(column)
    rows: list[int] = []
    columns: list[int] = []
    for row, query in enumerate(query_list):
        for doc in positives.get(query, ()):
            hits = columns_of.get(doc)
            if hits:
                rows.extend([row] * len(hits))
                columns.extend(hits)
    mask = torch.zeros(len(query_list), len(doc_list), dtype=torch.bool, device=device)
    if rows:
        mask[torch.tensor(rows, device=device), torch.tensor(columns, device=device)] = True
    if skip_diagonal:
        mask.fill_diagonal_(False)
    return mask


def info_nce_loss(
    query_codes: torch.Tensor,
    doc_codes: torch.Tensor,
    negative_codes: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    negative_mask: torch.Tensor | None = None,
    temperature: float = 0.05,
) -> torch.Tensor:
    """InfoNCE from queries to documents over sparse codes (paper Equation 6, ``info_nce`` of the paper code).

    Row ``i`` of ``query_codes`` is an anchor whose positive is row ``i`` of ``doc_codes``. Its negatives
    are the other rows of ``doc_codes`` and every row of ``negative_codes`` (the reconstruction batch of
    documents during training). Logits are cosine similarities divided by ``temperature``; entries where
    ``mask`` (anchors x pair documents) or ``negative_mask`` (anchors x negatives) is True are set to
    minus infinity. The cross entropy is averaged over anchors. An all zero code has cosine 0 with
    everything (``F.normalize`` keeps it at zero).
    """
    anchors = F.normalize(query_codes, dim=1)
    logits = anchors @ F.normalize(doc_codes, dim=1).t() / temperature
    if mask is not None:
        logits = logits.masked_fill(mask, float("-inf"))
    if negative_codes is not None:
        negative_logits = anchors @ F.normalize(negative_codes, dim=1).t() / temperature
        if negative_mask is not None:
            negative_logits = negative_logits.masked_fill(negative_mask, float("-inf"))
        logits = torch.cat([logits, negative_logits], dim=1)
    targets = torch.arange(anchors.shape[0], device=anchors.device)
    return F.cross_entropy(logits, targets, reduction="none").mean()
