"""Which features to interpret: firing rate band, minimum firing counts and optional theme entropy order.

Follows ``select_neurons.select_neurons`` of the paper code. The paper run kept every feature (band
``[0, 1]``) that fires on at least 5 evidence documents and 5 evidence queries (Appendix D.1); the theme
entropy only ordered the output, most theme pure first.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp

from doctosae.interpret.activations import ActivationIndex

__all__ = ["select_features", "theme_entropy"]


def _missing(label: Any) -> bool:
    """None, NaN or an empty list: a row without a theme (``is_missing_label``)."""
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return True
    return isinstance(label, (list, tuple, set)) and len(label) == 0


def theme_entropy(codes: sp.spmatrix, labels: Sequence[Any]) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    """Normalised entropy of each feature over the themes of the rows it fires on.

    Mirrors ``neuron_theme_entropy`` of the paper code: rows without a theme are dropped, a row with several
    themes splits its weight evenly, and the distribution is ``P(fires | theme)`` renormalised over themes, so
    that large themes do not dominate.

    Args:
        codes: Codes ``[n_rows, n_latents]``.
        labels: One theme per row (a string, or a list of strings for several themes).

    Returns:
        ``(entropy_norm, p, themes)``: the entropy divided by ``ln K`` (NaN for a feature that fires on no
        labelled row), the renormalised distribution ``[n_latents, K]`` and the ``K`` sorted theme names.
    """
    keep = np.flatnonzero([not _missing(label) for label in labels])
    rows = [labels[i] if isinstance(labels[i], (list, tuple, set)) else [labels[i]] for i in keep]
    rows = [[theme for theme in row if theme is not None] for row in rows]
    names = sorted({theme for row in rows for theme in row})
    if len(names) <= 1:
        raise ValueError(f"theme entropy needs more than one theme, got {names}")
    position = {name: t for t, name in enumerate(names)}
    onehot = np.zeros((len(rows), len(names)), dtype=np.float32)
    for i, row in enumerate(rows):
        for theme in row:
            onehot[i, position[theme]] = 1.0 / len(row)
    fired = sp.csr_matrix(codes)[keep] > 0
    n_fired = np.asarray(fired.sum(axis=0)).ravel()
    counts = np.asarray(fired.T.astype(np.float32) @ onehot)
    sizes = onehot.sum(axis=0)
    fire_given_theme = counts / np.where(sizes > 0, sizes, 1)[None, :]
    row_sum = fire_given_theme.sum(axis=1)
    live = (n_fired > 0) & (row_sum > 0)
    p = np.zeros_like(fire_given_theme)
    p[live] = fire_given_theme[live] / row_sum[live, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=1)
    entropy[~live] = np.nan
    return entropy / np.log(len(names)), p, names


def select_features(
    index: ActivationIndex,
    n: int | None = None,
    min_documents: int = 5,
    min_queries: int = 5,
    firing_rate_band: tuple[float, float] = (0.0, 1.0),
    themes: Sequence[Any] | None = None,
) -> list[int]:
    """Features worth interpreting, in the paper's order.

    Args:
        index: Evidence documents and queries with their codes.
        n: Keep the first ``n`` features after ordering (all by default).
        min_documents: Minimum number of evidence documents the feature fires on. Checked on the raw counts,
            before the two passages per article cap, as in the paper.
        min_queries: Minimum number of evidence queries it fires on (ignored without queries).
        firing_rate_band: Inclusive band on the document firing rate. The paper run used ``(0.0, 1.0)``, that
            is no band; the paper code default was ``(0.01, 0.20)``.
        themes: One theme per evidence document (for example CURE ``source_folder``). When given, features are
            ordered by ascending normalised theme entropy (most theme pure first), else by index.
    """
    low, high = firing_rate_band
    rates = index.firing_rates
    keep = (rates >= low) & (rates <= high) & (index.document_counts >= min_documents)
    if index.query_counts is not None:
        keep &= index.query_counts >= min_queries
    candidates = np.flatnonzero(keep)
    if themes is not None:
        if len(themes) != len(index.documents):
            raise ValueError(f"{len(themes)} themes for {len(index.documents)} evidence documents")
        entropy_norm, _, _ = theme_entropy(index.documents.codes, themes)
        # Default quicksort, NaN last: the same order as select_neurons.py.
        candidates = candidates[np.argsort(entropy_norm[candidates])]
    return [int(feature) for feature in candidates[:n]]
