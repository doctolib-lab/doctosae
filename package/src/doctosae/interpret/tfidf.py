"""TF IDF interpretation of a feature: the most weighted terms of its evidence texts (Appendix D.1).

One vectorizer is fit over every evidence document followed by every evidence query, with the exact settings
of the paper's ``fit_tfidf``. A feature's terms are the top terms of the plain mean of the L2 normalised rows
of its top documents and top queries (``top_terms_for_mask``): every evidence text weighs the same,
activations are not used and negatives are left out.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

__all__ = ["N_TERMS", "TFIDF_SETTINGS", "TfidfTerms"]

logger = logging.getLogger(__name__)

N_TERMS = 5

# Everything else is the ``scikit-learn`` default: lowercase, l2 row norm, smooth idf, no sublinear tf. Stop
# words go before bigrams are formed ("quality of life" gives "quality life"), and a token is two or more
# letters without digits (``covid-19`` gives ``covid``).
TFIDF_SETTINGS = {
    "max_features": 20000,
    "stop_words": "english",
    "ngram_range": (1, 2),
    "token_pattern": r"(?u)\b[^\d\W][^\d\W]+\b",
    "min_df": 3,
}


@dataclass
class TfidfTerms:
    """A TF IDF space over evidence documents then evidence queries.

    Attributes:
        matrix: L2 normalised TF IDF rows, documents first, then queries.
        vocabulary: Term of every column.
        n_documents: Number of document rows (query rows follow them).
    """

    matrix: sp.csr_matrix
    vocabulary: np.ndarray
    n_documents: int

    @classmethod
    def fit(cls, document_texts: Sequence[str], query_texts: Sequence[str] = ()) -> TfidfTerms:
        """Fit the paper's vectorizer over the documents followed by the queries (both deduplicated)."""
        texts = np.concatenate(
            [
                np.asarray(list(document_texts), dtype=object),
                np.asarray(list(query_texts), dtype=object),
            ]
        )
        vectorizer = TfidfVectorizer(**TFIDF_SETTINGS)
        try:
            matrix = vectorizer.fit_transform(texts)
        except ValueError as error:  # every term pruned: a corpus too small for min_df=3
            logger.warning("TF IDF interpretation disabled: %s", error)
            return cls(sp.csr_matrix((len(texts), 0)), np.array([], dtype=object), len(document_texts))
        return cls(sp.csr_matrix(matrix), np.array(vectorizer.get_feature_names_out()), len(document_texts))

    def terms(
        self, document_rows: Sequence[int], query_rows: Sequence[int] = (), n_terms: int = N_TERMS
    ) -> list[str]:
        """Top ``n_terms`` terms of the mean TF IDF row of the given documents and queries.

        Like the paper, ``n_terms`` terms are returned even when fewer have a non zero weight.
        """
        rows = np.sort(
            np.concatenate(
                [
                    np.asarray(document_rows, dtype=np.int64),
                    self.n_documents + np.asarray(query_rows, dtype=np.int64),
                ]
            )
        )
        if len(rows) == 0 or len(self.vocabulary) == 0:
            return []
        mean = np.asarray(self.matrix[rows].mean(axis=0)).ravel()
        return [str(term) for term in self.vocabulary[np.argsort(-mean)[:n_terms]]]
