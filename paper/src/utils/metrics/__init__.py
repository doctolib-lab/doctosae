"""IR metrics package. Re-exports from ir.py so existing callers
(`from src.utils.metrics import load_qrels_matrix, ndcg_at_k, ...`) keep working.
"""
from .ir import (average_ranks, cosine_scores, load_qrels_matrix, mean_auc, ndcg_at_k,
                 precision_recall_at_k, rrf_scores)

__all__ = ["average_ranks", "cosine_scores", "load_qrels_matrix", "mean_auc", "ndcg_at_k",
           "precision_recall_at_k", "rrf_scores"]
