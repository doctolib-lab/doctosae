"""Compute and cache original (no-SAE) baseline retrieval scores.

A baseline is a property of a (dataset, embedder) pair; dataset paths and id columns come from
datasets_constants.py.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.utils.datasets_constants import resolve_dataset

BASELINES_DIR = Path(__file__).parent.parent.parent / "baselines"  # repo root, alongside figures/


def get_baseline(embedder: str, dataset: str, queries_name: str, metric: str, compute_fn,
                 baselines_dir: Path = BASELINES_DIR) -> float | None:
    """One JSON per embedder (baselines_dir/{embedder}.json), appended to as new (dataset,
    queries_name, metric) triples are computed. ``queries_name`` is the query file's stem.
    """
    cache = baselines_dir / f"{embedder}.json"
    entries = json.loads(cache.read_text()) if cache.exists() else []
    for e in entries:
        if e["dataset"] == dataset and e["queries"] == queries_name and e["metric"] == metric:
            return e["score"]
    score = compute_fn()
    if score is None:
        return None
    entries.append({"dataset": dataset, "queries": queries_name, "metric": metric, "score": score})
    baselines_dir.mkdir(exist_ok=True)
    cache.write_text(json.dumps(entries, indent=2))
    return score


def get_dataset_baseline(name: str, embedder: str, metric: str, k: int = 10,
                         baselines_dir: Path = BASELINES_DIR) -> float | None:
    """Original (no-SAE) baseline for `name` (a DATASET_PATHS key) on `embedder`'s embeddings;
    compute and cache (via get_baseline, one JSON per embedder) if needed.
    """
    r = resolve_dataset(name, embedder)
    test_queries, test_qrels = r["queries"], r["qrels"]
    corpus_id_col, query_id_col, score_col = r["corpus_id_col"], r["query_id_col"], r["score_col"]
    queries_name = test_queries.stem

    def _compute() -> float | None:
        eval_corpus_paths = r["corpus_paths"]
        missing = [p for p in [test_queries, test_qrels, *eval_corpus_paths] if not p or not p.exists()]
        if missing:
            print(f"[baseline] no embeddings found for {name} ({embedder}) — skipping")
            return None

        eval_metric = metric.split("test_", 1)[1] if metric.startswith("test_") else metric
        cmd = [sys.executable, "-m", "src.evaluation.eval_ir",
               "--queries-file",  str(test_queries),
               "--corpus-file", *[str(p) for p in eval_corpus_paths],
               "--qrels-file",    str(test_qrels),
               "--metric",        eval_metric,
               "--emb-type",      "embeddings",
               "--corpus-id-col", corpus_id_col,
               "--query-id-col",  query_id_col,
               "--score-col",     score_col]
        if eval_metric == "ndcg":
            cmd += ["--k", str(k)]

        print(f"[baseline] computing {name} ({embedder}, {queries_name}) {metric}...")
        out = subprocess.run(cmd, capture_output=True, text=True)
        score = None
        for line in out.stdout.splitlines():
            if "=" in line and any(tok in line.lower() for tok in ("auc", "ndcg@")):
                try:
                    score = float(line.split("=")[-1].split()[0])
                    break
                except (ValueError, IndexError):
                    pass
        if score is None:
            print(f"[baseline] failed to parse eval_ir output:\n{out.stdout[-500:]}\n{out.stderr[-200:]}")
            return None
        print(f"[baseline] {name} ({embedder}, {queries_name}) {metric} = {score:.4f}")
        return score

    eval_metric = metric.split("test_", 1)[1] if metric.startswith("test_") else metric
    cache_metric = f"ndcg{k}" if eval_metric == "ndcg" else eval_metric
    return get_baseline(embedder, name, queries_name, cache_metric, _compute, baselines_dir)
