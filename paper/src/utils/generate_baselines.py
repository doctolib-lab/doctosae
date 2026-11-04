"""Compute and cache original (no-SAE) baselines for one or more datasets on one embedder.

Reads baselines.DATASET_PATHS by dataset name and populates the one-JSON-per-embedder cache
(baselines.get_baseline).

Usage:
    python -m src.utils.generate_baselines --embedder MedEmbed-large-v0.1          # every dataset
    python -m src.utils.generate_baselines --embedder MedEmbed-large-v0.1 --name CURE NFCorpus
"""
from __future__ import annotations

import argparse

from src.utils.baselines import get_dataset_baseline
from src.utils.datasets_constants import DATASET_PATHS

DEFAULT_METRIC = "test_ndcg"
METRIC_OVERRIDES: dict[str, str] = {}  # per-dataset metric, when it isn't test_ndcg


def generate(embedder: str, names: list[str] | None = None, k: int = 10) -> None:
    for name in names or sorted(DATASET_PATHS):
        metric = METRIC_OVERRIDES.get(name, DEFAULT_METRIC)
        score = get_dataset_baseline(name, embedder, metric, k)
        print(f"{name} ({embedder}) {metric} = {score}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embedder", required=True, help="Embedder identifier, e.g. bge-large-en-v1.5, MedEmbed-large-v0.1")
    p.add_argument("--name",     nargs="+", default=None, choices=sorted(DATASET_PATHS),
                   help="Dataset name(s) — omit to compute every known dataset")
    p.add_argument("--k",        type=int, default=10, help="k for ndcg@k baselines")
    args = p.parse_args()
    generate(args.embedder, args.name, args.k)
