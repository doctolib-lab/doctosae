"""Dataset definitions: every dataset's query/corpus/qrels paths, id columns and score column,
read by baselines.py and by eval_ood_cross_domain_sweep.sh via this module's CLI.
"""
from __future__ import annotations

import os
from pathlib import Path

# Root of every dataset folder below: the DATA_ROOT environment variable, else ~/data.
DATA = Path(os.environ.get("DATA_ROOT", "~/data")).expanduser()

# Query/corpus/qrels paths, relative to DATA with "{embedder}" filled in per call. Datasets with
# no train/val split of their own use "test_corpus" alone.
DATASET_PATHS = {
    "CURE": {
        "test_queries": "cure-beir/{embedder}/queries_test_en.pkl",
        "test_qrels":   "cure-beir/queries_split/qrels_test.tsv",
        "train_corpus": "cure-beir/{embedder}/corpus_train.pkl",
        "val_corpus":   "cure-beir/{embedder}/corpus_val.pkl",
        "corpus_id_col": "_id", "query_id_col": "_id", "score_col": "score", "max_score": 2,
        "genre": "academic literature",
    },
    "MIRIAD": {
        # corpus_test.pkl stands alone: prepare_data.py bakes a train-doc background slice in.
        "test_queries": "miriad-beir/{embedder}/queries_test.pkl",
        "test_qrels":   "miriad-beir/qrels/test.tsv",
        "test_corpus":  "miriad-beir/{embedder}/corpus_test.pkl",
        "corpus_id_col": "_id", "query_id_col": "_id", "score_col": "relevance", "max_score": 1,
        "genre": "academic literature",
    },
    "MedEmbed": {
        "test_queries": "medembed-triplets-beir/{embedder}/queries_test.pkl",
        "test_qrels":   "medembed-triplets-beir/queries_split/queries_test_qrels.tsv",
        "train_corpus": "medembed-triplets-beir/{embedder}/corpus_train.pkl",
        "val_corpus":   "medembed-triplets-beir/{embedder}/corpus_val.pkl",
        "corpus_id_col": "_id", "query_id_col": "_id", "score_col": "score", "max_score": 1,
    },
    "NFCorpus": {
        "test_queries": "NFCorpus/{embedder}/queries_test.pkl",
        "test_qrels":   "NFCorpus/qrels/test.tsv",
        "test_corpus":  "NFCorpus/{embedder}/corpus.pkl",
        "corpus_id_col": "_id", "query_id_col": "_id", "score_col": "score", "max_score": 2,
        "genre": "academic literature",
    },
    "Bioinformatics": {
        "test_queries": "R2MED-beir/Bioinformatics/{embedder}/query.pkl",
        "test_qrels":   "R2MED-beir/Bioinformatics/qrels.tsv",
        "test_corpus":  "R2MED-beir/Bioinformatics/{embedder}/corpus.pkl",
        "corpus_id_col": "id", "query_id_col": "id", "score_col": "score", "max_score": 1,
        "collection": "R2MED", "genre": "StackExchange queries over Wikipedia",
    },
    "Biology": {
        "test_queries": "R2MED-beir/Biology/{embedder}/query.pkl",
        "test_qrels":   "R2MED-beir/Biology/qrels.tsv",
        "test_corpus":  "R2MED-beir/Biology/{embedder}/corpus.pkl",
        "corpus_id_col": "id", "query_id_col": "id", "score_col": "score", "max_score": 1,
        "collection": "R2MED", "genre": "StackExchange queries over Wikipedia",
    },
    "Medical-Sciences": {
        "test_queries": "R2MED-beir/Medical-Sciences/{embedder}/query.pkl",
        "test_qrels":   "R2MED-beir/Medical-Sciences/qrels.tsv",
        "test_corpus":  "R2MED-beir/Medical-Sciences/{embedder}/corpus.pkl",
        "corpus_id_col": "id", "query_id_col": "id", "score_col": "score", "max_score": 1,
        "collection": "R2MED", "genre": "StackExchange queries over Wikipedia",
    },
}

# Default cross-domain targets of eval_ood_cross_domain_sweep.sh.
OOD_DATASETS = ["NFCorpus", "Bioinformatics", "Biology", "Medical-Sciences", "MIRIAD"]

# Root path fragment in a sweep's own corpus flags -> the DATASET_PATHS key it is scored against.
SWEEP_DATASET_ROOTS = {"cure-beir": "CURE", "miriad-beir": "MIRIAD",
                       "medembed-triplets-beir": "MedEmbed"}

# Dataset -> the OOD_DATASETS entry to drop for a sweep trained on it.
OOD_SELF_EXCLUDE = {"MIRIAD": "MIRIAD"}


def resolve_dataset(name: str, embedder: str) -> dict:
    """Resolve a DATASET_PATHS entry's "{embedder}" templates to absolute paths, returning
    {corpus_paths, queries, qrels, corpus_id_col, query_id_col, score_col}. corpus_paths is
    test_corpus alone, or [train_corpus, val_corpus].
    """
    if name not in DATASET_PATHS:
        raise ValueError(f"No DATASET_PATHS entry for {name!r} — known: {sorted(DATASET_PATHS)}")
    paths = DATASET_PATHS[name]
    resolved = {k: (DATA / v.format(embedder=embedder) if k.endswith(("queries", "qrels", "corpus")) else v)
               for k, v in paths.items()}
    corpus_paths = [resolved["test_corpus"]] if "test_corpus" in resolved \
        else [resolved["train_corpus"], resolved["val_corpus"]]
    return {
        "corpus_paths": corpus_paths,
        "queries": resolved["test_queries"],
        "qrels": resolved["test_qrels"],
        "corpus_id_col": resolved["corpus_id_col"],
        "query_id_col": resolved["query_id_col"],
        "score_col": resolved["score_col"],
    }


def max_score_for(name: str) -> float:
    """DATASET_PATHS[name]'s qrels score ceiling (default 1.0 if unset)."""
    return DATASET_PATHS.get(name, {}).get("max_score", 1.0)


def collection_of(name: str) -> str | None:
    """DATASET_PATHS[name]'s named collection (e.g. "R2MED" for its 3 subsets), None if it doesn't
    belong to one."""
    return DATASET_PATHS.get(name, {}).get("collection")


def genre_of(name: str) -> str | None:
    """DATASET_PATHS[name]'s text genre (e.g. "academic literature"), None if unset."""
    return DATASET_PATHS.get(name, {}).get("genre")


def dataset_from_sweep(grid: dict) -> str:
    """Infer the DATASET_PATHS key a training sweep's own baseline should use, from the root fragment
    appearing in its parsed corpus paths.
    """
    path = str(grid.get("train_corpus") or grid.get("test_corpus") or "").lower()
    for root, name in SWEEP_DATASET_ROOTS.items():
        if root in path:
            return name
    raise ValueError(f"Could not infer dataset from sweep corpus path {path!r} — "
                     f"known roots: {sorted(SWEEP_DATASET_ROOTS)}")


def ood_datasets_for(dataset: str) -> list[str]:
    """OOD_DATASETS with `dataset`'s own entry excluded, if any (see OOD_SELF_EXCLUDE)."""
    exclude = OOD_SELF_EXCLUDE.get(dataset)
    return [d for d in OOD_DATASETS if d != exclude]


def ood_corpus_files(ood_dir: Path, dataset: str, exclude: list[str] | None = None,
                     pattern: str = "corpus*_with_sae.pkl",
                     required: bool = True) -> list[tuple[str, Path]]:
    """(name, converted pkl) for each ood_datasets_for(dataset) under ood_dir, minus exclude.
    `pattern` selects the kind, e.g. "quer*_with_sae.pkl" for queries. Raises if a dataset isn't
    converted yet (unless required=False) or if a directory holds several matches.
    """
    out = []
    for name in ood_datasets_for(dataset):
        if name in (exclude or []):
            continue
        hits = sorted((ood_dir / name).glob(pattern))
        if not hits:
            if not required:
                continue
            raise FileNotFoundError(f"no {pattern} in {ood_dir / name} — convert it "
                                    f"first (see scripts/eval_ood_cross_domain_sweep.sh), or pass "
                                    f"--exclude {name}")
        if len(hits) > 1:
            raise ValueError(f"{ood_dir / name} holds several {pattern} "
                             f"({', '.join(h.name for h in hits)}) — expected one")
        out.append((name, hits[0]))
    return out


if __name__ == "__main__":
    # CLI for eval_ood_cross_domain_sweep.sh: one TSV line per OOD dataset.
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embedder", required=True)
    p.add_argument("--sweep-dataset", default=None, help="Self-exclude this dataset's OOD entry, if any")
    args = p.parse_args()
    for name in ood_datasets_for(args.sweep_dataset) if args.sweep_dataset else OOD_DATASETS:
        r = resolve_dataset(name, args.embedder)
        corpus = ",".join(str(p) for p in r["corpus_paths"])
        print(f"{name}\t{corpus}\t{r['queries']}\t{r['qrels']}\t{r['corpus_id_col']}\t{r['score_col']}")
