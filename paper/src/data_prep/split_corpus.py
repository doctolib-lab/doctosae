"""Split a corpus file (pkl or jsonl) into train/val, optionally forcing each split's query
positives into its own corpus. Chunks required by both are duplicated into both; the rest split
90/10 by --val-frac. Forcing neither side is a plain random partition ('_unforced' suffix).

--ids-from copies an existing split's id membership instead of partitioning, keeping every
embedder's corpus.pkl on one split.

Example (jsonl): python -m src.data_prep.split_corpus --corpus-file corpus.jsonl --corpus-id-col _id --query-id-col _id --force-train-positives --train-qrels train.tsv --train-queries queries_train.jsonl --output-dir splits
Example (pkl):   python -m src.data_prep.split_corpus --corpus-file corpus.pkl --corpus-id-col _id --ids-from splits/corpus_train.jsonl splits/corpus_val.jsonl --output-dir splits
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_prep.embed_texts import load_json_input
from src.utils.cli import expand_path_args


def _load_df(path: Path) -> pd.DataFrame:
    if ".jsonl" in path.name or path.suffix == ".json":
        return load_json_input(path)
    return pd.read_pickle(path)


def _save_df(df: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(json.dumps(row.to_dict(), ensure_ascii=False, default=str) + "\n")
    else:
        df.to_pickle(path)


def _out_ext(input_path: Path) -> str:
    return ".jsonl" if ".jsonl" in input_path.name else ".pkl"


def _required_corpus_ids(qrels_path: Path, qids: set[str]) -> set[str]:
    """Corpus-ids that must live in this split: positives (relevance>0) only."""
    qr = pd.read_csv(qrels_path, sep="\t", dtype={"query-id": str, "corpus-id": str})
    score_col = "score" if "score" in qr.columns else "relevance"
    keep = qr[score_col] > 0
    return set(qr.loc[qr["query-id"].isin(qids) & keep, "corpus-id"].unique())


def _split_ids(path: Path, corpus_id_col: str) -> set[str]:
    """Corpus ids held by an existing split file (for --ids-from)."""
    df = _load_df(path)
    if corpus_id_col not in df.columns:
        raise ValueError(f"{path}: expected column {corpus_id_col!r}, got {list(df.columns)}")
    return set(df[corpus_id_col].astype(str))


def _load_qids(path: Path, query_id_col: str) -> set[str]:
    """Load a queries file and return the set of unique query ID values."""
    df = _load_df(path)
    col = query_id_col if query_id_col in df.columns else "query_id"
    if col not in df.columns:
        raise ValueError(f"{path}: expected column {query_id_col!r}, got {list(df.columns)}")
    return set(df[col].astype(str).unique())


def _stratify_balance(df: pd.DataFrame, max_per_source: int, rng: np.random.Generator) -> pd.DataFrame:
    """Cap each ``source_folder`` group to at most ``max_per_source`` rows (smaller groups untouched)."""
    if "source_folder" not in df.columns:
        return df
    pieces = []
    for _, group in df.groupby("source_folder"):
        if len(group) > max_per_source:
            idx = rng.choice(group.index.to_numpy(), size=max_per_source, replace=False)
            pieces.append(group.loc[idx])
        else:
            pieces.append(group)
    return pd.concat(pieces).sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-file", required=True, type=Path,
                        help="Input corpus file (.pkl or .jsonl).")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--corpus-id-col", required=True,
                        help="Column that uniquely identifies each corpus row and matches the qrels 'corpus-id'.")
    parser.add_argument("--query-id-col", default="query_id",
                        help="Column in query files holding the query id (default: 'query_id'; use '_id' for BEIR jsonl).")
    parser.add_argument("--force-train-positives", action="store_true",
                        help="Force the train queries' positives into the train split (requires "
                             "--train-qrels, --train-queries). The train InfoNCE pairs need this.")
    parser.add_argument("--force-val-positives", action="store_true",
                        help="Force the val queries' positives into the val split (requires "
                             "--val-qrels, --val-queries), which also keeps them out of the train "
                             "corpus -- a condition test positives never get.")
    parser.add_argument("--ids-from", type=Path, nargs=2, default=None,
                        metavar=("TRAIN_FILE", "VAL_FILE"),
                        help="Copy the split from these files' id membership instead of partitioning.")
    parser.add_argument("--train-qrels", type=Path, default=None,
                        help="Train qrels (positives for train queries).")
    parser.add_argument("--val-qrels", type=Path, default=None,
                        help="Val qrels (positives for val queries).")
    parser.add_argument("--train-queries", type=Path, default=None,
                        help="Train queries file (.pkl or .jsonl); only the id column is used.")
    parser.add_argument("--val-queries", type=Path, default=None,
                        help="Val queries file (.pkl or .jsonl); only the id column is used.")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-source", type=int, default=0)
    parser.add_argument("--output-prefix", default="corpus")
    args = expand_path_args(parser.parse_args())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    ext = _out_ext(args.corpus_file)

    print(f"Loading {args.corpus_file}")
    df = _load_df(args.corpus_file)
    print(f"  rows: {len(df)}, columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")

    if args.corpus_id_col not in df.columns:
        raise ValueError(f"--corpus-id-col {args.corpus_id_col!r} not in corpus columns: {list(df.columns)}")

    if args.ids_from:
        train_ids = _split_ids(args.ids_from[0], args.corpus_id_col)
        val_ids   = _split_ids(args.ids_from[1], args.corpus_id_col)
        cid = df[args.corpus_id_col].astype(str)
        missing = set(cid) - (train_ids | val_ids)
        if missing:
            raise ValueError(f"{len(missing)} corpus ids appear in neither --ids-from file — the "
                             f"given split does not cover this corpus")
        train_df = df[cid.isin(train_ids)].reset_index(drop=True)
        val_df   = df[cid.isin(val_ids)].reset_index(drop=True)
        out_suffix = ""
        summary_extras = {"ids_from": [str(p) for p in args.ids_from]}
        print(f"  copied split: train={len(train_df)}  val={len(val_df)}  "
              f"(duplicated in both={len(train_ids & val_ids)})")
    elif args.force_train_positives or args.force_val_positives:
        if args.force_train_positives and (args.train_qrels is None or args.train_queries is None):
            raise ValueError("--force-train-positives requires --train-qrels and --train-queries")
        if args.force_val_positives and (args.val_qrels is None or args.val_queries is None):
            raise ValueError("--force-val-positives requires --val-qrels and --val-queries")

        train_required = (_required_corpus_ids(args.train_qrels,
                                               _load_qids(args.train_queries, args.query_id_col))
                          if args.force_train_positives else set())
        val_required   = (_required_corpus_ids(args.val_qrels,
                                               _load_qids(args.val_queries, args.query_id_col))
                          if args.force_val_positives else set())
        conflicts      = train_required & val_required
        print(f"  train required: {len(train_required)}   "
              f"val required: {len(val_required)}   "
              f"conflicts (duplicated in both): {len(conflicts)}")

        cid = df[args.corpus_id_col].astype(str)
        mask_train_req = cid.isin(train_required)
        mask_val_req   = cid.isin(val_required)
        nonreq_df = df[~(mask_train_req | mask_val_req)].reset_index(drop=True)
        print(f"  rows: train-required={int(mask_train_req.sum())}  "
              f"val-required={int(mask_val_req.sum())}  non-required={len(nonreq_df)}")

        if args.max_per_source > 0:
            before = len(nonreq_df)
            nonreq_df = _stratify_balance(nonreq_df, args.max_per_source, rng).reset_index(drop=True)
            print(f"  capped non-required per source to {args.max_per_source}: {before} -> {len(nonreq_df)}")

        n_nonreq = len(nonreq_df)
        n_val_extra = int(round(n_nonreq * args.val_frac))
        perm = rng.permutation(n_nonreq)
        val_idx = perm[:n_val_extra]
        train_extra_idx = perm[n_val_extra:]

        train_df = pd.concat([df[mask_train_req], nonreq_df.iloc[train_extra_idx]], ignore_index=True)
        val_df   = pd.concat([df[mask_val_req],   nonreq_df.iloc[val_idx]],         ignore_index=True)
        out_suffix = ""
        summary_extras = {
            "force_train_positives": bool(args.force_train_positives),
            "force_val_positives":   bool(args.force_val_positives),
            "train_qrels": str(args.train_qrels),
            "val_qrels": str(args.val_qrels),
            "train_queries": str(args.train_queries),
            "val_queries":   str(args.val_queries),
            "n_train_required": int(mask_train_req.sum()),
            "n_val_required":   int(mask_val_req.sum()),
            "n_conflicts_duplicated": int(len(conflicts)),
            "n_nonreq_after_cap": int(n_nonreq),
        }
    else:
        n = len(df)
        n_val = int(round(n * args.val_frac))
        perm = rng.permutation(n)
        val_df   = df.iloc[perm[:n_val]].reset_index(drop=True)
        train_df = df.iloc[perm[n_val:]].reset_index(drop=True)
        out_suffix = "_unforced"
        summary_extras = {}
        print(f"  plain random partition: train={len(train_df)}  val={len(val_df)}")

    train_path = args.output_dir / f"{args.output_prefix}_train{out_suffix}{ext}"
    val_path   = args.output_dir / f"{args.output_prefix}_val{out_suffix}{ext}"
    _save_df(train_df, train_path)
    _save_df(val_df,   val_path)

    summary = {
        "input": str(args.corpus_file),
        "n_total_input": int(len(df)),
        **summary_extras,
        "n_train": int(len(train_df)),
        "n_val":   int(len(val_df)),
        "train_path": str(train_path),
        "val_path":   str(val_path),
    }
    if not args.ids_from:  # meaningless when the split is copied wholesale
        summary.update(max_per_source=args.max_per_source, val_frac=args.val_frac, seed=args.seed)
    (args.output_dir / "corpus_split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
