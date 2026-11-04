"""Split a queries file (pkl or jsonl) into train/val/test, keeping rows that share
--group-by-col together (case-level split). With --emit-qrels-tsvs, also filters a source
qrels TSV into per-split TSVs.

Example (pkl):  python -m src.data_prep.split_queries --queries-file queries.pkl --output-dir splits --id-col query_id --group-by-col sample_id --emit-qrels-tsvs qrels.tsv
Example (jsonl): python -m src.data_prep.split_queries --queries-file queries_train.jsonl --output-dir splits --id-col _id --group-by-col _id
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


def split_queries(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    seed: int,
    id_col: str,
    group_by_col: str,
) -> dict[str, pd.DataFrame]:
    """Split queries 3-ways into train/val/test; rows sharing ``group_by_col`` stay together."""
    if train_frac + val_frac > 1.0 + 1e-9:
        raise ValueError("train_frac + val_frac must be ≤ 1.0 (test = remainder)")
    if id_col not in df.columns:
        raise ValueError(f"--id-col {id_col!r} not in columns: {list(df.columns)}")
    if group_by_col not in df.columns:
        raise ValueError(f"--group-by-col {group_by_col!r} not in columns: {list(df.columns)}")

    df = df.copy()
    df["query_id"] = df[id_col].astype(str)

    rng = np.random.default_rng(seed)
    groups = df[group_by_col].drop_duplicates().to_numpy().copy()
    rng.shuffle(groups)

    n = len(groups)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train_groups = set(groups[:n_train].tolist())
    val_groups   = set(groups[n_train : n_train + n_val].tolist())
    test_groups  = set(groups[n_train + n_val:].tolist())

    return {
        "train": df[df[group_by_col].isin(train_groups)].reset_index(drop=True),
        "val":   df[df[group_by_col].isin(val_groups)].reset_index(drop=True),
        "test":  df[df[group_by_col].isin(test_groups)].reset_index(drop=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries-file", required=True, type=Path,
                        help="Input queries file (.pkl or .jsonl).")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--id-col", required=True,
                        help="Column whose value is the per-row query id (copied into query_id).")
    parser.add_argument("--group-by-col", required=True,
                        help="Column whose unique values define the split unit "
                             "(pass the id col for independent row splits).")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-prefix", default="queries")
    parser.add_argument("--emit-qrels-tsvs", type=Path, default=None,
                        help="If set, filter this qrels TSV into per-split TSVs alongside the output files.")
    args = expand_path_args(parser.parse_args())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ext = _out_ext(args.queries_file)

    print(f"Loading {args.queries_file}")
    df = _load_df(args.queries_file)
    print(f"  rows: {len(df)}, columns: {list(df.columns)[:14]}{'...' if len(df.columns) > 14 else ''}")

    splits = split_queries(df, args.train_frac, args.val_frac, args.seed,
                           id_col=args.id_col, group_by_col=args.group_by_col)

    suffix = "_with_emb" if ext == ".pkl" else ""
    summary = {}
    for name, frame in splits.items():
        if len(frame) == 0:
            continue
        out_path = args.output_dir / f"{args.output_prefix}_{name}{suffix}{ext}"
        _save_df(frame, out_path)
        summary[name] = {"rows": int(len(frame)), "path": str(out_path)}
        print(f"  {name:5s} -> {out_path}  ({len(frame)} rows)")

    (args.output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))

    if args.emit_qrels_tsvs is not None:
        print(f"\nLoading qrels {args.emit_qrels_tsvs}")
        qrels = pd.read_csv(args.emit_qrels_tsvs, sep="\t", dtype={"query-id": str, "corpus-id": str})
        emit_groups = {
            "train": set(splits["train"]["query_id"]),
            "val":   set(splits["val"]["query_id"]),
            "test":  set(splits["test"]["query_id"]),
        }
        for name, qids in emit_groups.items():
            if not qids:
                continue
            sub = qrels[qrels["query-id"].isin(qids)]
            out_path = args.output_dir / f"{args.output_prefix}_{name}_qrels.tsv"
            sub.to_csv(out_path, sep="\t", index=False)
            print(f"  {name:8s} -> {out_path}  ({len(sub)} rows)")


if __name__ == "__main__":
    main()
