"""Split CUREv1 queries into train/val/test, applying one id assignment across all languages.

CUREv1 query ids are shared across languages, so the split is computed once on one reference
language's ids via split_queries() and applied to every language file and to qrels.

Example:
    python -m src.data_prep.cure.split_multilang_queries \
      --queries-dir ~/data/cure-beir --languages en es fr \
      --qrels ~/data/cure-beir/qrels/all.tsv \
      --output-dir ~/data/cure-beir/queries_split
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_prep.embed_texts import load_json_input
from src.data_prep.split_queries import _save_df, split_queries


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--queries-dir", required=True, type=Path,
                   help="Dir containing queries_<lang>.jsonl (one per --languages entry).")
    p.add_argument("--languages", nargs="+", default=["en", "es", "fr"])
    p.add_argument("--qrels", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ref_lang = args.languages[0]
    ref_df = load_json_input(args.queries_dir / f"queries_{ref_lang}.jsonl")
    print(f"Computing split on queries_{ref_lang}.jsonl ({len(ref_df)} rows)")
    ref_splits = split_queries(ref_df, args.train_frac, args.val_frac, args.seed, id_col="_id", group_by_col="_id")
    split_ids = {name: set(df["query_id"]) for name, df in ref_splits.items() if len(df)}
    print("  " + ", ".join(f"{name}={len(ids)} ids" for name, ids in split_ids.items()))

    for lang in args.languages:
        df = load_json_input(args.queries_dir / f"queries_{lang}.jsonl")
        ids_str = df["_id"].astype(str)
        for name, ids in split_ids.items():
            sub = df[ids_str.isin(ids)].reset_index(drop=True)
            out_path = args.output_dir / f"queries_{name}_{lang}.jsonl"
            _save_df(sub, out_path)
            print(f"  {name:5s} {lang}: {len(sub)} rows -> {out_path.name}")

    qrels = pd.read_csv(args.qrels, sep="\t", dtype={"query-id": str, "corpus-id": str})
    for name, ids in split_ids.items():
        sub = qrels[qrels["query-id"].isin(ids)]
        out_path = args.output_dir / f"qrels_{name}.tsv"
        sub.to_csv(out_path, sep="\t", index=False)
        print(f"  {name:5s} qrels: {len(sub)} rows -> {out_path.name}")


if __name__ == "__main__":
    main()
