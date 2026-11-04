"""Prepare abhinand/MedEmbed-training-triplets-v1 in BEIR format.

Reads the dataset's local parquet configs (already downloaded via
`hf download abhinand/MedEmbed-training-triplets-v1 --repo-type dataset`) and writes:
  <output-dir>/corpus.jsonl   — passages, from the "corpus" config (27,590 rows)
  <output-dir>/queries.jsonl  — questions, from the "queries" config (21,689 rows)
  <output-dir>/qrels/all.tsv  — one positive per query, from the "merged" config's
                                (query_id, pos_id) pairs (score=1) — explicit hard negatives
                                (neg_id) are dropped; not used anywhere downstream.

Example:
    python -m src.data_prep.medembed.prepare_data \
      --data-dir ~/data/medembed-training-triplets-v1 \
      --output-dir ~/data/medembed-triplets-beir
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_qrels(path: Path, query_ids: list[str], corpus_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for qid, cid in zip(query_ids, corpus_ids):
            f.write(f"{qid}\t{cid}\t1\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, type=Path,
                   help="Local dir from `hf download abhinand/MedEmbed-training-triplets-v1 --repo-type dataset`")
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()

    corpus = pd.read_parquet(args.data_dir / "corpus" / "train-00000-of-00001.parquet")
    queries = pd.read_parquet(args.data_dir / "queries" / "train-00000-of-00001.parquet")
    merged = pd.read_parquet(args.data_dir / "merged" / "train-00000-of-00001.parquet")

    out = args.output_dir
    _write_jsonl(out / "corpus.jsonl",
                [{"_id": r.id, "title": "", "text": r.text} for r in corpus.itertuples()])
    _write_jsonl(out / "queries.jsonl",
                [{"_id": r.id, "text": r.query} for r in queries.itertuples()])
    _write_qrels(out / "qrels" / "all.tsv", merged["query_id"].tolist(), merged["pos_id"].tolist())

    print(f"corpus={len(corpus)}  queries={len(queries)}  qrels={len(merged)}")
    print(f"\nDone → {out}")


if __name__ == "__main__":
    main()
