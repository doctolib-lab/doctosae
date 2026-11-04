"""Prepare MIRIAD dataset in BEIR format for retrieval evaluation and fine-tuning.

Outputs per split (train/eval/test):
  <output-dir>/corpus_{split}.jsonl   — passages (eval/test include 30k train background)
  <output-dir>/queries_{split}.jsonl  — questions
  <output-dir>/qrels/{split}.tsv      — relevance judgements (each question paired with its passage)

Example:
    python -m src.data_prep.miriad.prepare_data --output-dir ~/data/miriad-beir
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_qrels(path: Path, n: int, prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\trelevance\n")
        for i in range(n):
            f.write(f"{prefix}-{i}\t{prefix}-{i}\t1\n")


def _corpus(passages: list[str], prefix: str, background: list[str] | None = None) -> list[dict]:
    rows = [{"_id": f"{prefix}-{i}", "title": "", "text": t} for i, t in enumerate(passages)]
    if background:
        rows += [{"_id": f"train-{i}", "title": "", "text": t} for i, t in enumerate(background)]
    return rows


def _queries(questions: list[str], prefix: str) -> list[dict]:
    return [{"_id": f"{prefix}-{i}", "text": q} for i, q in enumerate(questions)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--n-train", type=int, default=100_000)
    p.add_argument("--n-eval", type=int, default=1_000)
    p.add_argument("--n-test", type=int, default=1_000)
    p.add_argument("--n-train-background", type=int, default=30_000,
                   help="Train passages added to eval/test corpus as background.")
    p.add_argument("--data-path", default=os.environ.get("MIRIAD_PATH", "tomaarsen/miriad-4.4M-split"),
                   help="Hugging Face dataset id or local copy with train, eval and test splits holding "
                        "the question and passage_text columns (default: $MIRIAD_PATH, else "
                        "tomaarsen/miriad-4.4M-split; the first download of its train split is about 12 GB)")
    args = p.parse_args()

    print(f"Loading MIRIAD from {args.data_path}")
    train_ds = load_dataset(args.data_path, split="train").select(range(args.n_train))
    eval_ds = load_dataset(args.data_path, split="eval").select(range(args.n_eval))
    test_ds = load_dataset(args.data_path, split="test").select(range(args.n_test))

    bg = list(train_ds["passage_text"][:args.n_train_background])

    out = args.output_dir
    for split, ds, background in [
        ("train", train_ds, None),
        ("eval",  eval_ds,  bg),
        ("test",  test_ds,  bg),
    ]:
        prefix = split
        passages = list(ds["passage_text"])
        questions = list(ds["question"])

        _write_jsonl(out / f"corpus_{split}.jsonl", _corpus(passages, prefix, background))
        _write_jsonl(out / f"queries_{split}.jsonl", _queries(questions, prefix))
        _write_qrels(out / "qrels" / f"{split}.tsv", len(ds), prefix)

        corpus_size = len(passages) + (len(background) if background else 0)
        print(f"{split:5s}  queries={len(ds)}  corpus={corpus_size}")

    print(f"\nDone → {out}")


if __name__ == "__main__":
    main()
