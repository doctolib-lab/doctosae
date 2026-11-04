"""Download CUREv1 (clinia/CUREv1) and consolidate its per-discipline HF configs into BEIR format.

CUREv1 ships corpus/qrels/queries as separate HF configs, each split by medical discipline.
Query ids are shared across languages, so queries_<lang>.jsonl files line up 1:1 by "_id" and a
train/val split is computed once on that shared id space and applied to every language file.

Outputs:
    <output-dir>/corpus.jsonl          — {_id, title, text, source_folder} across all disciplines
    <output-dir>/qrels/all.tsv         — {query-id, corpus-id, score} across all disciplines
    <output-dir>/queries_<lang>.jsonl  — {_id, text, expertise, discipline} per language

Example:
    python -m src.data_prep.cure.prepare_data --output-dir ~/data/cure-beir --languages en es fr
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import get_dataset_split_names, load_dataset

_DATASET = "clinia/CUREv1"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_corpus(disciplines: list[str]) -> list[dict]:
    """A passage can belong to more than one discipline; keep only its first occurrence
    (title/text are identical across duplicates, so which discipline's tag wins is arbitrary)."""
    rows = {}
    for disc in disciplines:
        ds = load_dataset(_DATASET, "corpus", split=disc)
        for r in ds:
            rows.setdefault(r["_id"], {"_id": r["_id"], "title": r["title"], "text": r["text"], "source_folder": disc})
    return list(rows.values())


def _build_qrels(disciplines: list[str]) -> list[dict]:
    rows = []
    for disc in disciplines:
        ds = load_dataset(_DATASET, "qrels", split=disc)
        rows.extend({"query-id": r["query-id"], "corpus-id": r["corpus-id"], "score": r["score"]} for r in ds)
    return rows


def _write_qrels_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for r in rows:
            f.write(f"{r['query-id']}\t{r['corpus-id']}\t{r['score']}\n")


def _build_queries(disciplines: list[str], lang: str) -> list[dict]:
    rows = []
    for disc in disciplines:
        ds = load_dataset(_DATASET, f"queries-{lang}", split=disc)
        rows.extend({"_id": r["_id"], "text": r["text"], "expertise": r["expertise"], "discipline": disc} for r in ds)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--languages", nargs="+", default=["en", "es", "fr"], choices=["en", "es", "fr"])
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    disciplines = get_dataset_split_names(_DATASET, "corpus")
    print(f"Disciplines ({len(disciplines)}): {disciplines}")

    corpus_rows = _build_corpus(disciplines)
    _write_jsonl(args.output_dir / "corpus.jsonl", corpus_rows)
    print(f"corpus: {len(corpus_rows)} passages -> corpus.jsonl")

    qrels_rows = _build_qrels(disciplines)
    _write_qrels_tsv(args.output_dir / "qrels" / "all.tsv", qrels_rows)
    print(f"qrels: {len(qrels_rows)} judgements -> qrels/all.tsv")

    for lang in args.languages:
        query_rows = _build_queries(disciplines, lang)
        _write_jsonl(args.output_dir / f"queries_{lang}.jsonl", query_rows)
        print(f"queries ({lang}): {len(query_rows)} -> queries_{lang}.jsonl")


if __name__ == "__main__":
    main()
