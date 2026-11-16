"""Split a BeIR queries.jsonl into one JSONL per qrels TSV (train/dev/test), keying on each row's _id.
Used for NFC where the split membership of each query is decided by which qrels TSV contains its qid.

Example: python -m src.data_prep.nfc.filter_queries_by_qrels --queries queries.jsonl --qrels-dir qrels/ --output-dir queries_split/
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd

from src.utils.cli import expand_path_args


def _load_qrels_query_ids(qrels_path: Path) -> set[str]:
    """Read a qrels TSV and return the set of unique query-ids in it."""
    qr = pd.read_csv(qrels_path, sep="\t")
    return set(qr["query-id"].astype(str).tolist())


def _open_text(path: Path):
    """Open a .jsonl or .jsonl.gz file in text mode."""
    return gzip.open(path, "rt", encoding="utf-8") if path.name.endswith(".gz") else open(path, "r", encoding="utf-8")


def main() -> None:
    """CLI: split a JSONL queries file by qrels-membership; one output JSONL per *.tsv in --qrels-dir."""
    p = argparse.ArgumentParser()
    p.add_argument("--queries", required=True, type=Path,
                   help="Input JSONL (or .jsonl.gz) with at least an '_id' field per row.")
    p.add_argument("--qrels-dir", required=True, type=Path,
                   help="Directory of qrels TSVs (e.g. train.tsv/dev.tsv/test.tsv). "
                        "One output JSONL is written per TSV.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Directory to write the per-split JSONL files.")
    p.add_argument("--output-prefix", default="queries",
                   help="Output filename prefix; files are <prefix>_<split>.jsonl.")
    p.add_argument("--id-field", default="_id",
                   help="JSONL field name holding the row id (default '_id', BeIR convention).")
    args = expand_path_args(p.parse_args())

    qrels_files = sorted(args.qrels_dir.glob("*.tsv"))
    if not qrels_files:
        raise FileNotFoundError(f"No *.tsv files in {args.qrels_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = {qf.stem: _load_qrels_query_ids(qf) for qf in qrels_files}
    for name, ids in splits.items():
        print(f"[{name}] {len(ids)} unique qids")

    writers = {
        name: open(args.output_dir / f"{args.output_prefix}_{name}.jsonl", "w", encoding="utf-8")
        for name in splits
    }
    counts = {name: 0 for name in splits}
    seen = 0
    try:
        with _open_text(args.queries) as f:
            for line in f:
                if not line.strip():
                    continue
                seen += 1
                row = json.loads(line)
                qid = str(row.get(args.id_field))
                out_line = line if line.endswith("\n") else line + "\n"
                for name, ids in splits.items():
                    if qid in ids:
                        writers[name].write(out_line)
                        counts[name] += 1
    finally:
        for w in writers.values():
            w.close()

    print(f"\nread {seen} rows total")
    for name, n in counts.items():
        print(f"  {args.output_prefix}_{name}.jsonl  →  {n} rows")


if __name__ == "__main__":
    main()
