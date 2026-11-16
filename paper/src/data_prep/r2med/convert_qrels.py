"""Convert R2MED qrels.jsonl ({q_id, p_id, score}) into BeIR-style qrels.tsv (query-id, corpus-id, score)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.cli import expand_path_args


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Path to R2MED qrels.jsonl.")
    p.add_argument("--output", type=Path, default=None,
                   help="Output TSV path (default: <input dir>/qrels.tsv).")
    args = expand_path_args(p.parse_args())

    out = args.output or args.input.with_name("qrels.tsv")
    n = 0
    with args.input.open() as fi, out.open("w") as fo:
        fo.write("query-id\tcorpus-id\tscore\n")
        for line in fi:
            o = json.loads(line)
            fo.write(f"{o['q_id']}\t{o['p_id']}\t{o['score']}\n")
            n += 1
    print(f"Wrote {n} rows to {out}")


if __name__ == "__main__":
    main()
