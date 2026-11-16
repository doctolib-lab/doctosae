"""Embed a text column of a JSON or JSONL file with gemini-embedding-001 on Vertex AI; resumable, writes a pkl.
Default batch_size=250 fits short texts; lower it (for example to 5 or 10) for long chunk content, to stay under
the Vertex cap of 20k tokens per request.

Example: python -m src.data_prep.embed_texts --input queries.jsonl --column query
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.cli import expand_path_args, resolve_embed_column


BATCH_SIZE = 250
SLEEP_BETWEEN_BATCHES = 1.0
MODEL_NAME = "gemini-embedding-001"
# Vertex project/region with access to gemini-embedding-001.
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "europe-west4")


def embed_texts(texts: list[str], model_name: str = MODEL_NAME) -> list[np.ndarray]:
    """Embed a list of strings via the google-genai SDK; returns float32 vectors."""
    from google import genai
    client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
    result = client.models.embed_content(model=model_name, contents=texts)
    return [np.array(e.values, dtype=np.float32) for e in result.embeddings]


def embed_dataframe(df: pd.DataFrame, column: str = "query", batch_size: int = BATCH_SIZE, model_name: str = MODEL_NAME) -> pd.DataFrame:
    """Fill the ``embeddings`` column for rows where it's null; resumable across calls."""
    df = df.copy()
    if "embeddings" not in df.columns:
        df["embeddings"] = None
    todo = df.index[df["embeddings"].isnull()].tolist()
    print(f"  {len(todo)} rows need embeddings, {len(df) - len(todo)} already done")
    for start in range(0, len(todo), batch_size):
        idx = todo[start : start + batch_size]
        for i, vec in zip(idx, embed_texts(df.loc[idx, column].tolist(), model_name=model_name)):
            df.at[i, "embeddings"] = vec
        print(f"  embedded {min(start + batch_size, len(todo))}/{len(todo)}")
        if start + batch_size < len(todo):
            time.sleep(SLEEP_BETWEEN_BATCHES)
    return df


def load_json_input(path: Path) -> pd.DataFrame:
    """Load .json / .jsonl / .json.gz / .jsonl.gz into a DataFrame (no pkl support — inputs are always JSON)."""
    name = path.name.lower()
    opener = gzip.open if name.endswith(".gz") else open
    is_jsonl = ".jsonl" in name
    with opener(path, "rt", encoding="utf-8") as f:
        if is_jsonl:
            records = [json.loads(line) for line in f if line.strip()]
        else:
            records = json.load(f)
            if not isinstance(records, list):
                raise ValueError(f"{path}: top-level JSON must be a list of objects")
    return pd.DataFrame(records)


def main() -> None:
    """CLI: load a JSON/JSONL input, fill ``embeddings`` for ``--column``, write pkl to ``--output``."""
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path,
                   help="Input file: .json, .jsonl, .json.gz, or .jsonl.gz (no pickle).")
    p.add_argument("--output", type=Path, default=None,
                   help="Output pkl path. Default: same dir and basename as --input, with '.pkl' extension.")
    p.add_argument("--column", default="query",
                   help="Text column to embed (e.g. 'text' for BeIR jsonl).")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                   help="Rows per Vertex API call. Lower for long texts (e.g. chunks/content) to stay under the 20k token limit.")
    p.add_argument("--model", default=MODEL_NAME, help="Vertex AI text embedding model name.")
    p.add_argument("--prepend-col", default="title",
                   help="Column to prepend to --column before embedding (default: 'title'; "
                        "skipped silently if absent).")
    args = expand_path_args(p.parse_args())
    if args.output is None:
        stem = args.input.name
        for suffix in (".jsonl.gz", ".json.gz", ".jsonl", ".json"):
            if stem.lower().endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        args.output = args.input.parent / f"{stem}.pkl"

    print(f"Loading {args.input}")
    df = load_json_input(args.input)
    print(f"  {len(df)} rows; columns: {list(df.columns)}")
    if args.column not in df.columns:
        raise ValueError(f"--column {args.column!r} not in columns: {list(df.columns)}")

    embed_col = resolve_embed_column(df, args.column, args.prepend_col)
    df = embed_dataframe(df, column=embed_col, batch_size=args.batch_size, model_name=args.model)
    n_filled = df["embeddings"].notna().sum()
    print(f"  embeddings filled: {n_filled}/{len(df)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if str(args.output).endswith(".jsonl"):
        with args.output.open("w") as fh:
            for _, row in df.iterrows():
                fh.write(json.dumps({**{c: row[c] for c in df.columns if c != "embeddings"}, "embeddings": row["embeddings"].tolist()}, ensure_ascii=False) + "\n")
    else:
        df.to_pickle(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
