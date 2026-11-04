"""Embed a text column of a JSON/JSONL with a HuggingFace SentenceTransformer model; resumable, writes a pkl.

Supports asymmetric models (e.g. google/embeddinggemma-300m) via encode_query/encode_document,
and instruction-based models (e.g. BAAI/bge-large-en-v1.5) via --query-instruction.

Use --mode query for queries and --mode document for corpus passages.

Example: python -m src.data_prep.embed_texts_hf_models --input queries_eval.jsonl --mode query
         python -m src.data_prep.embed_texts_hf_models --input corpus_train.jsonl --mode document
         python -m src.data_prep.embed_texts_hf_models --input queries_eval.jsonl --mode query \\
             --model BAAI/bge-large-en-v1.5 \\
             --query-instruction "Represent this sentence for searching relevant passages: "
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from src.data_prep.embed_texts import load_json_input
from src.utils.cli import expand_path_args, resolve_embed_column

BATCH_SIZE = 32
MODEL_NAME = "google/embeddinggemma-300m"


def _encode(model: SentenceTransformer, texts: list[str], mode: str,
            query_instruction: str, batch_size: int) -> np.ndarray:
    if mode == "query":
        if query_instruction:
            return model.encode([query_instruction + t for t in texts],
                                batch_size=batch_size, show_progress_bar=True,
                                normalize_embeddings=True)
        return model.encode_query(texts, batch_size=batch_size, show_progress_bar=True)
    if hasattr(model, "encode_document"):
        return model.encode_document(texts, batch_size=batch_size, show_progress_bar=True)
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        normalize_embeddings=True)


def embed_dataframe(df: pd.DataFrame, column: str, mode: str, query_instruction: str = "",
                    batch_size: int = BATCH_SIZE, model_name: str = MODEL_NAME) -> pd.DataFrame:
    """Fill the ``embeddings`` column for rows where it's null; resumable across calls."""
    df = df.copy()
    if "embeddings" not in df.columns:
        df["embeddings"] = None
    todo = df.index[df["embeddings"].isnull()].tolist()
    print(f"  {len(todo)} rows need embeddings, {len(df) - len(todo)} already done")
    if not todo:
        return df

    texts = df.loc[todo, column].tolist()
    n_gpus = torch.cuda.device_count()

    if n_gpus > 1:
        from concurrent.futures import ThreadPoolExecutor
        print(f"  using {n_gpus} GPUs")
        chunk_size = (len(texts) + n_gpus - 1) // n_gpus
        chunks = [texts[i * chunk_size:(i + 1) * chunk_size] for i in range(n_gpus)]
        chunks = [c for c in chunks if c]

        def _encode_chunk(args):
            chunk, device = args
            print(f"  loading model on {device}...", flush=True)
            m = SentenceTransformer(model_name, device=device)
            return _encode(m, chunk, mode, query_instruction, batch_size)

        with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
            vecs = np.concatenate(list(ex.map(_encode_chunk,
                                              [(c, f"cuda:{i}") for i, c in enumerate(chunks)])))
    else:
        model = SentenceTransformer(model_name)
        vecs = _encode(model, texts, mode, query_instruction, batch_size)

    for i, vec in zip(todo, vecs):
        df.at[i, "embeddings"] = vec.astype(np.float32)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path,
                   help="Input file: .json, .jsonl, .json.gz, or .jsonl.gz (no pickle).")
    p.add_argument("--output", type=Path, default=None,
                   help="Output pkl path. Default: same dir and basename as --input, with '.pkl' extension.")
    p.add_argument("--column", default="text",
                   help="Text column to embed (default: 'text' for BEIR jsonl).")
    p.add_argument("--mode", required=True, choices=["query", "document"],
                   help="Encoding mode: 'query' uses encode_query (or instruction+encode), 'document' uses encode_document (or encode).")
    p.add_argument("--query-instruction", default="",
                   help="Instruction prefix for queries (e.g. 'Represent this sentence for searching relevant passages: '). "
                        "If set, overrides encode_query and uses encode with the prefix.")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--model", default=MODEL_NAME, help="SentenceTransformer model name or path.")
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

    df = embed_dataframe(df, column=embed_col, mode=args.mode, query_instruction=args.query_instruction,
                         batch_size=args.batch_size, model_name=args.model)
    n_filled = df["embeddings"].notna().sum()
    print(f"  embeddings filled: {n_filled}/{len(df)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
