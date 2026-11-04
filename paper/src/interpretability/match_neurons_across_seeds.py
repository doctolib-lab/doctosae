"""Match two SAE seeds' neurons by the similarity of their Interpreter LLM descriptions.

Each neuron's description is embedded with a sentence encoder (default bge-large-en-v1.5, no
instruction prefix -- this is a symmetric description-to-description comparison, not retrieval).
Matches are reported two ways: greedy (each A-neuron's single closest B-neuron, many-to-one) and
one-to-one (scipy's linear_sum_assignment, maximizing total similarity, no neuron reused twice),
plus the mean similarity over all A x B pairs.

Usage:
    uv run python -m src.interpretability.match_neurons_across_seeds \
        --labels-a .../s42_h3072/interpretation/labels.json \
        --labels-b .../s41_h3072/interpretation/labels.json \
        --output seed42_vs_seed41_matches.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer

from src.interpretability.predict_neurons_binary import build_class_description

DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"


def load_descriptions(labels_file: Path) -> tuple[list[int], list[str]]:
    """(neuron ids, description strings) for every successfully-labelled neuron -- the same
    string build_class_description builds for the Activation Prediction LLM's "llm" type."""
    rows = json.load(open(labels_file))["neurons"]
    ids, descs = [], []
    for r in rows:
        if r["topic"] in ("LABEL_ERROR", "PARSE_ERROR"):
            continue
        ids.append(r["neuron"])
        descs.append(build_class_description({"neuron": r["neuron"]}, "llm", label_row=r).strip())
    return ids, descs


def match_neurons(labels_a: Path, labels_b: Path, model_name: str = DEFAULT_MODEL,
                  batch_size: int = 64) -> dict:
    """Embed both label files' descriptions and match A's neurons onto B's, greedily and
    one-to-one. Returns ids, the cosine-similarity-based matches and summary stats for each."""
    ids_a, descs_a = load_descriptions(labels_a)
    ids_b, descs_b = load_descriptions(labels_b)
    model = SentenceTransformer(model_name)
    emb_a = model.encode(descs_a, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    emb_b = model.encode(descs_b, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    S = emb_a @ emb_b.T  # normalized, so dot product is cosine similarity

    greedy_j = S.argmax(axis=1)
    greedy = [{"a": ids_a[i], "b": ids_b[int(j)], "similarity": float(S[i, j])}
             for i, j in enumerate(greedy_j)]

    row, col = linear_sum_assignment(-S)
    assignment = [{"a": ids_a[i], "b": ids_b[j], "similarity": float(S[i, j])}
                 for i, j in zip(row, col)]

    def _stats(pairs: list[dict]) -> dict:
        sims = np.array([p["similarity"] for p in pairs])
        return {"n": len(sims), "mean": float(sims.mean()), "median": float(np.median(sims)),
                "p10": float(np.percentile(sims, 10)), "p90": float(np.percentile(sims, 90)),
                "min": float(sims.min()), "max": float(sims.max())}

    return {"labels_a": str(labels_a), "labels_b": str(labels_b), "model": model_name,
            "n_neurons_a": len(ids_a), "n_neurons_b": len(ids_b),
            "greedy": {"stats": _stats(greedy), "pairs": greedy},
            "assignment": {"stats": _stats(assignment), "pairs": assignment},
            "all_pairs": {"n": int(S.size), "mean": float(S.mean())}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels-a", required=True, type=Path, help="label_neurons.py's output JSON, seed A")
    p.add_argument("--labels-b", required=True, type=Path, help="label_neurons.py's output JSON, seed B")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Sentence encoder (default: {DEFAULT_MODEL})")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    result = match_neurons(args.labels_a, args.labels_b, args.model, args.batch_size)
    print(f"{result['n_neurons_a']} vs {result['n_neurons_b']} labelled neurons")
    print(f"  greedy:     {result['greedy']['stats']}")
    print(f"  assignment: {result['assignment']['stats']}")
    print(f"  all pairs:  {result['all_pairs']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.output, "w"), indent=1)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
