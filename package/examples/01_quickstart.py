"""Quickstart: train a DoctoSAE on NFCorpus and compare its retrieval with the dense embeddings.

The small encoder ``BAAI/bge-small-en-v1.5`` (384 dimensions) keeps the whole run, downloads included, to a
few minutes on a laptop CPU. The paper trains on CURE with ``BAAI/bge-large-en-v1.5``: see
``02_reproduce_cure.py``.

    python examples/01_quickstart.py
    python examples/01_quickstart.py --output runs/quickstart --k 32 --device cpu

Embeddings are cached in ``<output>/embeddings`` and the best SAE is saved in ``<output>/sae``, where
``03_interpret_features.py`` picks it up.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from doctosae import (
    DoctoSAE,
    EmbeddedDataset,
    Embedder,
    Interpreter,
    RetrievalDataset,
    Trainer,
    TrainingConfig,
    evaluate,
    load_dataset,
)

ENCODER = "BAAI/bge-small-en-v1.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DoctoSAE on NFCorpus with a small encoder.")
    parser.add_argument("--output", type=Path, default=Path("runs/quickstart"), help="output folder")
    parser.add_argument("--k", type=int, default=32, help="active features per embedding (default: 32)")
    parser.add_argument(
        "--expansion", type=int, default=3, help="features per embedding dimension (default: 3)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="training epochs (default: 100, as the paper)"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--device", help="cuda, mps or cpu (default: the best available)")
    return parser.parse_args()


def embed(dataset: RetrievalDataset, folder: Path, device: str | None) -> EmbeddedDataset:
    """Embed the corpus and every query split once, then reuse the saved arrays while the encoder, its
    settings and the texts stay the same."""
    embedder = Embedder(ENCODER, device=device)
    if (folder / "meta.json").exists():
        try:
            saved = EmbeddedDataset.load(folder, dataset)
        except ValueError as error:  # other texts under the same ids
            logging.warning("Embedding again: %s", error)
        else:
            if saved.embedding == embedder.config:
                return saved
            logging.warning("%s holds embeddings of other encoder settings: embedding again", folder)
    data = embedder.embed_dataset(dataset)
    data.save(folder)
    return data


def main() -> None:
    args = parse_args()
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("doctosae").setLevel(logging.INFO)  # progress of doctosae, warnings only from libraries

    # 1. NFCorpus: 3,633 documents, train, val and test queries with graded relevance judgments.
    dataset = load_dataset("nfcorpus")
    print(dataset.describe())

    # 2. Dense embeddings from a frozen encoder. Queries of the BGE v1.5 models, this small one included, get
    #    the retrieval instruction of their model card, the one the paper used for BAAI/bge-large-en-v1.5.
    data = embed(dataset, args.output / "embeddings", args.device)

    # 3. A TopK SAE with m = 3d features, k of which are active for any text.
    torch.manual_seed(args.seed)
    sae = DoctoSAE(
        d_in=data.dim,
        n_latents=args.expansion * data.dim,
        k=args.k,
        embedder=data.embedder,
        query_instruction=data.query_instruction,
    )

    # 4. The paper objective and hyperparameters. NFCorpus is small: an epoch is a handful of steps.
    config = TrainingConfig(epochs=args.epochs, seed=args.seed, device=args.device)
    trainer = Trainer(sae, config, output_dir=args.output / "sae").fit(
        data, train_split="train", val_split="val"
    )
    sae = trainer.best_model

    # 5. Retrieval of the test queries over the whole corpus: dense embeddings against sparse codes.
    dense = evaluate(None, data, split="test", device=args.device)
    sparse = evaluate(sae, data, split="test", device=args.device)
    print(f"\n{'NFCorpus test':<34}{'NDCG@10':>9}{'recall@100':>12}")
    print(f"{'dense ' + data.embedder:<34}{dense['ndcg@10']:>9.4f}{dense['recall@100']:>12.4f}")
    print(f"{f'DoctoSAE, k = {sae.k}':<34}{sparse['ndcg@10']:>9.4f}{sparse['recall@100']:>12.4f}")

    # 6. A first look at three features: the texts they fire on most and their TF IDF terms.
    interpreter = Interpreter(sae, data, split="train", device=args.device)
    for card in interpreter.describe(interpreter.select_features(n=3)):
        print("\n" + card.to_markdown(max_chars=160))
    print(f"\nSaved the SAE to {args.output / 'sae'}. Next: python examples/03_interpret_features.py")


if __name__ == "__main__":
    main()
