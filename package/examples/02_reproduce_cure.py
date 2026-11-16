"""Train the SAE setting of Tables 2 and 3 of the paper on CURE (BGE embeddings, k = 64, m = 3072) and
evaluate it as in Table 1.

For every seed, a MedSAE is trained on the 1,400 CURE training queries and 223,425 training passages with the
defaults of ``TrainingConfig`` (see below for the InfoNCE weight and temperature), the checkpoint with the
best validation NDCG@10 is kept, and its NDCG@10 is computed for the 400 test queries over all 244,600
passages. The same checkpoints can be scored out of domain. With several seeds the results are reported as in
the paper: mean and 95% confidence interval with the t quantile (n minus 1 degrees of freedom).

Paper values for this setting (Table 1, mean over seeds 41, 42 and 43): CURE 57.07, NFCorpus 31.89, MIRIAD
68.75, R2MED Bioinformatics 22.50, Biology 8.30 and Medical Sciences 22.72. The paper picks the InfoNCE weight
and temperature per k on validation NDCG@10; the defaults (10⁻⁵ and 0.05) are one point of that grid, and
``paper/scripts/sweep_cure_topk_bge.sh`` runs the whole grid.

A GPU is advised: embedding CURE encodes 244,600 passages with ``BAAI/bge-large-en-v1.5``, and every run takes
43,600 steps with a validation over the full corpus every 100 steps. The interpretation results of Table 3
come from these checkpoints: ``03_interpret_features.py`` documents that protocol.

    python examples/02_reproduce_cure.py --output runs/cure
    python examples/02_reproduce_cure.py --output runs/cure --seeds 41 42 43 --ood nfcorpus r2med_biology
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from medsae import EmbeddedDataset, Embedder, MedSAE, Trainer, TrainingConfig, evaluate, load_dataset

ENCODER = "BAAI/bge-large-en-v1.5"
K, EXPANSION = 64, 3
PAPER_NDCG = {
    "cure": 57.07,
    "nfcorpus": 31.89,
    "miriad": 68.75,
    "r2med_bioinformatics": 22.50,
    "r2med_biology": 8.30,
    "r2med_medical_sciences": 22.72,
}

logger = logging.getLogger("medsae.examples.reproduce_cure")  # shown with the INFO logs of medsae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MedSAE on CURE with BGE embeddings, k = 64, m = 3072."
    )
    parser.add_argument("--output", type=Path, default=Path("runs/cure"), help="output folder")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="training seeds (paper: 41 42 43)")
    parser.add_argument(
        "--ood",
        nargs="*",
        default=[],
        choices=sorted(set(PAPER_NDCG) - {"cure"}),
        help="out of domain datasets",
    )
    parser.add_argument(
        "--lambda-nce", type=float, help="InfoNCE weight (default: the TrainingConfig one, 0.00001)"
    )
    parser.add_argument(
        "--temperature", type=float, help="InfoNCE temperature (default: the TrainingConfig one)"
    )
    parser.add_argument("--device", help="cuda, mps or cpu (default: the best available)")
    return parser.parse_args()


def embedded(name: str, folder: Path, device: str | None, splits: list[str] | None = None) -> EmbeddedDataset:
    """A dataset with its BGE embeddings, computed and saved on first use, then reused while the encoder, its
    settings and the texts stay the same."""
    dataset = load_dataset(name)
    embedder = Embedder(ENCODER, device=device)
    if (folder / "meta.json").exists():
        try:
            saved = EmbeddedDataset.load(folder, dataset)
        except ValueError as error:  # other texts under the same ids
            logger.warning("Embedding %s again: %s", name, error)
        else:
            if saved.embedding == embedder.config:
                return saved
            logger.warning("%s holds embeddings of other encoder settings: embedding again", folder)
    data = embedder.embed_dataset(dataset, splits=splits)
    data.save(folder)
    return data


def mean_and_ci(values: list[float]) -> tuple[float, float]:
    """Mean and half width of the 95% confidence interval over seeds, with the t quantile as the paper."""
    if len(values) < 2:
        return float(np.mean(values)), float("nan")
    sem = np.std(values, ddof=1) / np.sqrt(len(values))
    return float(np.mean(values)), float(stats.t.ppf(0.975, len(values) - 1) * sem)


def main() -> None:
    args = parse_args()
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("medsae").setLevel(logging.INFO)  # progress of medsae, warnings only from libraries

    cure = embedded("cure", args.output / "embeddings" / "cure", args.device)
    ood = {
        name: embedded(name, args.output / "embeddings" / name, args.device, ["test"]) for name in args.ood
    }
    overrides = {"lambda_nce": args.lambda_nce, "temperature": args.temperature}

    results: dict[str, list[float]] = {name: [] for name in ["cure", *args.ood]}
    for seed in args.seeds:
        torch.manual_seed(seed)  # initial weights, as the paper code seeds torch before building the SAE
        sae = MedSAE(
            d_in=cure.dim,
            n_latents=EXPANSION * cure.dim,
            k=K,
            embedder=cure.embedder,
            query_instruction=cure.query_instruction,
        )
        config = TrainingConfig(
            seed=seed, device=args.device, **{k: v for k, v in overrides.items() if v is not None}
        )
        run_dir = args.output / f"bge_k{K}_m{sae.n_latents}_seed{seed}"
        best = (
            Trainer(sae, config, output_dir=run_dir)
            .fit(cure, train_split="train", val_split="val")
            .best_model
        )
        for name, data in {"cure": cure, **ood}.items():
            ndcg = 100 * evaluate(best, data, split="test", device=args.device)["ndcg@10"]
            results[name].append(ndcg)
            logger.info("seed %d, %s: NDCG@10 = %.2f", seed, name, ndcg)

    print(f"\n{'NDCG@10 (%)':<26}{'MedSAE':>16}{'paper':>9}")
    summary = {}
    for name, values in results.items():
        mean, half_width = mean_and_ci(values)
        shown = f"{mean:.2f}" if np.isnan(half_width) else f"{mean:.2f} ± {half_width:.2f}"
        print(f"{name:<26}{shown:>16}{PAPER_NDCG[name]:>9.2f}")
        summary[name] = {"seeds": args.seeds, "ndcg@10": values, "mean": mean, "ci95": half_width}
    (args.output / "results.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
