"""The ``medsae`` command line.

Each command is a thin layer over the Python API and its help names the function it calls, so anything done
here can be scripted in Python the same way. Heavy libraries (PyTorch, pandas, sentence transformers) are
imported inside the commands, which keeps ``medsae --help`` fast.

A typical session embeds a dataset once, then trains, evaluates and interprets on the saved embeddings::

    medsae embed --dataset nfcorpus --model BAAI/bge-small-en-v1.5 --output runs/nfcorpus/embeddings
    medsae train --dataset nfcorpus --embeddings runs/nfcorpus/embeddings --k 32 \\
        --output-dir runs/nfcorpus/sae
    medsae evaluate --dataset nfcorpus --embeddings runs/nfcorpus/embeddings --sae runs/nfcorpus/sae
    medsae interpret --dataset nfcorpus --embeddings runs/nfcorpus/embeddings --sae runs/nfcorpus/sae \\
        --output runs/nfcorpus/features
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import textwrap
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from medsae import __version__

if TYPE_CHECKING:
    from medsae.data import EmbeddedDataset, RetrievalDataset
    from medsae.embed import Embedder, GeminiEmbedder
    from medsae.interpret import LLM
    from medsae.sae import MedSAE

__all__ = ["PAPER_TRAINING_VALUES", "build_parser", "main"]

logger = logging.getLogger(__name__)

PROG = "medsae"

# Hyperparameters the paper holds fixed (Appendix A), quoted in the help of `medsae train`. The InfoNCE weight
# and temperature are not among them: the paper selects them per k on validation NDCG@10 from a grid, of which
# the TrainingConfig defaults are one point. The defaults themselves live in medsae.TrainingConfig, and the
# test suite checks that both agree.
PAPER_TRAINING_VALUES: dict[str, float | int] = {
    "lr": 5e-4,
    "batch_size": 512,
    "epochs": 100,
    "alpha_q": 0.1,
    "eval_every": 100,
    "seed": 42,
}

# Errors caused by the input (a wrong path or name, a missing optional dependency, a bad value): printed on
# one line, or as a full traceback with --verbose.
_USER_ERRORS = (OSError, KeyError, ValueError, ImportError)

Subparsers = argparse._SubParsersAction  # the type argparse returns from add_subparsers


def _given(**kwargs: Any) -> dict[str, Any]:
    """The keyword arguments the user set, so that the Python API keeps ownership of every default."""
    return {key: value for key, value in kwargs.items() if value is not None}


def _require_new_folder(path: Path) -> None:
    """Refuse to save a checkpoint into a folder in use: ``save_pretrained`` never replaces an existing model
    card, which would then describe another checkpoint."""
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"{path} already exists and is not an empty folder; choose a new output folder")


def _json_value(value: Any) -> Any:
    """numpy scalars and arrays as Python numbers and lists, anything else as its string."""
    return value.tolist() if hasattr(value, "tolist") else str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=_json_value)
    path.write_text(text + "\n", encoding="utf-8")


def _print_metrics(title: str, metrics: dict[str, Any]) -> None:
    print(title)
    for name, value in metrics.items():
        print(f"  {name:<12} {value:.4f}" if isinstance(value, float) else f"  {name:<12} {value}")


# Shared loaders


def _load_dataset(name: str, cache_dir: Path | None) -> RetrievalDataset:
    """``medsae.load_dataset``: a registry name or a folder in the BEIR layout."""
    from medsae.data import load_dataset

    return load_dataset(name, **_given(cache_dir=cache_dir))


def _load_embeddings(folder: Path, dataset_name: str, cache_dir: Path | None) -> EmbeddedDataset:
    """``medsae.EmbeddedDataset.load``: embeddings saved by ``medsae embed`` for that very dataset."""
    from medsae.data import EmbeddedDataset

    return EmbeddedDataset.load(folder, _load_dataset(dataset_name, cache_dir))


def _load_sae(spec: str) -> MedSAE:
    """A folder written by ``save_pretrained`` (``MedSAE.from_pretrained``), else a Hub repository
    (``medsae.load_pretrained``)."""
    path = Path(spec).expanduser()
    if path.exists():
        from medsae.sae import MedSAE

        return MedSAE.from_pretrained(path)
    if spec.count("/") != 1 or spec.startswith((".", "/", "~")):
        raise FileNotFoundError(f"no MedSAE folder at {spec} (a Hub repository id reads organisation/name)")
    from medsae.hub import load_pretrained

    return load_pretrained(repo_id=spec)


def _embedder(
    model: str | None = None,
    sae: MedSAE | None = None,
    *,
    device: str | None = None,
    no_normalize: bool = False,
    max_seq_length: int | None = None,
    **options: Any,
) -> Embedder | GeminiEmbedder:
    """``medsae.Embedder``, or ``medsae.embed.GeminiEmbedder`` for ``gemini-embedding-001``; with an SAE, the
    encoder recorded in it (``Embedder.from_sae``). Options set to ``None`` keep the API defaults."""
    from medsae.embed import Embedder, GeminiEmbedder
    from medsae.hub import resolve_embedder

    target = sae.embedder if sae is not None else model
    gemini = bool(target) and resolve_embedder(target).lower().startswith("gemini")
    options = _given(**options)
    if not gemini:  # the Gemini API has no device, truncation length or normalization switch
        options |= _given(
            device=device, max_seq_length=max_seq_length, normalize=False if no_normalize else None
        )
    if sae is not None:
        return Embedder.from_sae(sae, **options)
    return GeminiEmbedder(target, **options) if gemini else Embedder(**_given(model=model), **options)


# datasets


def _add_datasets(commands: Subparsers, parent: argparse.ArgumentParser) -> None:
    sub = _add_command(
        commands,
        parent,
        "datasets",
        "list the benchmarks of the paper, or load one",
        "Without NAME, list the registered benchmarks with their size and license (medsae.list_datasets). "
        "With NAME, download it, print its terms of use and describe its splits (medsae.load_dataset).",
        "medsae datasets\nmedsae datasets nfcorpus",
    )
    sub.add_argument(
        "name", nargs="?", metavar="NAME", help="benchmark name or BEIR folder to load and describe"
    )
    sub.add_argument(
        "--cache-dir", type=Path, metavar="DIR", help="Hugging Face cache folder for the downloads"
    )
    sub.set_defaults(handler=_run_datasets)


def _run_datasets(args: argparse.Namespace) -> int:
    from medsae.data import get_dataset_info, list_datasets

    if args.name is not None:
        print(_load_dataset(args.name, args.cache_dir).describe())
        return 0
    rows = [("NAME", "DOCUMENTS", "QUERIES", "LICENSE", "HUB")]
    for name in list_datasets():
        info = get_dataset_info(name)
        queries = ", ".join(f"{split} {count:,}" for split, count in info.n_queries.items())
        rows.append((name, f"{info.n_documents:,}", queries, info.license, info.hf_id))
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    for row in rows:
        cells = [row[0].ljust(widths[0]), row[1].rjust(widths[1])]
        cells += [cell.ljust(width) for cell, width in zip(row[2:], widths[2:], strict=True)]
        print("  ".join(cells).rstrip())
    print("\nDatasets are downloaded on first use; `medsae datasets NAME` prints the terms of one.")
    return 0


# pretrained


def _add_pretrained(commands: Subparsers, parent: argparse.ArgumentParser) -> None:
    sub = _add_command(
        commands,
        parent,
        "pretrained",
        "list, name or download pretrained checkpoints",
        "Without options, list the MedSAE checkpoints published on the Hugging Face Hub "
        "(medsae.list_pretrained). With --embedder and --k, print the repository of that checkpoint "
        "(medsae.pretrained_repo_id); add --download to save it into a new folder (medsae.load_pretrained).",
        "medsae pretrained\nmedsae pretrained --embedder bge --k 64\n"
        "medsae pretrained --embedder bge --k 64 --download checkpoints/bge_k64",
    )
    sub.add_argument("--author", help="Hub user or organisation to list (default: doctolib-lab)")
    sub.add_argument("--embedder", help="bge, medembed, gemini or a full model id (default: bge)")
    sub.add_argument("--k", type=int, metavar="K", help="active features of the checkpoint (default: 64)")
    sub.add_argument("--repo-id", help="explicit Hub repository, instead of --embedder and --k")
    sub.add_argument("--revision", help="branch, tag or commit of the repository")
    sub.add_argument("--download", metavar="DIR", type=Path, help="save the checkpoint into this new folder")
    sub.set_defaults(handler=_run_pretrained)


def _run_pretrained(args: argparse.Namespace) -> int:
    from medsae.hub import HUB_ORG, list_pretrained, load_pretrained, pretrained_repo_id

    if args.embedder is None and args.k is None and args.repo_id is None and args.download is None:
        published = list_pretrained(**_given(author=args.author))
        print(
            "\n".join(published) or f"No MedSAE checkpoint is published under {args.author or HUB_ORG} yet."
        )
        return 0
    repo_id = args.repo_id or pretrained_repo_id(**_given(embedder=args.embedder, k=args.k))
    if args.download is None:
        print(repo_id)
        return 0
    _require_new_folder(args.download)
    sae = load_pretrained(repo_id=repo_id, **_given(revision=args.revision))
    sae.save_pretrained(args.download)
    print(f"Saved {repo_id} ({sae.extra_repr()}) to {args.download}")
    return 0


# embed


def _add_embed(commands: Subparsers, parent: argparse.ArgumentParser) -> None:
    sub = _add_command(
        commands,
        parent,
        "embed",
        "embed a dataset with a frozen text encoder",
        "Embed the corpus and the queries of every split (medsae.Embedder.embed_dataset) and save the arrays "
        "with the encoder settings (medsae.EmbeddedDataset.save). Queries of the BGE v1.5 models get the "
        "retrieval instruction of their model card, the one the paper used for BGE large, and other models "
        "none, unless --query-instruction says otherwise. gemini-embedding-001 goes through "
        "medsae.embed.GeminiEmbedder, with the Google Cloud credentials of the environment (pip install "
        '"medsae[gemini]").',
        "medsae embed --dataset nfcorpus --model BAAI/bge-small-en-v1.5 --output runs/nfcorpus/embeddings\n"
        "medsae embed --dataset r2med_biology --sae runs/nfcorpus/sae --output runs/r2med_biology/embeddings",
    )
    _add_dataset_arguments(sub)
    sub.add_argument(
        "--output", metavar="DIR", type=Path, required=True, help="folder to save the embeddings in"
    )
    encoder = sub.add_mutually_exclusive_group()
    encoder.add_argument("--model", help="model id or alias: bge, medembed, gemini (default: BGE large)")
    encoder.add_argument(
        "--sae", metavar="PATH_OR_REPO", help="use the encoder recorded in this MedSAE (Embedder.from_sae)"
    )
    sub.add_argument("--query-instruction", help="text put before every query (an empty string disables it)")
    sub.add_argument("--document-instruction", help="text put before every document")
    sub.add_argument("--splits", nargs="+", metavar="SPLIT", help="query splits to embed (default: all)")
    sub.add_argument("--batch-size", type=int, metavar="N", help="texts per forward pass or API request")
    sub.add_argument(
        "--max-seq-length", type=int, metavar="N", help="truncation length in tokens (default: the model's)"
    )
    sub.add_argument("--no-normalize", action="store_true", help="keep raw vectors instead of unit norm ones")
    sub.add_argument("--no-progress", action="store_true", help="hide the progress bars")
    _add_device_argument(sub)
    sub.set_defaults(handler=_run_embed)


def _run_embed(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args.dataset, args.cache_dir)
    embedder = _embedder(
        args.model,
        _load_sae(args.sae) if args.sae is not None else None,
        device=args.device,
        no_normalize=args.no_normalize,
        max_seq_length=args.max_seq_length,
        query_instruction=args.query_instruction,
        document_instruction=args.document_instruction,
        batch_size=args.batch_size,
        show_progress=False if args.no_progress else None,
    )
    data = embedder.embed_dataset(dataset, **_given(splits=args.splits))
    data.save(args.output)
    print(f"Saved the {data.embedder} embeddings of {dataset.name} ({data.dim} dimensions) to {args.output}")
    return 0


# train


def _add_train(commands: Subparsers, parent: argparse.ArgumentParser) -> None:
    paper = PAPER_TRAINING_VALUES
    sub = _add_command(
        commands,
        parent,
        "train",
        "train a MedSAE on embedded data",
        "Train a TopK sparse autoencoder with the reconstruction and InfoNCE objective of the paper "
        "(medsae.Trainer.fit). Training options default to the values the paper holds fixed, and to one "
        "point of its grid for the InfoNCE weight and temperature (medsae.TrainingConfig). The checkpoint "
        "with the best validation NDCG@10 is saved in --output-dir at every improvement, then scored on the "
        "test split if there is one.",
        "medsae train --dataset nfcorpus --embeddings runs/nfcorpus/embeddings --k 32 \\\n"
        "    --output-dir runs/nfcorpus/sae",
    )
    _add_dataset_arguments(sub)
    _add_embeddings_argument(sub)
    sub.add_argument(
        "--output-dir", metavar="DIR", type=Path, required=True, help="where to save the best SAE"
    )
    model = sub.add_argument_group("model (medsae.MedSAE)")
    model.add_argument(
        "--k", type=int, metavar="K", default=64, help="active features per embedding (default: %(default)s)"
    )
    width = model.add_mutually_exclusive_group()
    width.add_argument("--n-latents", type=int, metavar="M", help="number of features m")
    width.add_argument(
        "--expansion",
        type=int,
        metavar="R",
        default=3,
        help="m as a multiple of the embedding dimension (default: 3)",
    )
    model.add_argument(
        "--activation",
        choices=("relu", "none"),
        default="relu",
        help="applied after the TopK (default: relu)",
    )
    training = sub.add_argument_group("training (medsae.TrainingConfig)")
    training.add_argument(
        "--lr", type=float, metavar="LR", help=f"Adam learning rate (paper: {paper['lr']:.4f})"
    )
    training.add_argument(
        "--batch-size", type=int, metavar="N", help=f"batch size (paper: {paper['batch_size']})"
    )
    training.add_argument(
        "--epochs", type=int, metavar="N", help=f"passes over the training corpus (paper: {paper['epochs']})"
    )
    training.add_argument(
        "--lambda-nce",
        type=float,
        metavar="W",
        help="InfoNCE weight (default 0.00001; the paper selects it per k from 0, 0.000005 and 0.00001)",
    )
    training.add_argument(
        "--temperature",
        type=float,
        metavar="T",
        help="InfoNCE temperature (default 0.05; the paper selects it per k from 0.01, 0.05 and 0.1)",
    )
    training.add_argument(
        "--alpha-q",
        type=float,
        metavar="W",
        help=f"weight of the query reconstruction (paper: {paper['alpha_q']})",
    )
    training.add_argument(
        "--eval-every",
        type=int,
        metavar="N",
        help=f"steps between two validations (paper: {paper['eval_every']})",
    )
    training.add_argument(
        "--seed", type=int, metavar="SEED", help=f"random seed (paper: {paper['seed']}, also 41 and 43)"
    )
    training.add_argument(
        "--weight-decay", type=float, metavar="W", help="Adam weight decay on the encoder (paper: 0)"
    )
    training.add_argument(
        "--k-aux", type=int, metavar="N", help="dead features in the AuxK loss (paper: disabled)"
    )
    training.add_argument(
        "--aux-coef", type=float, metavar="W", help="weight of the AuxK loss (default: 1/32)"
    )
    training.add_argument(
        "--corpus-val-fraction",
        type=float,
        metavar="F",
        help="documents held out of training when the dataset has no corpus split (paper: 0.1)",
    )
    training.add_argument(
        "--compute-val-loss", action="store_true", help="also log the validation loss (monitoring only)"
    )
    training.add_argument(
        "--eval-batch-size", type=int, metavar="N", help="embeddings encoded at once during validation"
    )
    training.add_argument(
        "--train-split", default="train", metavar="SPLIT", help="query split to train on (default: train)"
    )
    validation = training.add_mutually_exclusive_group()
    validation.add_argument(
        "--val-split", default="val", metavar="SPLIT", help="split that selects the checkpoint (default: val)"
    )
    validation.add_argument(
        "--no-val", action="store_true", help="no checkpoint selection: keep the last weights"
    )
    training.add_argument("--no-progress", action="store_true", help="hide the progress bar")
    _add_device_argument(sub)
    sub.set_defaults(handler=_run_train)


def _run_train(args: argparse.Namespace) -> int:
    import torch

    from medsae.metrics import evaluate
    from medsae.sae import MedSAE
    from medsae.training import Trainer, TrainingConfig

    config = TrainingConfig(
        **_given(
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lambda_nce=args.lambda_nce,
            temperature=args.temperature,
            alpha_q=args.alpha_q,
            weight_decay=args.weight_decay,
            k_aux=args.k_aux,
            aux_coef=args.aux_coef,
            eval_every=args.eval_every,
            corpus_val_fraction=args.corpus_val_fraction,
            seed=args.seed,
            eval_batch_size=args.eval_batch_size,
            device=args.device,
        ),
        compute_val_loss=args.compute_val_loss,
        show_progress=not args.no_progress,
    )
    data = _load_embeddings(args.embeddings, args.dataset, args.cache_dir)
    torch.manual_seed(config.seed)  # initial weights: the paper code seeds torch before building the SAE
    sae = MedSAE(
        d_in=data.dim,
        n_latents=args.n_latents or args.expansion * data.dim,
        k=args.k,
        activation=args.activation,
        embedder=data.embedder,
        query_instruction=data.query_instruction,
    )
    trainer = Trainer(sae, config, output_dir=args.output_dir)
    trainer.fit(data, train_split=args.train_split, val_split=None if args.no_val else args.val_split)
    if math.isnan(trainer.best_metric):
        selected = f"weights of the last step, {trainer.best_step}"
    else:
        split = trainer.summary.get("selection_split")
        selected = f"best {split} NDCG@10 {trainer.best_metric:.4f} at step {trainer.best_step}"
    print(f"Saved the SAE ({trainer.best_model.extra_repr()}; {selected}) to {args.output_dir}")
    if "test" in data.queries and args.train_split != "test":
        metrics = evaluate(trainer.best_model, data, split="test", device=args.device)
        _print_metrics(f"{data.dataset.name} test, MedSAE sparse codes", metrics)
    return 0


# evaluate


def _add_evaluate(commands: Subparsers, parent: argparse.ArgumentParser) -> None:
    sub = _add_command(
        commands,
        parent,
        "evaluate",
        "retrieval metrics of a MedSAE or of the dense embeddings",
        "Rank the whole corpus for every query of a split by cosine similarity of the sparse codes, or of "
        "the dense embeddings without --sae, and report NDCG@10 as in the paper, recall and MRR "
        "(medsae.evaluate).",
        "medsae evaluate --dataset nfcorpus --embeddings runs/nfcorpus/embeddings\n"
        "medsae evaluate --dataset nfcorpus --embeddings runs/nfcorpus/embeddings --sae runs/nfcorpus/sae",
    )
    _add_dataset_arguments(sub)
    _add_embeddings_argument(sub)
    _add_sae_argument(sub, required=False, default="the dense embeddings")
    sub.add_argument(
        "--split", default="test", metavar="SPLIT", help="query split to score (default: %(default)s)"
    )
    sub.add_argument(
        "--cutoff", type=int, metavar="K", default=10, help="k of NDCG@k and MRR@k (default: %(default)s)"
    )
    sub.add_argument("--output", metavar="FILE", type=Path, help="also write the metrics to this JSON file")
    _add_device_argument(sub)
    sub.set_defaults(handler=_run_evaluate)


def _run_evaluate(args: argparse.Namespace) -> int:
    from medsae.metrics import evaluate

    data = _load_embeddings(args.embeddings, args.dataset, args.cache_dir)
    sae = _load_sae(args.sae) if args.sae else None
    if sae is not None and sae.d_in != data.dim:
        raise ValueError(
            f"the SAE reads {sae.d_in} dimensional embeddings but {data.embedder} gives {data.dim}: embed "
            "the dataset with the encoder of the SAE (medsae embed --sae ...)"
        )
    metrics = evaluate(sae, data, split=args.split, k=args.cutoff, device=args.device)
    codes = "MedSAE sparse codes" if sae is not None else f"dense {data.embedder} embeddings"
    _print_metrics(f"{data.dataset.name} {args.split}, {codes}", metrics)
    if args.output is not None:
        _write_json(
            args.output, {"dataset": data.dataset.name, "split": args.split, "sae": args.sae, **metrics}
        )
    return 0


# interpret


def _add_interpret(commands: Subparsers, parent: argparse.ArgumentParser) -> None:
    sub = _add_command(
        commands,
        parent,
        "interpret",
        "describe the features: top texts, TF IDF terms and LLM labels",
        "Select the features that fire on enough documents and queries (medsae.Interpreter.select_features) "
        "and write their cards (medsae.Interpreter.describe): top documents and queries, negatives, TF IDF "
        "terms and, with --llm, the topic, description and keywords of an Interpreter LLM. With --score, an "
        "Activation Prediction LLM checks the cards on held out texts (medsae.Interpreter.score), in domain "
        "or on the corpus of --score-on.",
        "medsae interpret --dataset nfcorpus --embeddings runs/nfcorpus/embeddings \\\n"
        "    --sae runs/nfcorpus/sae --output runs/nfcorpus/features --n-features 20\n"
        "medsae interpret ... --llm openai --llm-model MODEL --base-url http://localhost:8000/v1 --score",
    )
    _add_dataset_arguments(sub)
    _add_embeddings_argument(sub)
    _add_sae_argument(sub, required=True)
    sub.add_argument(
        "--output",
        metavar="DIR",
        type=Path,
        required=True,
        help="folder for cards.json, cards.md, scores.json",
    )
    sub.add_argument(
        "--split",
        default="train",
        metavar="SPLIT",
        help="query split used as evidence (default: %(default)s)",
    )
    selection = sub.add_argument_group("feature selection (medsae.Interpreter.select_features)")
    chosen = selection.add_mutually_exclusive_group()
    chosen.add_argument("--features", type=int, nargs="+", metavar="ID", help="describe these features")
    chosen.add_argument(
        "--n-features", type=int, metavar="N", help="describe the first N selected features (default: all)"
    )
    selection.add_argument(
        "--min-documents", type=int, metavar="N", help="documents a feature must fire on (paper: 5)"
    )
    selection.add_argument(
        "--min-queries", type=int, metavar="N", help="queries a feature must fire on (paper: 5)"
    )
    selection.add_argument(
        "--theme-column",
        metavar="COLUMN",
        help="corpus column of themes; order features by theme entropy, most theme specific first "
        "(source_folder for CURE, as in Table 3 of the paper: the order drives the seeded draws of --score)",
    )
    cards = sub.add_argument_group("feature cards (medsae.Interpreter.describe)")
    cards.add_argument("--n-documents", type=int, metavar="N", help="top documents per card (paper: 5)")
    cards.add_argument("--n-queries", type=int, metavar="N", help="top queries per card (paper: 5)")
    cards.add_argument(
        "--n-negatives", type=int, metavar="N", help="documents the feature does not fire on (paper: 2)"
    )
    cards.add_argument("--n-terms", type=int, metavar="N", help="TF IDF terms per card (paper: 5)")
    scoring = sub.add_argument_group("Activation Prediction (medsae.Interpreter.score)")
    scoring.add_argument("--score", action="store_true", help="score the cards (needs --llm)")
    scoring.add_argument(
        "--score-on",
        metavar="DATASET",
        help="score on the corpus of this other dataset (out of domain); --dataset itself scores in domain",
    )
    scoring.add_argument(
        "--score-on-embeddings",
        metavar="DIR",
        type=Path,
        help="embeddings of --score-on saved by `medsae embed --sae ...` (default: embed its corpus now)",
    )
    scoring.add_argument(
        "--methods",
        nargs="+",
        choices=("tfidf", "llm", "snippets"),
        help="interpretations to score (default: all)",
    )
    scoring.add_argument(
        "--seed", type=int, metavar="SEED", help="seed of the evaluation rows and pairs (paper: 42)"
    )
    _add_llm_arguments(sub)
    sub.add_argument("--no-progress", action="store_true", help="hide the progress bars")
    _add_device_argument(sub)
    sub.set_defaults(handler=_run_interpret)


def _llm(args: argparse.Namespace) -> LLM | None:
    """The LLM backend of ``--llm``; ``None`` keeps the cards to evidence and TF IDF terms."""
    if args.llm == "none":
        return None
    if args.llm == "fake":
        from medsae.interpret import FakeLLM

        logger.warning(
            "FakeLLM writes placeholder labels and scores: use it to try the pipeline, not to report."
        )
        return FakeLLM()
    if args.llm == "transformers":
        from medsae.interpret import TransformersLLM

        return TransformersLLM(**_given(model=args.llm_model, device=args.device))
    from medsae.interpret import OpenAICompatibleLLM

    if args.llm_model is None:
        raise ValueError("--llm openai needs --llm-model, the name of the model served at --base-url")
    if not (args.base_url or os.environ.get("OPENAI_BASE_URL")):
        raise ValueError(
            "--llm openai needs --base-url (or OPENAI_BASE_URL), the server the texts are sent to; for the "
            "OpenAI API, pass --base-url https://api.openai.com/v1"
        )
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    return OpenAICompatibleLLM(args.llm_model, **_given(base_url=args.base_url, api_key=api_key))


def _run_interpret(args: argparse.Namespace) -> int:
    from medsae.interpret import Interpreter

    if args.score and args.llm == "none":
        raise ValueError("--score needs an Activation Prediction LLM: set --llm")
    if args.score_on_embeddings is not None and args.score_on is None:
        raise ValueError("--score-on-embeddings needs --score-on, the dataset they embed")
    data = _load_embeddings(args.embeddings, args.dataset, args.cache_dir)
    sae = _load_sae(args.sae)
    llm = _llm(args)
    interpreter = Interpreter(
        sae, data, split=args.split, device=args.device, show_progress=not args.no_progress
    )
    features = args.features or interpreter.select_features(
        **_given(
            n=args.n_features,
            min_documents=args.min_documents,
            min_queries=args.min_queries,
            theme_column=args.theme_column,
        )
    )
    cards = interpreter.describe(
        features,
        llm=llm,
        **_given(
            n_documents=args.n_documents,
            n_queries=args.n_queries,
            n_negatives=args.n_negatives,
            n_terms=args.n_terms,
        ),
    )
    interpreter.save_cards(cards, args.output / "cards.json")
    (args.output / "cards.md").write_text(
        "\n\n".join(card.to_markdown() for card in cards) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(cards)} feature cards to {args.output / 'cards.json'} and cards.md")
    if not args.score:
        return 0
    other = None
    if args.score_on is not None and _same_dataset(args.score_on, data.dataset.name):
        logger.warning("--score-on names the --dataset itself: it is scored in domain")
    elif args.score_on is not None:
        if args.score_on_embeddings is not None:
            other = _load_embeddings(args.score_on_embeddings, args.score_on, args.cache_dir)
        else:  # the pool is the corpus only: no query to embed
            corpus_only = _load_dataset(args.score_on, args.cache_dir)
            encoder = _embedder(sae=sae, device=args.device, show_progress=not args.no_progress)
            other = encoder.embed_dataset(corpus_only, splits=[])
    report = interpreter.score(cards, llm, data=other, **_given(methods=args.methods, seed=args.seed))
    _write_json(args.output / "scores.json", report.to_dict())
    where = "out of domain" if report.out_of_domain else "in domain"
    print(f"\nActivation Prediction on {report.dataset}, {where} (details in scores.json):")
    print(report.to_frame().to_string(float_format="{:.3f}".format))
    for method, error in report.failed.items():
        print(f"{method}: not scored, {error}", file=sys.stderr)
    return 1 if report.failed else 0


def _same_dataset(name: str, dataset_name: str) -> bool:
    """True when a dataset argument names the dataset of the interpreter (registry aliases resolved)."""
    from medsae.data import resolve_dataset_name

    key = resolve_dataset_name(name) or Path(name).expanduser().name
    return key == (resolve_dataset_name(dataset_name) or dataset_name)


# explain


def _add_explain(commands: Subparsers, parent: argparse.ArgumentParser) -> None:
    sub = _add_command(
        commands,
        parent,
        "explain",
        "show which features make one passage outrank another",
        "Split the cosine similarity of the sparse codes of a query and two passages into one contribution "
        "per feature, and show the features whose contributions differ most, as in Figure 2 of the paper "
        "(medsae.Interpreter.explain). Texts of the dataset reuse their embeddings, other texts are embedded "
        "with the encoder of the SAE, and the cards.json of `medsae interpret` names the features.",
        "medsae explain --dataset nfcorpus --embeddings runs/nfcorpus/embeddings --sae runs/nfcorpus/sae \\\n"
        '    --cards runs/nfcorpus/features/cards.json --query "..." --doc-a "..." --doc-b "..."',
    )
    _add_dataset_arguments(sub)
    _add_embeddings_argument(sub)
    _add_sae_argument(sub, required=True)
    sub.add_argument("--query", required=True, help="text of the query")
    sub.add_argument("--doc-a", required=True, help="text of the first passage")
    sub.add_argument("--doc-b", required=True, help="text of the second passage")
    sub.add_argument(
        "--cards", type=Path, help="cards.json written by `medsae interpret`, to name the features"
    )
    sub.add_argument("--n-features", type=int, metavar="N", help="features to show (paper: 4)")
    sub.add_argument("--no-rank", action="store_true", help="skip ranking the passages against the corpus")
    sub.add_argument(
        "--output", metavar="FILE", type=Path, help="also write the explanation to this JSON file"
    )
    _add_device_argument(sub)
    sub.set_defaults(handler=_run_explain)


def _run_explain(args: argparse.Namespace) -> int:
    from medsae.interpret import Interpreter, load_cards

    cards = load_cards(args.cards) if args.cards is not None else None
    data = _load_embeddings(args.embeddings, args.dataset, args.cache_dir)
    sae = _load_sae(args.sae)
    # No evidence queries are needed to explain, so any dataset works, even one with a test split only.
    interpreter = Interpreter(sae, data, split=None, device=args.device, show_progress=False)
    # Texts that are not in the dataset are embedded by the encoder of the SAE, loaded only if needed.
    embedder = _embedder(sae=sae, device=args.device, show_progress=False) if sae.embedder else None
    explanation = interpreter.explain(
        args.query,
        args.doc_a,
        args.doc_b,
        embedder=embedder,
        cards=cards,
        rank=not args.no_rank,
        **_given(n_features=args.n_features),
    )
    print(explanation)
    if args.output is not None:
        _write_json(args.output, explanation.to_dict())
    return 0


# push


def _add_push(commands: Subparsers, parent: argparse.ArgumentParser) -> None:
    sub = _add_command(
        commands,
        parent,
        "push",
        "upload a MedSAE folder to the Hugging Face Hub, private by default",
        "Upload a checkpoint folder to the Hub as a private repository unless --public is given "
        "(medsae.hub.push_to_hub). The license of the model card follows the training data recorded by the "
        "trainer, and --license overrides it. Log in first with `hf auth login` or set HF_TOKEN. Only push "
        "weights trained on data you may share: the decoder and b_pre carry aggregate information about it.",
        "medsae push runs/cure/sae doctolib-lab/medsae-bge-large-en-v1.5-k64 --license cc-by-nc-4.0",
    )
    sub.add_argument("folder", type=Path, help="MedSAE folder written by `medsae train` or `medsae convert`")
    sub.add_argument(
        "repo_id", help="target repository, for example doctolib-lab/medsae-bge-large-en-v1.5-k64"
    )
    sub.add_argument("--public", action="store_true", help="make the repository public (default: private)")
    sub.add_argument("--license", help="Hub license id of the weights, for example cc-by-nc-4.0 for CURE")
    sub.add_argument("--datasets", nargs="+", metavar="HUB_ID", help="Hub ids of the training datasets")
    sub.add_argument("--commit-message", help="message of the upload commit")
    sub.set_defaults(handler=_run_push)


def _run_push(args: argparse.Namespace) -> int:
    from medsae.hub import push_to_hub
    from medsae.sae import MedSAE

    if not (args.folder / "config.json").is_file():  # never fall back to a Hub repository of that name
        raise FileNotFoundError(f"{args.folder} is not a MedSAE folder (no config.json)")
    sae = MedSAE.from_pretrained(args.folder)
    if not (args.license or sae.metadata.get("license")):
        logger.warning(
            "No license is recorded for this checkpoint, so its model card says license: other (unknown). "
            "Weights inherit the terms of their training data: pass --license (cc-by-nc-4.0 for CURE)."
        )
    url = push_to_hub(
        sae,
        args.repo_id,
        private=not args.public,
        **_given(license=args.license, datasets=args.datasets, commit_message=args.commit_message),
    )
    print(f"Pushed {args.folder} to {args.repo_id} ({'public' if args.public else 'private'}): {url}")
    return 0


# convert


def _add_convert(commands: Subparsers, parent: argparse.ArgumentParser) -> None:
    sub = _add_command(
        commands,
        parent,
        "convert",
        "turn a paper checkpoint folder into a MedSAE folder",
        "Read the sae_config.json and the checkpoint written by paper/src/training/train_sae.py "
        "(medsae.from_paper_checkpoint) and save a MedSAE folder: config.json, model.safetensors and a model "
        "card, ready for MedSAE.from_pretrained and `medsae push`.",
        "medsae convert runs/paper/bge_k64_seed42 checkpoints/bge_k64 --embedder bge",
    )
    sub.add_argument(
        "paper_folder", type=Path, help="experiment folder with sae_config.json and best_sae.pth"
    )
    sub.add_argument("output", type=Path, help="new folder for the MedSAE checkpoint")
    sub.add_argument(
        "--checkpoint", help="checkpoint file (default: best_sae.pth, the best validation NDCG@10)"
    )
    sub.add_argument("--embedder", help="encoder of the run: bge, medembed, gemini or a full model id")
    sub.add_argument("--query-instruction", help="query instruction used to embed the training queries")
    sub.set_defaults(handler=_run_convert)


def _run_convert(args: argparse.Namespace) -> int:
    from medsae.hub import from_paper_checkpoint

    _require_new_folder(args.output)
    sae = from_paper_checkpoint(
        args.paper_folder,
        **_given(
            checkpoint=args.checkpoint, embedder=args.embedder, query_instruction=args.query_instruction
        ),
    )
    sae.save_pretrained(args.output)
    print(f"Converted {args.paper_folder} ({sae.extra_repr()}) to {args.output}")
    return 0


# Parser


class _Formatter(argparse.HelpFormatter):
    """Wraps descriptions to the terminal, keeps the example lines of the epilog as written and gives option
    names a wider column. Lines never break inside a word, so Hub and license ids stay whole when copied."""

    def __init__(self, prog: str) -> None:
        super().__init__(prog, max_help_position=30)

    def _split_lines(self, text: str, width: int) -> list[str]:
        text = " ".join(text.split())
        return textwrap.wrap(text, width, break_on_hyphens=False, break_long_words=False)

    def _fill_text(self, text: str, width: int, indent: str) -> str:
        if text.startswith("examples:"):
            return "".join(indent + line for line in text.splitlines(keepends=True))
        text = " ".join(text.split())
        return textwrap.fill(
            text,
            width,
            initial_indent=indent,
            subsequent_indent=indent,
            break_on_hyphens=False,
            break_long_words=False,
        )


def _add_command(
    commands: Subparsers,
    parent: argparse.ArgumentParser,
    name: str,
    summary: str,
    description: str,
    examples: str,
) -> argparse.ArgumentParser:
    epilog = "examples:\n" + "\n".join(f"  {line}" for line in examples.splitlines())
    return commands.add_parser(
        name,
        parents=[parent],
        help=summary,
        description=description,
        epilog=epilog,
        formatter_class=_Formatter,
    )


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        metavar="NAME_OR_PATH",
        required=True,
        help="benchmark name (see `medsae datasets`) or folder in the BEIR layout (medsae.load_dataset)",
    )
    parser.add_argument(
        "--cache-dir", type=Path, metavar="DIR", help="Hugging Face cache folder for the downloads"
    )


def _add_embeddings_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--embeddings",
        metavar="DIR",
        type=Path,
        required=True,
        help="embeddings of that dataset saved by `medsae embed` (medsae.EmbeddedDataset.load)",
    )


def _add_sae_argument(parser: argparse.ArgumentParser, required: bool, default: str | None = None) -> None:
    text = "local MedSAE folder or Hub repository"
    parser.add_argument(
        "--sae",
        metavar="PATH_OR_REPO",
        required=required,
        help=f"{text} (default: {default})" if default else text,
    )


def _add_device_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device", help="torch device such as cuda, mps or cpu (default: the best available)"
    )


def _add_llm_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("LLM backend")
    group.add_argument(
        "--llm",
        choices=("none", "fake", "openai", "transformers"),
        default="none",
        help="none: evidence and TF IDF terms only (default); fake: deterministic placeholder for dry runs; "
        "openai: any OpenAI compatible server (OpenAICompatibleLLM); transformers: local model "
        "(TransformersLLM)",
    )
    group.add_argument(
        "--llm-model",
        help="served model name (openai) or Hub id (transformers, default google/medgemma-4b-it: a gated "
        "model, accept its terms on the Hub and run `hf auth login`)",
    )
    group.add_argument(
        "--base-url",
        help="URL of the OpenAI compatible server the texts are sent to, such as a vLLM or Ollama endpoint; "
        "required with --llm openai unless OPENAI_BASE_URL is set (https://api.openai.com/v1 for OpenAI)",
    )
    group.add_argument(
        "--api-key-env",
        metavar="VAR",
        default="OPENAI_API_KEY",
        help="environment variable holding the API key (default: %(default)s), so that no key is typed on "
        "the command line",
    )


def build_parser() -> argparse.ArgumentParser:
    """The ``medsae`` argument parser."""
    # Logging flags work before and after the command; SUPPRESS keeps a flag given before it from being reset.
    logging_flags = argparse.ArgumentParser(add_help=False)
    logging_flags.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="debug logs and full tracebacks",
    )
    logging_flags.add_argument(
        "-q", "--quiet", action="store_true", default=argparse.SUPPRESS, help="warnings and errors only"
    )
    parser = argparse.ArgumentParser(
        prog=PROG,
        parents=[logging_flags],
        formatter_class=_Formatter,
        description="MedSAE: sparse autoencoders for interpretable and transferable medical retrieval.",
        epilog="Run `medsae <command> --help` for the options of a command.",
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="<command>", required=True)
    for add in (
        _add_datasets,
        _add_pretrained,
        _add_embed,
        _add_train,
        _add_evaluate,
        _add_interpret,
        _add_explain,
        _add_push,
        _add_convert,
    ):
        add(commands, logging_flags)
    return parser


def _configure_logging(verbose: bool, quiet: bool) -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("medsae").setLevel(
        logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``medsae`` console script; returns the exit code."""
    args = build_parser().parse_args(argv)
    verbose = getattr(args, "verbose", False)
    _configure_logging(verbose, getattr(args, "quiet", False))
    handler: Callable[[argparse.Namespace], int] = args.handler
    try:
        return handler(args)
    except KeyboardInterrupt:
        return 130
    except _USER_ERRORS as error:
        if verbose:
            raise
        # KeyError quotes its message: show the message itself.
        message = error.args[0] if isinstance(error, KeyError) and error.args else error
        print(f"{PROG} {args.command}: error: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
