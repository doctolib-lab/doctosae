"""Training loop of the paper (``paper/src/training/train_sae.py``) as a reusable class.

Each step draws one batch from each of three loaders: training documents, which define the epoch; training
queries, reshuffled at every pass; and (query, positive document) pairs with every query at most once per
batch. The loss is::

    (N_c MSE_c + alpha_q N_q MSE_q) / (N_c + alpha_q N_q) + lambda_nce InfoNCE  (+ aux_coef AuxK)

where InfoNCE compares sparse codes and takes as negatives the other pairs' documents and the batch of
training documents, with every known positive of the anchor masked. After the backward pass, the decoder
gradient loses its component along each decoder column, Adam updates the parameters and the decoder
columns are rescaled to unit norm. Every ``eval_every`` steps and at the last step, the validation
queries are scored against the full corpus and the weights with the strictly highest NDCG@10 are kept; with
an output folder they are also written there at every improvement, as the paper writes ``best_sae.pth``.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Set
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from doctosae._ids import embedding_differences, embedding_settings, resolve_embedder, shareable_model_id
from doctosae.data.dataset import EmbeddedDataset, RetrievalDataset
from doctosae.metrics import Judgments, pool_ndcg
from doctosae.retrieval import encode_corpus
from doctosae.sae import DoctoSAE, _jsonable
from doctosae.training.config import TrainingConfig
from doctosae.training.losses import (
    info_nce_loss,
    positive_mask,
    reconstruction_loss,
    squared_error,
    weighted_loss,
)
from doctosae.training.sampling import (
    PairBatch,
    PairBatchSampler,
    ShuffledBatches,
    iter_forever,
    positive_pairs,
    positive_sets,
)

__all__ = ["SELECTION_METRIC", "Trainer", "train_step", "build_optimizer", "resolve_device"]

logger = logging.getLogger(__name__)

#: Name of the checkpoint selection metric in ``history``, ``summary`` and the saved metadata.
SELECTION_METRIC = "val_ndcg@10"

# Metadata keys that describe one training of the weights; fine tuning moves them into ``parents``.
_TRAINING_KEYS = (
    "train_dataset",
    "datasets",
    "license",
    "license_name",
    "license_link",
    "terms",
    "embedder",
    "training",
    "best_step",
    f"best_{SELECTION_METRIC}",
    "paper_config",
    "paper_checkpoint",
)


def resolve_device(device: torch.device | str | None = None) -> torch.device:
    """``device`` itself, or else the first available of CUDA, Apple MPS and CPU."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(sae: DoctoSAE, lr: float, weight_decay: float = 0.0) -> torch.optim.Adam:
    """Adam with weight decay on the encoder only (``build_optimizer`` of the paper code)."""
    no_decay = {id(sae.decoder.weight), id(sae.b_pre)}
    groups = [
        {"params": [p for p in sae.parameters() if id(p) not in no_decay], "weight_decay": weight_decay},
        {"params": [p for p in sae.parameters() if id(p) in no_decay], "weight_decay": 0.0},
    ]
    return torch.optim.Adam(groups, lr=lr)


def train_step(
    sae: DoctoSAE,
    optimizer: torch.optim.Optimizer,
    corpus: torch.Tensor,
    corpus_rows: np.ndarray,
    queries: torch.Tensor,
    pairs: PairBatch | None,
    positives: Mapping[int, Set[int]] | None,
    config: TrainingConfig,
    dead: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """One optimisation step of the paper (``train_step`` of ``paper/src/training/train_sae.py``).

    Args:
        corpus: Batch of training document embeddings, reconstructed and used as InfoNCE negatives.
        corpus_rows: Their corpus rows, to mask the anchors' known positives among those negatives.
        queries: Batch of training query embeddings, reconstructed.
        pairs: Batch of (query, positive document) pairs for InfoNCE, or None when ``lambda_nce`` is 0.
        positives: Query row to the corpus rows of its positives, the masks of both InfoNCE blocks.
        dead: Boolean mask of the features silent during the previous epoch, for the auxiliary loss.

    Returns detached statistics: the loss terms, ``l0`` (mean active features per reconstructed row),
    ``negatives`` (mean unmasked InfoNCE negatives per anchor) and ``fired``, the features active in this
    step.
    """
    optimizer.zero_grad()
    corpus_hat, corpus_codes = sae(corpus)
    queries_hat, query_codes = sae(queries)
    mse_c = reconstruction_loss(corpus, corpus_hat)
    mse_q = reconstruction_loss(queries, queries_hat)
    n_corpus, n_queries = corpus.shape[0], queries.shape[0]
    loss = weighted_loss(mse_c, mse_q, n_corpus, n_queries, config.alpha_q)
    stats = {"mse_c": mse_c, "mse_q": mse_q}
    fired = (corpus_codes != 0).any(dim=0) | (query_codes != 0).any(dim=0)

    if pairs is not None:
        _, anchor_codes = sae(pairs.queries)
        _, doc_codes = sae(pairs.documents)
        positives = positives or {}
        in_batch = positive_mask(
            pairs.query_rows, pairs.doc_rows, positives, corpus.device, skip_diagonal=True
        )
        in_corpus = positive_mask(pairs.query_rows, corpus_rows, positives, corpus.device)
        nce = info_nce_loss(anchor_codes, doc_codes, corpus_codes, in_batch, in_corpus, config.temperature)
        loss = loss + config.lambda_nce * nce
        unmasked = (in_batch.shape[1] - in_batch.sum(dim=1) - 1) + (in_corpus.shape[1] - in_corpus.sum(dim=1))
        stats |= {"nce": nce, "negatives": unmasked.float().mean()}
        fired |= (anchor_codes != 0).any(dim=0) | (doc_codes != 0).any(dim=0)

    if config.k_aux:
        aux = torch.zeros((), device=corpus.device)
        if dead is not None and dead.any():
            # Dead features regress the detached residual of the main reconstruction (Gao et al., 2024).
            aux_c = squared_error((corpus - corpus_hat).detach(), sae.aux_recon(corpus, dead, config.k_aux))
            aux_q = squared_error(
                (queries - queries_hat).detach(), sae.aux_recon(queries, dead, config.k_aux)
            )
            aux = weighted_loss(aux_c, aux_q, n_corpus, n_queries, config.alpha_q)
            loss = loss + config.aux_coef * aux
        stats["aux"] = aux

    loss.backward()
    sae.project_decoder_gradient()
    optimizer.step()
    sae.normalize_decoder()
    stats["l0"] = (torch.cat([corpus_codes, query_codes]) != 0).sum(dim=1).float().mean()
    return {name: value.detach() for name, value in stats.items()} | {"fired": fired}


class Trainer:
    """Train a :class:`~doctosae.DoctoSAE` with the objective, schedule and checkpoint selection of the paper.

    Example::

        torch.manual_seed(42)
        sae = DoctoSAE(d_in=data.dim, n_latents=3 * data.dim, k=64)
        trainer = Trainer(sae, TrainingConfig(), output_dir="runs/cure_bge_k64").fit(data)
        evaluate(trainer.best_model, data, split="test")

    After :meth:`fit`, ``sae`` holds the weights with the best validation NDCG@10 and ``best_model`` is
    that same object; ``history`` has one row per evaluation, ``best_metric`` and ``best_step`` describe
    the selected weights and ``summary`` the run. With an ``output_dir``, the model is written there with
    ``save_pretrained`` (``config.json``, ``model.safetensors``, ``README.md``) next to ``history.json``
    and ``summary.json``, at every improvement of the validation NDCG@10 and at the end. If training stops
    early (an error, an interrupt), the best weights so far are loaded into ``sae``, kept as ``best_model``
    and on disk, ``summary["completed"]`` is false, and the error is raised again.
    """

    def __init__(
        self, sae: DoctoSAE, config: TrainingConfig | None = None, output_dir: str | Path | None = None
    ) -> None:
        self.sae = sae
        self.config = config or TrainingConfig()
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.history: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {}
        self.best_metric = math.nan
        self.best_step: int | None = None
        self.best_model: DoctoSAE | None = None

    def fit(
        self, data: EmbeddedDataset, train_split: str = "train", val_split: str | None = "val"
    ) -> Trainer:
        """Train on the ``train_split`` queries and select the checkpoint on the ``val_split`` queries.

        Only the documents of the corpus split ``"train"`` enter the loss terms; the other documents are
        held out (the paper split, built with :func:`doctosae.data.splits.split_corpus` when the dataset
        has none). Validation always scores against the whole corpus. Without usable ``val_split``
        queries the checkpoint is selected on the training queries, with a warning, and
        ``val_split=None`` keeps the weights of the last step.

        A new SAE (``b_pre`` still zero) starts with ``b_pre`` at the geometric median of the training
        documents and queries, as in the paper; a trained one keeps its ``b_pre`` and is fine tuned.
        """
        cfg = self.config
        device = resolve_device(cfg.device)
        sae = self.sae.to(device)
        _match_embedder(sae, data)
        run = _Run.prepare(data, train_split, val_split, cfg)
        # Metadata fixed for the whole run, written at every save: provenance and the shareable encoder id.
        lineage = _lineage(sae, data.dataset) | {"embedder": _recorded_embedder(data.embedder)}

        if not sae.b_pre.detach().any():
            # Training rows in corpus order, then training queries: the median's subsample matches the paper.
            # The concatenation is passed directly so that it is freed before the first step.
            sae.init_b_pre(
                np.concatenate([data.corpus[run.train_rows], data.queries[train_split]]), seed=cfg.seed
            )
        optimizer = build_optimizer(sae, cfg.lr, cfg.weight_decay)
        generator = torch.Generator().manual_seed(cfg.seed)
        document_batches = ShuffledBatches(len(run.train_rows), run.batch_sizes["documents"], generator)
        query_batches = iter_forever(ShuffledBatches(len(run.queries), run.batch_sizes["queries"], generator))
        pair_batches = positives = None
        if run.pairs is not None:
            pair_batches = iter_forever(PairBatchSampler(run.pairs, run.batch_sizes["pairs"], seed=cfg.seed))
            positives = positive_sets(run.pairs)
        total_steps = cfg.epochs * len(document_batches)
        logger.info("Training for %d steps on %s: %s", total_steps, device, run.counts())

        self.history = []
        self.best_metric, self.best_step, self.best_model = math.nan, None, None
        best_value, best_state = -math.inf, None
        fired = torch.zeros(sae.n_latents, dtype=torch.bool, device=device)
        dead: torch.Tensor | None = None
        window = _RunningMeans()
        started = time.perf_counter()

        def describe(step: int, completed: bool) -> None:
            seconds = time.perf_counter() - started
            self._describe(sae, data, run, lineage, total_steps, len(document_batches), device, seconds)
            self.summary |= {"completed": completed, "last_step": step}

        step = 0
        sae.train()
        try:
            with tqdm(total=total_steps, desc="training", unit="step", disable=not cfg.show_progress) as bar:
                for epoch in range(1, cfg.epochs + 1):
                    for batch in document_batches:
                        step += 1
                        rows = run.train_rows[batch.numpy()]
                        pairs = None if pair_batches is None else run.pair_batch(next(pair_batches), device)
                        queries = run.queries[next(query_batches)].to(device)
                        corpus = run.corpus[torch.from_numpy(rows)].to(device)
                        stats = train_step(sae, optimizer, corpus, rows, queries, pairs, positives, cfg, dead)
                        fired |= stats.pop("fired")
                        window.add(stats)
                        bar.update()
                        if step % cfg.eval_every and step != total_steps:
                            continue
                        record = self._evaluate(sae, data, run, step, epoch, window, fired)
                        record["seconds"] = round(time.perf_counter() - started, 3)
                        self.history.append(record)
                        value = record[SELECTION_METRIC]
                        # Strictly better: the first best step wins.
                        if value is not None and value > best_value:
                            best_value, best_state = value, _cpu_state(sae)
                            self.best_metric, self.best_step = value, step
                            if self.output_dir is not None:
                                # As the paper writes best_sae.pth: a run that stops later keeps them.
                                describe(step, completed=False)
                                self._save(sae)
                        bar.set_postfix(mse_c=f"{record['mse_c']:.4g}", ndcg=_format(value))
                    # Features silent for the whole epoch feed the auxiliary loss of the next one.
                    dead = None if bool(fired.all()) else ~fired
                    fired = torch.zeros_like(fired)
        except BaseException:
            self._keep_best(sae, best_state, lambda: describe(step, completed=False))
            raise

        if best_state is not None:
            sae.load_state_dict(best_state)
        else:
            self.best_metric, self.best_step = math.nan, total_steps
        sae.eval()
        self.best_model = sae
        describe(step, completed=True)
        if self.output_dir is not None:
            self._save(sae)
        return self

    def _keep_best(self, sae: DoctoSAE, best_state: dict[str, torch.Tensor] | None, describe: Any) -> None:
        """After an error or an interrupt: load the best weights so far and mark the saved run incomplete.

        The weights themselves were written at their evaluation; only the summary and history are updated
        here. Nothing raised here may hide the error that stopped training.
        """
        if best_state is None:
            return
        try:
            sae.load_state_dict(best_state)
            sae.eval()
            self.best_model = sae
            describe()
            if self.output_dir is not None:
                self._write_json("history.json", self.history)
                self._write_json("summary.json", self.summary)
            logger.warning(
                "Training stopped early: the weights of step %s (%s %.4f) are kept%s.",
                self.best_step,
                SELECTION_METRIC,
                self.best_metric,
                f" in {self.output_dir}" if self.output_dir is not None else "",
            )
        except Exception:
            logger.exception("Could not keep the best weights after training stopped")

    def _evaluate(
        self,
        sae: DoctoSAE,
        data: EmbeddedDataset,
        run: _Run,
        step: int,
        epoch: int,
        window: _RunningMeans,
        fired: torch.Tensor,
    ) -> dict[str, Any]:
        """History row of one evaluation: training statistics since the previous row, validation metrics."""
        record: dict[str, Any] = {"step": step, "epoch": epoch, "nce": None, "negatives": None, "aux": None}
        record |= window.pop_means()
        record["dead_features"] = int((~fired).sum())  # silent since the start of the current epoch
        record[SELECTION_METRIC] = None
        if run.selection is not None:
            split, judgments = run.selection
            corpus_codes = encode_corpus(sae, data.corpus, batch_size=self.config.eval_batch_size)
            query_codes = encode_corpus(sae, data.queries[split], batch_size=self.config.eval_batch_size)
            record[SELECTION_METRIC] = pool_ndcg(query_codes, corpus_codes, judgments, k=10)
        if run.val_loss is not None:
            record |= run.val_loss(sae, run.corpus)
        logger.info(
            "step %d (epoch %d): mse_c %.5f, mse_q %.5f, nce %s, l0 %.1f, dead %d, %s %s",
            step,
            epoch,
            record["mse_c"],
            record["mse_q"],
            _format(record["nce"]),
            record["l0"],
            record["dead_features"],
            SELECTION_METRIC,
            _format(record[SELECTION_METRIC]),
        )
        return record

    def _describe(
        self,
        sae: DoctoSAE,
        data: EmbeddedDataset,
        run: _Run,
        lineage: dict[str, Any],
        total_steps: int,
        steps_per_epoch: int,
        device: torch.device,
        seconds: float,
    ) -> None:
        """Fill ``summary`` and the metadata saved with the model (JSON values only, no paths)."""
        dataset, cfg = data.dataset, self.config
        best = None if math.isnan(self.best_metric) else float(self.best_metric)
        selection_split = run.selection[0] if run.selection is not None else None
        embedder = lineage["embedder"]
        sae.metadata.update(
            {
                **lineage,
                "train_dataset": dataset.name,
                "training": {
                    **cfg.hyperparameters(),
                    "train_split": run.train_split,
                    "selection_split": selection_split,
                    "steps": total_steps,
                    "batch_sizes": run.batch_sizes,
                },
                "best_step": self.best_step,
                f"best_{SELECTION_METRIC}": best,
            }
        )
        for key in ("license_name", "license_link", "terms", "paper_config", "paper_checkpoint"):
            if sae.metadata.get(key) is None:
                sae.metadata.pop(key, None)
        if data.embedding is not None:  # the encoder settings, so that Embedder.from_sae rebuilds them
            sae.metadata["embedding"] = {
                key: value
                for key, value in data.embedding.items()
                if key not in ("model", "query_instruction")
            }
        self.summary = {
            "dataset": dataset.name,
            "embedder": embedder,
            "train_split": run.train_split,
            "selection_split": selection_split,
            "best_step": self.best_step,
            f"best_{SELECTION_METRIC}": best,
            "steps": total_steps,
            "steps_per_epoch": steps_per_epoch,
            "batch_sizes": run.batch_sizes,
            **run.counts(),
            "model": {"d_in": sae.d_in, "n_latents": sae.n_latents, "k": sae.k, "activation": sae.activation},
            "config": cfg.hyperparameters(),
            "device": device.type,
            "seconds": round(seconds, 1),
        }

    def _write_json(self, name: str, payload: Any) -> None:
        (self.output_dir / name).write_text(json.dumps(_jsonable(payload), indent=2) + "\n")

    def _save(self, sae: DoctoSAE) -> None:
        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        # save_pretrained never overwrites an existing model card, whose license must follow the new weights.
        (output_dir / "README.md").unlink(missing_ok=True)
        sae.save_pretrained(output_dir)
        self._write_json("history.json", self.history)
        self._write_json("summary.json", self.summary)
        logger.info("Saved the weights of step %s to %s", self.best_step, output_dir)


@dataclass
class _Run:
    """What :meth:`Trainer.fit` derives from the data before the first step."""

    corpus: torch.Tensor  # every document embedding, on the CPU
    queries: torch.Tensor  # training query embeddings, on the CPU
    train_split: str
    train_rows: np.ndarray  # corpus rows trained on, in corpus order
    held_out_rows: np.ndarray
    pairs: np.ndarray | None  # (training query row, corpus row) positive pairs, None without InfoNCE
    batch_sizes: dict[str, int]
    selection: tuple[str, Judgments] | None  # split and judgments of checkpoint selection
    val_loss: _ValidationLoss | None

    @classmethod
    def prepare(
        cls, data: EmbeddedDataset, train_split: str, val_split: str | None, cfg: TrainingConfig
    ) -> _Run:
        if train_split not in data.queries:
            raise ValueError(f"no {train_split!r} query embeddings; embedded splits: {sorted(data.queries)}")
        dataset = data.dataset
        train_rows, held_out_rows = _corpus_split(dataset, train_split, cfg.corpus_val_fraction)
        queries = _as_tensor(data.queries[train_split])
        pairs = _training_pairs(dataset, train_split, train_rows) if cfg.lambda_nce > 0 else None
        selection = _selection(data, train_split, val_split)
        val_loss = None
        if cfg.compute_val_loss:
            if selection is not None and selection[0] != train_split:
                val_loss = _ValidationLoss(data, selection[0], held_out_rows, train_rows, cfg)
            else:
                logger.warning("compute_val_loss needs validation queries: the validation loss is skipped.")
        return cls(
            corpus=_as_tensor(data.corpus),
            queries=queries,
            train_split=train_split,
            train_rows=train_rows,
            held_out_rows=held_out_rows,
            pairs=pairs,
            batch_sizes=_batch_sizes(cfg.batch_size, len(train_rows), len(queries), pairs),
            selection=selection,
            val_loss=val_loss,
        )

    def pair_batch(self, indices: list[int], device: torch.device) -> PairBatch:
        return _pair_batch(indices, self.pairs, self.queries, self.corpus, device)

    def counts(self) -> dict[str, int]:
        return {
            "train_documents": len(self.train_rows),
            "held_out_documents": len(self.held_out_rows),
            "train_queries": len(self.queries),
            "train_pairs": len(self.pairs) if self.pairs is not None else 0,
        }


class _RunningMeans:
    """Sums of per step statistics, kept on the device so that steps never wait for a host sync."""

    def __init__(self) -> None:
        self._sums: dict[str, torch.Tensor] = {}
        self._count = 0

    def add(self, stats: Mapping[str, torch.Tensor]) -> None:
        for name, value in stats.items():
            self._sums[name] = self._sums[name] + value if name in self._sums else value.clone()
        self._count += 1

    def pop_means(self) -> dict[str, float]:
        means = {name: float(total) / self._count for name, total in self._sums.items()}
        self._sums, self._count = {}, 0
        return means


class _ValidationLoss:
    """The validation loss of the paper (``val_step``), for monitoring: it never selects the checkpoint.

    Mirrors the training loss on validation data: the evaluation corpus (held out documents first, then
    training documents) in fixed order as reconstruction batches and InfoNCE negatives, the validation
    queries in fixed order, and validation pairs from a sampler whose random state carries over between
    evaluations. Document reconstruction is measured on held out documents only, and batches are combined
    with the paper's weights.
    """

    def __init__(
        self,
        data: EmbeddedDataset,
        split: str,
        held_out_rows: np.ndarray,
        train_rows: np.ndarray,
        config: TrainingConfig,
    ) -> None:
        dataset = data.dataset
        self.config = config
        self.order = np.concatenate([held_out_rows, train_rows])
        self.held_out = np.zeros(len(dataset.corpus), dtype=bool)
        self.held_out[held_out_rows] = True
        self.queries = _as_tensor(data.queries[split])
        self.pairs: np.ndarray | None = None
        self.positives: dict[int, frozenset[int]] = {}
        self.sampler: PairBatchSampler | None = None
        if config.lambda_nce > 0 and split in dataset.qrels:
            query_rows = {
                query_id: row for row, query_id in enumerate(dataset.queries[split]["query_id"].tolist())
            }
            pairs = positive_pairs(
                dataset.qrels[split], query_rows, dataset.doc_index, dataset.positive_threshold
            )
            if len(pairs):
                self.pairs, self.positives = pairs, positive_sets(pairs)
                self.sampler = PairBatchSampler(pairs, config.batch_size, seed=config.seed, drop_last=False)

    @torch.no_grad()
    def __call__(self, sae: DoctoSAE, corpus: torch.Tensor) -> dict[str, float | None]:
        cfg, device = self.config, sae.device
        totals: dict[str, float] = defaultdict(float)
        query_batches = iter_forever(torch.arange(len(self.queries)).split(cfg.batch_size))
        pair_batches = iter_forever(self.sampler) if self.sampler is not None else None
        for start in range(0, len(self.order), cfg.batch_size):
            rows = self.order[start : start + cfg.batch_size]
            x_corpus = corpus[torch.from_numpy(rows)].to(device)
            x_queries = self.queries[next(query_batches)].to(device)
            corpus_hat, corpus_codes = sae(x_corpus)
            queries_hat, _ = sae(x_queries)
            if pair_batches is not None:
                pairs = _pair_batch(next(pair_batches), self.pairs, self.queries, corpus, device)
                masks = (
                    positive_mask(
                        pairs.query_rows, pairs.doc_rows, self.positives, device, skip_diagonal=True
                    ),
                    positive_mask(pairs.query_rows, rows, self.positives, device),
                )
                anchor_codes, doc_codes = sae(pairs.queries)[1], sae(pairs.documents)[1]
                nce = info_nce_loss(anchor_codes, doc_codes, corpus_codes, *masks, cfg.temperature)
                totals["nce"] += nce.item() * len(pairs.query_rows)
                totals["n_nce"] += len(pairs.query_rows)
            mse_q = reconstruction_loss(x_queries, queries_hat)
            n_q = len(x_queries)
            totals["mse_q"] += mse_q.item() * n_q
            totals["n_q"] += n_q
            keep = torch.from_numpy(self.held_out[rows]).to(device)
            n_c = int(keep.sum())
            if n_c > 0:
                mse_c = reconstruction_loss(x_corpus[keep], corpus_hat[keep])
                totals["mse_c"] += mse_c.item() * n_c
                totals["n_c"] += n_c
                totals["mse"] += weighted_loss(mse_c, mse_q, n_c, n_q, cfg.alpha_q).item() * (n_c + n_q)
                totals["n_mse"] += n_c + n_q
            elif cfg.alpha_q > 0:
                totals["mse"] += mse_q.item() * n_q
                totals["n_mse"] += n_q
        nce = totals["nce"] / totals["n_nce"] if totals["n_nce"] else None
        return {
            "val_loss": totals["mse"] / max(totals["n_mse"], 1) + cfg.lambda_nce * (nce or 0.0),
            "val_mse_c": totals["mse_c"] / max(totals["n_c"], 1),
            "val_mse_q": totals["mse_q"] / max(totals["n_q"], 1),
            "val_nce": nce,
        }


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _cpu_state(sae: DoctoSAE) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().to("cpu", copy=True) for name, tensor in sae.state_dict().items()}


def _as_tensor(array: np.ndarray) -> torch.Tensor:
    """Float32 tensor sharing memory with ``array``, copied only when read only or not contiguous."""
    return torch.from_numpy(np.require(array, dtype=np.float32, requirements=["C", "W"]))


def _match_embedder(sae: DoctoSAE, data: EmbeddedDataset) -> None:
    """Check that the SAE fits the embeddings, and record their embedder and query instruction on it.

    A ``None`` query instruction is unknown: the SAE adopts the one of the data, and data of unknown
    instruction are not checked. Other recorded settings (revision, document instruction, normalization,
    truncation) must agree too, so that a fine tuning never mixes embeddings computed differently.
    """
    if data.dim != sae.d_in:
        raise ValueError(
            f"the SAE takes {sae.d_in} dimensional embeddings, the data has {data.dim} dimensions"
        )
    if sae.embedder is None:
        sae.embedder = data.embedder
    elif data.embedder and resolve_embedder(sae.embedder) != resolve_embedder(data.embedder):
        raise ValueError(
            f"the SAE is declared for {sae.embedder} embeddings but the data comes from {data.embedder}"
        )
    if sae.query_instruction is None:
        sae.query_instruction = data.query_instruction
    elif data.query_instruction is not None and data.query_instruction != sae.query_instruction:
        raise ValueError(
            f"the SAE expects the query instruction {sae.query_instruction!r} but the queries were embedded "
            f"with {data.query_instruction!r}"
        )
    differences = embedding_differences(
        embedding_settings(sae.embedder, sae.query_instruction, sae.metadata.get("embedding")),
        embedding_settings(data.embedder, data.query_instruction, data.embedding),
    )
    if differences:
        raise ValueError(
            "the SAE was trained on embeddings computed with other settings than these: "
            + "; ".join(differences)
        )


def _recorded_embedder(embedder: str | None) -> str | None:
    """The embedder as the metadata records it: a Hub id or model name, never a local path."""
    shared = shareable_model_id(embedder)
    if shared != embedder:
        logger.warning(
            "The embeddings come from a local model folder: the metadata records only its name, %r. Embed "
            'with Embedder(folder, name="org/model") to record the Hub id of the model instead.',
            shared,
        )
    return shared


def _lineage(sae: DoctoSAE, dataset: RetrievalDataset) -> dict[str, Any]:
    """Data provenance of the weights after this training: the datasets, their license and their terms.

    Fine tuning trained weights keeps where they came from: their earlier training moves into ``parents``,
    the Hub ids of every dataset are kept, and a license is never replaced silently. When the licenses differ
    the card says ``license: other`` with a name joining both ids, and a warning asks for a review; when the
    earlier license is unknown, so is the new one.
    """
    if dataset.license is None:
        logger.warning(
            "%s declares no license: set sae.metadata['license'] to the terms of the training data "
            "before sharing the checkpoint.",
            dataset.name,
        )
    # Keys set to None are removed from the metadata: they described the earlier training only.
    current = {
        "datasets": list(dataset.hf_ids),
        "license": dataset.license,
        "license_name": None,
        "license_link": None,
        "terms": _registry_terms(dataset),
    }
    if not sae.is_trained:
        return current
    previous = {key: sae.metadata[key] for key in _TRAINING_KEYS if key in sae.metadata}
    license, license_name = _merged_license(
        previous.get("license"), previous.get("license_name"), dataset.license
    )
    if (license, license_name) != (dataset.license, None):
        logger.warning(
            "Fine tuning weights trained on %s (license %s) with %s (license %s): the model card license is "
            "%s. Review the terms of every training dataset before sharing the checkpoint.",
            previous.get("train_dataset", "unrecorded data"),
            previous.get("license_name") or previous.get("license") or "unknown",
            dataset.name,
            dataset.license or "unknown",
            license_name or license or "unknown",
        )
    keeps_link = license == previous.get("license") and license_name == previous.get("license_name")
    return current | {
        "parents": [*sae.metadata.get("parents", []), previous],
        "datasets": list(dict.fromkeys([*previous.get("datasets", []), *dataset.hf_ids])),
        "license": license,
        "license_name": license_name,
        "license_link": previous.get("license_link") if keeps_link else None,
        "paper_config": None,
        "paper_checkpoint": None,
    }


def _merged_license(
    previous: str | None, previous_name: str | None, new: str | None
) -> tuple[str | None, str | None]:
    """``(license, license_name)`` of weights trained on data under ``previous`` then under ``new``."""
    if not previous:
        return None, None  # the terms of the earlier data are unknown, so are those of the weights
    ids = (previous_name or "unknown").split("-and-") if previous == "other" else [previous]
    if not new or new in ids:
        return previous, previous_name
    return "other", "-and-".join([*ids, new])


def _registry_terms(dataset: RetrievalDataset) -> str | None:
    """Usage terms of a benchmark loaded from the registry (same name and Hub ids), else None."""
    from doctosae.data.registry import DATASETS, resolve_dataset_name

    key = resolve_dataset_name(dataset.name)
    if key is None or list(DATASETS[key].hf_ids) != list(dataset.hf_ids):
        return None
    return DATASETS[key].terms


def _corpus_split(
    dataset: RetrievalDataset, train_split: str, val_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Corpus rows trained on, sorted in corpus order like the paper's split files, and the held out rows.

    Without a corpus split on the dataset, the positives of the training queries are forced into training and
    ``val_fraction`` of the other documents are held out, as ``split_corpus.py --force-train-positives``.
    """
    n_docs = len(dataset.corpus)
    splits = dataset.corpus_splits
    if splits is None and val_fraction > 0:
        # Imported here: only datasets without a corpus split of their own need the paper's splitting code.
        from doctosae.data.splits import split_corpus

        splits = split_corpus(dataset, val_frac=val_fraction, train_split=train_split)
    if splits is None:
        train = np.arange(n_docs)
    elif "train" not in splits:
        raise ValueError(f"{dataset.name}.corpus_splits has no 'train' entry (found {sorted(splits)})")
    else:
        train = np.unique(np.asarray(splits["train"], dtype=np.int64))
    if train.size == 0 or train[0] < 0 or train[-1] >= n_docs:
        raise ValueError(f"the training corpus split must be a non empty set of rows in [0, {n_docs})")
    return train, np.setdiff1d(np.arange(n_docs), train, assume_unique=True)


def _training_pairs(dataset: RetrievalDataset, split: str, train_rows: np.ndarray) -> np.ndarray:
    """(query row, corpus row) of every positive judgment of ``split`` whose document is trained on."""
    if split not in dataset.qrels:
        raise ValueError(
            f"{dataset.name} has no {split!r} qrels for the InfoNCE pairs; set lambda_nce=0 to train on "
            "reconstruction only"
        )
    query_rows = {query_id: row for row, query_id in enumerate(dataset.queries[split]["query_id"].tolist())}
    doc_ids = dataset.corpus["doc_id"].to_numpy()
    doc_rows = dict(zip(doc_ids[train_rows].tolist(), train_rows.tolist(), strict=True))
    pairs = positive_pairs(dataset.qrels[split], query_rows, doc_rows, dataset.positive_threshold)
    if len(pairs) == 0:
        raise ValueError(
            f"no positive (query, document) pair among the {split!r} qrels and the training documents; set "
            "lambda_nce=0 to train on reconstruction only"
        )
    return pairs


def _batch_sizes(batch_size: int, n_docs: int, n_queries: int, pairs: np.ndarray | None) -> dict[str, int]:
    """Batch size of each loader: ``batch_size``, or less when a loader has fewer rows (tiny datasets)."""
    available = {"documents": n_docs, "queries": n_queries}
    if pairs is not None:
        available["pairs"] = len(np.unique(pairs[:, 0]))  # each query appears at most once per pair batch
    empty = [name for name, size in available.items() if size == 0]
    if empty:
        raise ValueError(f"no training {' or '.join(empty)}")
    sizes = {name: min(batch_size, size) for name, size in available.items()}
    clamped = {name: size for name, size in sizes.items() if size < batch_size}
    if clamped:
        logger.warning(
            "batch_size=%d exceeds the training data; smaller batches are used: %s", batch_size, clamped
        )
    return sizes


def _selection(
    data: EmbeddedDataset, train_split: str, val_split: str | None
) -> tuple[str, Judgments] | None:
    """Query split and judgments the checkpoint is selected on, or None to keep the last weights."""
    if val_split is None:
        return None
    dataset = data.dataset
    if val_split in data.queries and val_split in dataset.qrels:
        judgments = Judgments.from_dataset(dataset, val_split)
        if judgments.evaluable.any():
            return val_split, judgments
    if train_split in dataset.qrels:
        judgments = Judgments.from_dataset(dataset, train_split)
        if judgments.evaluable.any():
            logger.warning(
                "No %r queries with a relevant document in the corpus: the checkpoint is selected on the %r "
                "queries instead, which overestimates its quality.",
                val_split,
                train_split,
            )
            return train_split, judgments
    logger.warning("No queries with judged documents to select a checkpoint on: the last weights are kept.")
    return None


def _pair_batch(
    indices: list[int], pairs: np.ndarray, queries: torch.Tensor, corpus: torch.Tensor, device: torch.device
) -> PairBatch:
    query_rows, doc_rows = pairs[indices, 0], pairs[indices, 1]
    return PairBatch(
        queries[torch.from_numpy(query_rows)].to(device),
        corpus[torch.from_numpy(doc_rows)].to(device),
        query_rows,
        doc_rows,
    )
