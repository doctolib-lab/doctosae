"""Train a Sparse Autoencoder on dense embeddings (loss = MSE + β·L1 + λ·InfoNCE).
Val mirrors train for InfoNCE; val MSE/L1 only on chunks held out of train_corp. Primary
best checkpoint: best_sae.pth by val_<objective> (ndcg by default), used
for end-of-training val+test eval. Reference checkpoint: best_sae_loss.pth by
val_loss (saved but not evaluated).

Eval corpus for val/test: optional --val-corpus / --test-corpus appended with
train_corpus[:n_train_eval] (deduped). --test-corpus falls back to --val-corpus when absent. If
neither corpus is passed, n_train_eval must be > 0.

Example: python -m src.training.train_sae --train-corpus corpus_train.pkl --train-queries train.pkl --val-queries val.pkl --test-queries test.pkl --train-qrels trainval.tsv --val-qrels val.tsv --test-qrels test.tsv --corpus-id-col id --experiment-folder runs/x
"""
from __future__ import annotations

import argparse, json, math, time, warnings
from tqdm import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.sae_architectures import return_sae, geometric_median, _hidden_dim
from src.utils.training_utils import recon_loss, squared_error, weighted_loss, build_optimizer
from src.utils.cli import expand_path_args
from src.utils.datasets import EmbDataset, PairDataset
from src.utils.contrastive_utils import (
    load_pairs, info_nce,
    UniqueQueryBatchSampler, build_pos_dict, build_pos_mask,
)
from src.utils.encode import encode_in_batches, encode_in_batches_sparse
from src.utils.metrics import cosine_scores, load_qrels_matrix, ndcg_at_k, mean_auc
from src.utils.datasets_constants import dataset_from_sweep, max_score_for


# ───── data ────────────────────────────────────────────────────────────────

def load_emb_pkl(path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load a pickle with an ``embeddings`` column. Returns (df, X) where X is (N, D) float32."""
    df = pd.read_pickle(path)
    X = np.vstack(df["embeddings"].values).astype(np.float32)
    return df, X


def _index_only(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Index a corpus df on ``id_col`` and drop every column: only the index is used downstream, and
    keeping the columns would pin the embeddings duplicate for the whole run.
    """
    return df.set_index(id_col)[[]]


def iter_forever(loader):
    """Yield batches forever; unlike ``itertools.cycle``, re-iterates the loader each pass
    so the underlying sampler re-runs and fresh random draws are produced every cycle."""
    while True:
        for batch in loader:
            yield batch


# ───── step functions ──────────────────────────────────────────────────────

def train_step(model, opt, x_corp, c_idx_corp, x_q,
               xq, xd, q_idx_p, c_idx_p, score_p,
               pos_dict, beta, lam, T, recon_mode, device, alpha_q=1.0, symmetric_infonce=False,
               use_scaled_scores=False, k_aux=None, aux_coef=0.0, dead_mask=None):
    """One optimizer step: MSE + beta*L1 on (x_corp union x_q), lambda*InfoNCE on the pair batch with
    z_corp as the negative pool, plus Gao et al. 2024's auxiliary loss when k_aux is set.
    """
    m = model.module if isinstance(model, nn.DataParallel) else model
    opt.zero_grad()
    rc, zc = model(x_corp)
    rq, zq_emb = model(x_q)

    mse_c = recon_loss(x_corp, rc, mode=recon_mode)
    mse_q = recon_loss(x_q,    rq, mode=recon_mode)
    l1_c  = m.sparsity_loss(zc)
    l1_q  = m.sparsity_loss(zq_emb)
    nc, nq = x_corp.size(0), x_q.size(0)
    mse = weighted_loss(mse_c, mse_q, nc, nq, alpha_q)
    l1  = weighted_loss(l1_c,  l1_q,  nc, nq, alpha_q)

    if xq is not None:
        _, z_anchor = model(xq)
        _, z_pos    = model(xd)
        pos_in  = build_pos_mask(q_idx_p, c_idx_p,    pos_dict, device, skip_diagonal=True)
        pos_neg = build_pos_mask(q_idx_p, c_idx_corp, pos_dict, device)
        nce = info_nce(z_anchor, z_pos, z_neg=zc,
                       pos_mask=pos_in, pos_mask_neg=pos_neg,
                       scaled_scores=score_p if use_scaled_scores else None,
                       temperature=T, symmetric=symmetric_infonce)
        n_negs = ((pos_in.size(1) - pos_in.sum(dim=1) - 1)
                  + (pos_neg.size(1) - pos_neg.sum(dim=1))).float().mean().item()
    else:
        z_anchor = z_pos = None
        nce = torch.zeros((), device=device)
        n_negs = 0.0

    # Dead latents reconstruct the main model's detached residual, summed over dimensions.
    aux = torch.zeros((), device=device)
    if k_aux and dead_mask is not None and dead_mask.any():
        aux_c = squared_error((x_corp - rc).detach(), m.aux_recon(x_corp, dead_mask, k_aux))
        aux_q = squared_error((x_q - rq).detach(), m.aux_recon(x_q, dead_mask, k_aux))
        aux = weighted_loss(aux_c, aux_q, nc, nq, alpha_q)

    loss = mse + beta * l1 + lam * nce + aux_coef * aux
    loss.backward()
    if hasattr(m, "project_decoder_gradient"):
        m.project_decoder_gradient()
    opt.step()
    if hasattr(m, "normalize_decoder"):
        m.normalize_decoder()
    with torch.no_grad():
        z_main = torch.cat([zc, zq_emb])
        fired_terms = [(zc != 0).any(dim=0), (zq_emb != 0).any(dim=0)]
        if z_anchor is not None:
            fired_terms += [(z_anchor != 0).any(dim=0), (z_pos != 0).any(dim=0)]
        fired = fired_terms[0]
        for t in fired_terms[1:]:
            fired = fired | t
        l0 = (z_main != 0).sum(dim=1).float().mean().item()
    return (mse_c.detach().item(), mse_q.detach().item(), float(l1_c.detach()),
            float(l1_q.detach()), nce.detach().item(), l0, fired, n_negs, float(aux.detach()))


@torch.no_grad()
def val_step(model, val_corp_loader, val_query_loader, val_pair_loader,
             pos_dict_val, T, recon_mode, device, held_out_mask=None, alpha_q=1.0, symmetric_infonce=False,
             use_scaled_scores=False):
    """Held-out monitoring (no grad), mirroring ``train_step``, on one val corpus + query + pair
    batch. Averages are example-count-weighted.
    """
    model.eval()
    m = model.module if isinstance(model, nn.DataParallel) else model
    tot_mse = tot_l1 = tot_nce = tot_negs = 0.0
    tot_mse_c = tot_mse_q = tot_l1_c = tot_l1_q = 0.0
    n_combined = n_q = n_corp = n_nce = 0
    query_iter = iter_forever(val_query_loader)
    pair_iter  = iter_forever(val_pair_loader) if val_pair_loader is not None else None
    for x_corp, c_idx_corp in val_corp_loader:
        x_corp = x_corp.to(device); c_idx_corp = c_idx_corp.to(device)
        x_q, _ = next(query_iter); x_q = x_q.to(device)

        rc, zc = model(x_corp)
        rq, zq_emb = model(x_q)

        if pair_iter is not None:
            xq, xd, q_idx_p, c_idx_p, score_p = next(pair_iter)
            xq = xq.to(device); xd = xd.to(device); score_p = score_p.to(device)
            _, z_anchor = model(xq)
            _, z_pos    = model(xd)
            pos_in  = build_pos_mask(q_idx_p, c_idx_p,    pos_dict_val, device, skip_diagonal=True)
            pos_neg = build_pos_mask(q_idx_p, c_idx_corp, pos_dict_val, device)
            nce = info_nce(z_anchor, z_pos, z_neg=zc,
                           pos_mask=pos_in, pos_mask_neg=pos_neg,
                           scaled_scores=score_p if use_scaled_scores else None,
                           temperature=T, symmetric=symmetric_infonce)
            n_anchor = xq.size(0)
            tot_nce  += nce.item() * n_anchor
            tot_negs += ((pos_in.size(1) - pos_in.sum(dim=1) - 1)
                         + (pos_neg.size(1) - pos_neg.sum(dim=1))).float().mean().item() * n_anchor
            n_nce += n_anchor

        mse_q = recon_loss(x_q, rq, mode=recon_mode)
        l1_q  = m.sparsity_loss(zq_emb)
        nq = x_q.size(0)
        tot_mse_q += mse_q.item() * nq; tot_l1_q += float(l1_q) * nq; n_q += nq

        if held_out_mask is not None:
            keep = held_out_mask[c_idx_corp]
            x_corp_ho, rc_ho, zc_ho = x_corp[keep], rc[keep], zc[keep]
        else:
            x_corp_ho, rc_ho, zc_ho = x_corp, rc, zc

        nc = x_corp_ho.size(0)
        if nc > 0:
            mse_c = recon_loss(x_corp_ho, rc_ho, mode=recon_mode)
            l1_c  = m.sparsity_loss(zc_ho)
            tot_mse_c += mse_c.item() * nc; tot_l1_c += float(l1_c) * nc; n_corp += nc
            tot_mse += weighted_loss(mse_c, mse_q, nc, nq, alpha_q).item() * (nc + nq)
            tot_l1  += float(weighted_loss(l1_c,  l1_q,  nc, nq, alpha_q)) * (nc + nq)
            n_combined += nc + nq
        elif alpha_q > 0:
            tot_mse += mse_q.item() * nq
            tot_l1  += float(l1_q) * nq
            n_combined += nq
    model.train()
    if n_q == 0:
        print("[WARNING] val_step: no query batches — val MSE/L1 are zero", flush=True)
    if n_corp == 0:
        print("[WARNING] val_step: no corpus batches — val MSE_corp/L1_corp are zero", flush=True)
    return (tot_mse / max(n_combined, 1),
            tot_l1  / max(n_combined, 1),
            tot_mse_c / max(n_corp, 1),
            tot_mse_q / max(n_q, 1),
            tot_l1_c  / max(n_corp, 1),
            tot_l1_q  / max(n_q, 1),
            tot_nce / max(n_nce, 1),
            tot_negs / max(n_nce, 1))


def _encode_eval(model, X, device):
    """Encode a full split for periodic eval: sparse (csr) for TopK-family models, dense otherwise."""
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "k"):
        return encode_in_batches_sparse(model, X, device)
    return encode_in_batches(model, X, device)


@torch.no_grad()
def ndcg_eval(model, Xq, Xc, chunk_uids, qrels_path, q_df, device, k=10, score_col="relevance", qp=None):
    """Encode queries + corpus through the SAE and compute NDCG@k against a qrels TSV."""
    Zq = _encode_eval(model, Xq, device)
    Zc = _encode_eval(model, Xc, device)
    S, _ = cosine_scores(Zq, Zc, chunk_uids)
    if qp is None:
        qp = load_qrels_matrix(q_df, qrels_path, pd.DataFrame(index=chunk_uids), score_col=score_col)
    val = ndcg_at_k(Zq, Zc, chunk_uids, qp, k=k, S=S)
    model.train()
    return val


@torch.no_grad()
def auc_eval(model, Xq, Xc, chunk_uids, qrels_path, q_df, device, score_col="relevance"):
    """Encode queries + corpus through the SAE and compute mean per-query AUC over judged chunks."""
    Zq = _encode_eval(model, Xq, device)
    Zc = _encode_eval(model, Xc, device)
    qid_to_row = {str(qid): i for i, qid in enumerate(q_df["id"].astype(str))}
    cid_to_row = {str(cid): i for i, cid in enumerate(chunk_uids)}
    qrels_df = pd.read_csv(qrels_path, sep="\t")
    auc, _ = mean_auc(Zq, Zc, qid_to_row, cid_to_row, qrels_df, score_col=score_col)
    model.train()
    return float(auc)


def score_eval(model, Xq, Xc, chunk_uids, qrels_path, q_df, device, objective, score_col="relevance", qp=None):
    """Dispatch to ndcg_eval or auc_eval based on --objective."""
    if objective == "ndcg":
        return ndcg_eval(model, Xq, Xc, chunk_uids, qrels_path, q_df, device, score_col=score_col, qp=qp)
    if objective == "auc":
        return auc_eval(model, Xq, Xc, chunk_uids, qrels_path, q_df, device, score_col=score_col)
    raise ValueError(f"unknown objective: {objective!r}")


# ───── main ────────────────────────────────────────────────────────────────

def _build_eval_corpus(xc_split: np.ndarray | None, uids_split: np.ndarray | None,
                       Xc_train: np.ndarray, train_uids: np.ndarray,
                       n_train_eval: int) -> tuple[np.ndarray, np.ndarray]:
    """Build eval corpus: optional split corpus + train[:n_train_eval], deduped."""
    parts_x = [] if xc_split is None else [xc_split]
    parts_u = [] if uids_split is None else [uids_split]
    if n_train_eval > 0:
        parts_x.append(Xc_train[:n_train_eval])
        parts_u.append(train_uids[:n_train_eval])
    if not parts_x:
        raise ValueError("Eval corpus is empty: provide --val/test-corpus or --val-score-train-size > 0")
    Xc = np.vstack(parts_x)
    uids = np.concatenate(parts_u)
    _, first_idx = np.unique(uids, return_index=True)
    first_idx.sort()
    return Xc[first_idx], uids[first_idx]


def main():
    args = expand_path_args(_parse_args())
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    exp_dir = Path(args.experiment_folder); exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"[corpus] train={args.train_corpus}")
    train_corp_df, Xc_train = load_emb_pkl(args.train_corpus)
    train_corp_df = _index_only(train_corp_df, args.corpus_id_col)
    train_uids = np.asarray(train_corp_df.index.tolist())

    has_infonce = args.lambda_ > 0
    n_train_eval = min(
        args.val_score_train_size if args.val_score_train_size is not None else len(Xc_train),
        len(Xc_train)
    )

    # Val eval corpus
    Xc_val, val_uids = None, None
    if args.val_corpus:
        val_corp_df, Xc_val = load_emb_pkl(args.val_corpus)
        val_corp_df = _index_only(val_corp_df, args.corpus_id_col)
        val_uids = np.asarray(val_corp_df.index.tolist())
    Xc_eval_val, uids_eval_val = _build_eval_corpus(Xc_val, val_uids, Xc_train, train_uids, n_train_eval)
    val_held_out_np = ~pd.Index(uids_eval_val).isin(train_corp_df.index)
    val_held_out_t = torch.tensor(val_held_out_np, dtype=torch.bool, device=device)

    # Falls back to --val-corpus.
    Xc_test_split, test_split_uids = None, None
    test_corpus_path = args.test_corpus or args.val_corpus
    if test_corpus_path:
        test_corp_df, Xc_test_split = load_emb_pkl(test_corpus_path)
        test_corp_df = _index_only(test_corp_df, args.corpus_id_col)
        test_split_uids = np.asarray(test_corp_df.index.tolist())
    Xc_eval_test, uids_eval_test = _build_eval_corpus(Xc_test_split, test_split_uids, Xc_train, train_uids, n_train_eval)

    val_qrels = args.val_qrels or args.train_qrels

    q_train_df, Xq_train = load_emb_pkl(args.train_queries)
    q_val_df,   Xq_val   = load_emb_pkl(args.val_queries)
    q_test_df,  Xq_test  = load_emb_pkl(args.test_queries)

    size_line = (
        f"[sizes] corpus: train={len(Xc_train)} "
        f"val={len(Xc_val) if Xc_val is not None else 0} "
        f"(held-out val={int(val_held_out_np.sum())})  |  "
        f"eval corpus: val={len(uids_eval_val)} test={len(uids_eval_test)}  |  "
        f"queries: train={len(Xq_train)} val={len(Xq_val)}"
    )
    print(size_line)
    with open(exp_dir / "training_log.txt", "a") as f:
        f.write(size_line + "\n")
    for df in [q_train_df, q_val_df, q_test_df]:
        df["id"] = df[args.query_id_col].astype(str)

    # eval corpus DataFrames (for load_pairs index mapping)
    eval_val_corp_df  = pd.DataFrame(index=uids_eval_val)

    if has_infonce:
        pairs_train, scores_train = load_pairs(q_train_df, train_corp_df,    args.train_qrels)
        pairs_val,   scores_val   = load_pairs(q_val_df,   eval_val_corp_df, val_qrels)
        if args.use_scaled_scores:
            max_score = max_score_for(dataset_from_sweep({"train_corpus": args.train_corpus}))
            scores_train = scores_train / max_score
            scores_val = scores_val / max_score
        pos_dict     = build_pos_dict(pairs_train)
        pos_dict_val = build_pos_dict(pairs_val)
        print(f"[pairs] train={len(pairs_train)}  val={len(pairs_val)}")
    else:
        pos_dict = pos_dict_val = None
        print("[pairs] skipped (lambda=0)")

    # loaders
    Xc_train_mse, mse_ids = Xc_train, np.arange(len(Xc_train))
    if args.train_corpus_cap is not None and args.train_corpus_cap < len(Xc_train):
        mse_ids = np.random.default_rng(args.seed).choice(len(Xc_train), args.train_corpus_cap, replace=False)
        Xc_train_mse = Xc_train[mse_ids]
    assert np.array_equal(Xc_train[mse_ids], Xc_train_mse)
    corpus_loader = DataLoader(EmbDataset(Xc_train_mse, mse_ids), batch_size=args.batch,
                               shuffle=True, drop_last=True, num_workers=args.num_workers)
    query_loader  = DataLoader(EmbDataset(Xq_train, np.arange(len(Xq_train))),
                               batch_size=args.batch, shuffle=True, drop_last=True,
                               num_workers=args.num_workers)
    pair_loader = (DataLoader(PairDataset(pairs_train, scores_train, Xq_train, Xc_train),
                              batch_sampler=UniqueQueryBatchSampler(pairs_train, args.batch, seed=args.seed),
                              num_workers=args.num_workers)
                   if has_infonce else None)
    val_corp_loader  = DataLoader(EmbDataset(Xc_eval_val, np.arange(len(Xc_eval_val))), batch_size=args.batch,
                                  shuffle=False, drop_last=False, num_workers=args.num_workers)
    val_query_loader = DataLoader(EmbDataset(Xq_val, np.arange(len(Xq_val))),
                                  batch_size=args.batch, shuffle=False, drop_last=False,
                                  num_workers=args.num_workers)
    val_pair_loader = (DataLoader(PairDataset(pairs_val, scores_val, Xq_val, Xc_eval_val),
                                  batch_sampler=UniqueQueryBatchSampler(pairs_val, args.batch, seed=args.seed, drop_last=False),
                                  num_workers=args.num_workers)
                       if has_infonce else None)

    # model + b_pre at geometric median of training corpus
    input_dim = Xc_train.shape[1]
    model = return_sae(input_dim=input_dim, sparsity_type=args.sparsity_type,
                       hidden_dim=args.hidden, k=args.k,
                       intermediate_dim=args.intermediate,
                       activation_fn=args.activation_fn, init=args.init).to(device)

    # DataParallel, gated on the resolved device.
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        print(f"[DataParallel] Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model

    init = torch.tensor(geometric_median(np.vstack([Xc_train, Xq_train]), seed=args.seed),
                        dtype=torch.float32, device=device)
    if hasattr(raw_model, "set_b_pre"): raw_model.set_b_pre(init)
    opt = build_optimizer(model, args.lr, args.wd)
    hidden_dim = _hidden_dim(model)

    _save_config(args, input_dim, exp_dir)

    qp_val = (load_qrels_matrix(q_val_df, val_qrels, pd.DataFrame(index=uids_eval_val), score_col=args.score_col)
              if args.objective == "ndcg" else None)

    # training loop (step-based: validate every N optimizer steps)
    query_iter = iter_forever(query_loader)
    pair_iter  = iter_forever(pair_loader) if has_infonce else None
    best_val_loss = math.inf; best_step_loss = 0
    best_val_score = -math.inf; best_step = 0
    log_path = exp_dir / "training_log.txt"
    history = {k: [] for k in ("mse_c","mse_q","l1_c","l1_q","nce",
                               "val_mse_c","val_mse_q","val_l1_c","val_l1_q","val_nce",
                               "l0","val_score","step")}

    total_steps = args.epochs * len(corpus_loader)
    tot_mse_c = tot_mse_q = tot_l1_c = tot_l1_q = tot_nce = tot_l0 = tot_negs = tot_aux = n_batches = 0
    fired = torch.zeros(hidden_dim, dtype=torch.bool, device=device)
    # Latents silent for a whole epoch: the auxiliary loss's dead set.
    dead_mask = None
    t0 = time.time()
    step = 0
    model.train()
    pbar = tqdm(total=total_steps, desc="train", leave=True)
    for epoch in range(1, args.epochs + 1):
        pbar.set_description(f"ep {epoch}/{args.epochs}")
        for x_corp, c_idx_corp in corpus_loader:
            step += 1
            x_corp = x_corp.to(device); c_idx_corp = c_idx_corp.to(device)
            x_q, _ = next(query_iter); x_q = x_q.to(device)
            if pair_iter is not None:
                xq, xd, q_idx_p, c_idx_p, score_p = next(pair_iter)
                xq = xq.to(device); xd = xd.to(device); score_p = score_p.to(device)
            else:
                xq = xd = q_idx_p = c_idx_p = score_p = None
            mc, mq, lc, lq, n, l0, fired_batch, n_negs, a = train_step(model, opt, x_corp, c_idx_corp, x_q,
                                                                    xq, xd, q_idx_p, c_idx_p, score_p,
                                                                    pos_dict, args.beta, args.lambda_, args.temperature,
                                                                    args.recon_loss, device, alpha_q=args.alpha_q,
                                                                    symmetric_infonce=args.symmetric_infonce,
                                                                    use_scaled_scores=args.use_scaled_scores,
                                                                    k_aux=args.k_aux, aux_coef=args.aux_coef,
                                                                    dead_mask=dead_mask)
            tot_mse_c += mc; tot_mse_q += mq; tot_l1_c += lc; tot_l1_q += lq
            tot_nce += n; tot_l0 += l0; tot_negs += n_negs; tot_aux += a; n_batches += 1
            fired |= fired_batch
            pbar.update(1)
            pbar.set_postfix(mse=f"{(mc+mq)/2:.4f}", nce=f"{n:.4f}", l0=f"{l0:.1f}")
            is_last = step == total_steps

            # ── validation every --val-every-steps (metrics averaged over the window since last val) ──
            if step % args.val_every_steps == 0 or is_last:
                vmse, vl1, vmse_c, vmse_q, vl1_c, vl1_q, vnce, vnegs = val_step(
                    model, val_corp_loader, val_query_loader, val_pair_loader,
                    pos_dict_val, args.temperature, args.recon_loss, device,
                    held_out_mask=val_held_out_t, alpha_q=args.alpha_q,
                    symmetric_infonce=args.symmetric_infonce,
                    use_scaled_scores=args.use_scaled_scores)
                val_loss = vmse + args.beta * vl1 + args.lambda_ * vnce
                val_score = score_eval(model, Xq_val, Xc_eval_val, uids_eval_val, val_qrels, q_val_df, device,
                                       objective=args.objective, score_col=args.score_col, qp=qp_val)

                nce_str      = f"{tot_nce/n_batches:.4f}" if has_infonce else "n/a"
                val_nce_str  = f"{vnce:.4f}"             if has_infonce else "n/a"
                negs_str     = f"{tot_negs/n_batches:.1f}" if has_infonce else "n/a"
                val_negs_str = f"{vnegs:.1f}"             if has_infonce else "n/a"

                history["mse_c"].append(tot_mse_c/n_batches); history["mse_q"].append(tot_mse_q/n_batches)
                history["l1_c"].append(tot_l1_c/n_batches);   history["l1_q"].append(tot_l1_q/n_batches)
                history["nce"].append(tot_nce/n_batches if has_infonce else None)
                history["val_mse_c"].append(vmse_c); history["val_mse_q"].append(vmse_q)
                history["val_l1_c"].append(vl1_c);   history["val_l1_q"].append(vl1_q)
                history["val_nce"].append(vnce if has_infonce else None)
                history["l0"].append(tot_l0/n_batches)
                history["val_score"].append(val_score)
                history["step"].append(step)

                if val_loss < best_val_loss:
                    best_val_loss, best_step_loss = val_loss, step
                    torch.save(raw_model.state_dict(), exp_dir / "best_sae_loss.pth")
                if val_score > best_val_score and not math.isnan(val_score):
                    best_val_score, best_step = val_score, step
                    torch.save(raw_model.state_dict(), exp_dir / "best_sae.pth")

                n_dead = int((~fired).sum().item())
                line = (f"ep {epoch:>3} step {step:>6}  mse_c={tot_mse_c/n_batches:.5f} mse_q={tot_mse_q/n_batches:.5f}  "
                        f"l1_c={tot_l1_c/n_batches:.5f} l1_q={tot_l1_q/n_batches:.5f}  "
                        f"nce={nce_str}  negs={negs_str}  l0={tot_l0/n_batches:.1f}  "
                        f"val_mse_c={vmse_c:.5f} val_mse_q={vmse_q:.5f}  "
                        f"val_l1_c={vl1_c:.5f} val_l1_q={vl1_q:.5f}  val_nce={val_nce_str}  val_negs={val_negs_str}  "
                        f"val_loss={val_loss:.5f}  val_{args.objective}={val_score:.4f}  dead={n_dead}  "
                        f"aux={tot_aux/n_batches:.5f}  {time.time()-t0:.0f}s")
                print(line)
                with open(log_path, "a") as f: f.write(line + "\n")
                t0 = time.time()
                tot_mse_c = tot_mse_q = tot_l1_c = tot_l1_q = tot_nce = tot_l0 = tot_negs = tot_aux = n_batches = 0

        # fired spans the epoch, so dead= is "silent so far this epoch".
        dead_mask = ~fired
        fired.zero_()

    pbar.close()

    # final eval on best-by-val-<objective> checkpoint
    obj = args.objective
    raw_model.load_state_dict(torch.load(exp_dir / "best_sae.pth", map_location=device, weights_only=True))
    val_score_best  = score_eval(model, Xq_val,  Xc_eval_val,  uids_eval_val,  val_qrels,        q_val_df,  device, obj, score_col=args.score_col)
    test_score      = score_eval(model, Xq_test, Xc_eval_test, uids_eval_test, args.test_qrels,  q_test_df, device, obj, score_col=args.score_col)
    line = (f"\n=== best (by val_{obj}) step={best_step}  val_{obj}={val_score_best:.4f}  test_{obj}={test_score:.4f}")
    print(line)
    with open(log_path, "a") as f: f.write(line + "\n")

    # final eval on best-by-val_loss checkpoint
    raw_model.load_state_dict(torch.load(exp_dir / "best_sae_loss.pth", map_location=device, weights_only=True))
    val_score_loss  = score_eval(model, Xq_val,  Xc_eval_val,  uids_eval_val,  val_qrels,       q_val_df,  device, obj, score_col=args.score_col)
    test_score_loss = score_eval(model, Xq_test, Xc_eval_test, uids_eval_test, args.test_qrels, q_test_df, device, obj, score_col=args.score_col)
    line_loss = (f"=== best (by val_loss) step={best_step_loss}  val_loss={best_val_loss:.5f}  "
                 f"val_{obj}={val_score_loss:.4f}  test_{obj}={test_score_loss:.4f}")
    print(line_loss)
    with open(log_path, "a") as f: f.write(line_loss + "\n")

    _save_summary(args, exp_dir, best_val_loss, best_step_loss,
                  best_val_score, best_step, test_score, obj,
                  val_score_loss, test_score_loss)
    _plot_curves(history, has_infonce, best_step_loss, best_step, obj, exp_dir, args.sparsity_type)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, default=1536)
    p.add_argument("--sparsity-type", default="l1", choices=["l1", "l1_2layer", "kl", "topk", "topknorelu"])
    p.add_argument("--intermediate", type=int, default=None)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--k-aux", type=int, default=None, dest="k_aux",
                   help="Dead latents reconstructing the residual (Gao et al. 2024), typically 2k. "
                        "Omit to disable the auxiliary loss")
    p.add_argument("--aux-coef", type=float, default=0.03125, dest="aux_coef",
                   help="Auxiliary loss weight, used only with --k-aux")
    p.add_argument("--activation-fn", default="relu")
    p.add_argument("--init", default="default", choices=["default", "gao"],
                   help="'default': decoder tied to the encoder's transpose (Kaiming-uniform "
                        "directions), then unit-normed -- encoder itself is never rescaled. "
                        "'gao': Gao et al. 2024 tied init -- same decoder-only-norm scheme, but "
                        "with fresh uniformly-random directions (not the encoder's own Kaiming "
                        "draw) and a zeroed encoder bias. Single encoder/decoder pair only (not "
                        "l1_2layer/kl).")
    p.add_argument("--beta", type=float, default=5e-3)
    p.add_argument("--lambda", type=float, default=5e-3, dest="lambda_")
    p.add_argument("--query-recon-weight", type=float, default=1, dest="alpha_q",
                   help="α: scales the query contribution in MSE+L1 (combined = (N_corp·c + α·N_q·q)/(N_corp + α·N_q)). 1.0 = old behavior; 0 = corpus-only reconstruction.")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--symmetric-infonce", action="store_true", dest="symmetric_infonce",
                   help="Use symmetric InfoNCE (q→d + d→q). Default: q→d only.")
    p.add_argument("--use-scaled-scores", action="store_true", dest="use_scaled_scores",
                   help="Scale each pair's qrels score by 1/max(score) (inferred from the train qrels) before use in the InfoNCE loss.")
    p.add_argument("--recon-loss", default="mse", choices=["mse", "cosine"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--val-every-steps", type=int, default=20, dest="val_every_steps",
                   help="run validation (val loss + NDCG/AUC + checkpoint + log) every N optimizer steps")
    p.add_argument("--val-score-train-size", type=int, default=None, dest="val_score_train_size",
                   help="number of train corpus rows to include in the val-score retrieval pool (default: all)")
    p.add_argument("--train-corpus-cap", type=int, default=None, dest="train_corpus_cap",
                   help="subsample the train corpus to N rows for the corpus MSE loader only "
                        "(reduces steps/epoch, so queries are cycled fewer times per epoch); "
                        "pairs/eval corpus still use the full train corpus")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--corpus-id-col", default="id",
                   help="Column in the corpus pkl that uniquely identifies each row "
                        "(e.g. document for BeIR). Used as the corpus index and "
                        "must match the qrels 'corpus-id' values.")
    p.add_argument("--query-id-col", default="query_id",
                   help="Column in the queries pkl that uniquely identifies each row "
                        "(e.g. '_id' for BeIR). Must match the qrels 'query-id' values.")
    p.add_argument("--score-col", default="relevance",
                   help="Column name in the qrels TSV holding the relevance score (e.g. 'score' for BeIR).")
    p.add_argument("--objective", default="ndcg", choices=["ndcg", "auc"],
                   help="Metric used to select best_sae.pth and reported at end-of-training (default: ndcg). "
                        "Use 'auc' when most (query, document) pairs are unjudged and NDCG@10 is biased.")
    for arg in ("train-corpus", "train-queries", "val-queries", "test-queries",
                "train-qrels", "test-qrels", "experiment-folder"):
        p.add_argument(f"--{arg}", type=Path, required=True)
    p.add_argument("--val-corpus", type=Path, default=None,
                   help="Val corpus pkl. Appended with train[:n_train_eval]. If absent, eval corpus = train[:n_train_eval].")
    p.add_argument("--test-corpus", type=Path, default=None,
                   help="Test corpus pkl. Appended with train[:n_train_eval]. If absent, falls back to "
                        "--val-corpus (so val-only positives aren't dropped from test retrieval); "
                        "if that's also absent, eval corpus = train[:n_train_eval].")
    p.add_argument("--val-qrels", type=Path, default=None,
                   help="Qrels TSV for val evaluation. Defaults to --train-qrels if not set.")
    return p.parse_args()


def _save_config(args, input_dim, exp_dir):
    (exp_dir / "sae_config.json").write_text(json.dumps({
        "input_dim": input_dim, "hidden": args.hidden, "sparsity_type": args.sparsity_type,
        "intermediate": args.intermediate, "k": args.k, "activation_fn": args.activation_fn,
        "init": args.init, "beta": args.beta, "lambda": args.lambda_, "alpha_q": args.alpha_q,
        "temperature": args.temperature, "recon_loss": args.recon_loss,
        "k_aux": args.k_aux, "aux_coef": args.aux_coef if args.k_aux else None,
    }, indent=2))


def _plot_curves(history, has_infonce, best_step_loss, best_step, objective, exp_dir, sparsity_type):
    """Save one PNG with a subplot per loss term (train + val overlaid), l0 and val_<objective>. Grey
    dashed line: best step by val_loss; green: best by val_<objective>. l1_c/l1_q are omitted for
    topk/topknorelu."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = [("mse_c", "val_mse_c"), ("mse_q", "val_mse_q")]
    if sparsity_type not in ("topk", "topknorelu"):
        pairs += [("l1_c", "val_l1_c"), ("l1_q", "val_l1_q")]
    if has_infonce:
        pairs.append(("nce", "val_nce"))
    n_panels = len(pairs) + 2  # + l0 + val_score
    n_cols = 3
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    axes = axes.flatten()
    xs = history["step"]
    def _mark(ax):
        ax.axvline(best_step_loss, color="grey",  linestyle="--", alpha=0.6)
        ax.axvline(best_step,      color="green", linestyle="--", alpha=0.6)
    for ax, (tk, vk) in zip(axes, pairs):
        ax.plot(xs, history[tk], label="train", color="C0")
        ax.plot(xs, history[vk], label="val",   color="C1")
        _mark(ax)
        ax.set_title(tk); ax.set_xlabel("step"); ax.legend(fontsize=8)
    ax = axes[len(pairs)]
    ax.plot(xs, history["l0"], color="C3")
    _mark(ax)
    ax.set_title("l0 (train)"); ax.set_xlabel("step")
    ax = axes[len(pairs) + 1]
    ax.plot(xs, history["val_score"], color="C2")
    _mark(ax)
    ax.set_title(f"val_{objective}"); ax.set_xlabel("step")
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    fig.tight_layout()
    out = exp_dir / "training_curves.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")


def _save_summary(args, exp_dir, best_val_loss, best_step_loss,
                  best_val_score, best_step, test_score, objective,
                  val_score_loss=None, test_score_loss=None):
    summary = {
        "objective": objective,
        "best_step": int(best_step),
        f"best_val_{objective}": float(best_val_score),
        f"test_{objective}": float(test_score),
        "best_step_loss": int(best_step_loss),
        "best_val_loss": float(best_val_loss),
        "config": vars(args),
    }
    if val_score_loss is not None:
        summary[f"val_{objective}_at_best_loss"]  = float(val_score_loss)
        summary[f"test_{objective}_at_best_loss"] = float(test_score_loss)
    (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
