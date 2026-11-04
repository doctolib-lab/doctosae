"""2D projection of a convert_embeddings.py '*_with_sae.pkl', one panel per --emb-type:
embeddings (original), "reconstructed embeddings" (x_hat), "sae embeddings" (sparse code z).

Rows are L2-normalized by default (--no-normalize to disable). Sparse codes stay sparse through
PCA (TruncatedSVD) and get an SVD pre-reduction before t-SNE/UMAP (--pre-dims). --max-rows
subsamples for the projection only; stats run on the full corpus first.

Neurons are chosen as select_neurons.py chooses them (firing-rate band, --rank-by,
--diversity-overlap). A neuron's points are the documents its top terms are computed over.

Usage:
    RUN=runs/repo_cure/bge-large-en-v1.5/cure_k64_l1e-5_a0.1_lr5e-4_temp0.05_s42_h3072
    python -m src.interpretability.plot_2d_projection \\
        --sae-file $RUN/converted/corpus_val_with_sae.pkl \\
        --emb-type embeddings "reconstructed embeddings" "sae embeddings" \\
        --method tsne --color-by discipline --max-rows 4000 \\
        --model-name cure_k64_l1e-5_..._h3072 --output-dir $RUN/charts
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

from src.interpretability.chart_style import (apply_style, add_model_footer, distinct_colors, save,
                                             INK, MUTED, PALETTE)
from src.utils.cli import embedded_text_col
from src.utils.interpretability_utils import (STOPWORDS_BY_LANGUAGE, dedup_by_text,
                                              dense_codes_from_column, fit_tfidf, is_missing_label,
                                              neuron_fired_columns, neuron_fired_mask,
                                              neuron_firing_stats, rank_by_score, report_selection,
                                              select_diverse_neurons, sparse_codes_from_column,
                                              stopwords_for, top_terms_for_mask, topn_fired_mask)

SPARSE_COL = "sae embeddings"


def load_space(df, emb_type: str, hidden: int | None):
    """(N, D) matrix for one embedding column: CSR if sparse-stored, dense otherwise."""
    if emb_type not in df.columns:
        raise ValueError(f"no column {emb_type!r} in the pkl (found: {list(df.columns)})")
    col = df[emb_type]
    if emb_type == SPARSE_COL and isinstance(col.iloc[0], tuple):
        h = hidden if hidden is not None else df.attrs.get("sae_hidden")
        if h is None:
            raise ValueError(f"{emb_type!r} is stored sparse but the pkl carries no "
                             f"attrs['sae_hidden'] -- pass --hidden")
        return sparse_codes_from_column(col.values, h)
    return dense_codes_from_column(col)


def project(X, method: str, pre_dims: int, seed: int, perplexity: float):
    """(N, 2) projection of X (dense or CSR). PCA = TruncatedSVD when sparse (no densifying via
    mean-centering); t-SNE/UMAP get an SVD pre-reduction first (the standard recipe)."""
    if method == "pca":
        return (TruncatedSVD(2, random_state=seed) if sp.issparse(X)
                else PCA(2, random_state=seed)).fit_transform(X)
    d = min(pre_dims, min(X.shape) - 1)
    Xp = TruncatedSVD(d, random_state=seed).fit_transform(X) if d < X.shape[1] else X
    Xp = np.asarray(Xp.todense()) if sp.issparse(Xp) else np.asarray(Xp)
    if method == "tsne":
        return TSNE(2, random_state=seed, init="pca",
                    perplexity=min(perplexity, len(Xp) - 1)).fit_transform(Xp)
    try:
        import umap
    except ImportError:
        raise SystemExit("--method umap needs the umap-learn package (not installed); "
                         "use --method tsne or pca")
    return umap.UMAP(n_components=2, random_state=seed).fit_transform(Xp)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sae-file", required=True, type=Path,
                   help="A convert_embeddings.py '*_with_sae.pkl' (holds all three spaces).")
    p.add_argument("--emb-type", nargs="+",
                   default=["embeddings", "reconstructed embeddings", SPARSE_COL],
                   help="Embedding columns to project, one panel each, left to right "
                        f"(default: the three spaces -- embeddings, reconstructed, {SPARSE_COL!r}).")
    p.add_argument("--method", choices=["pca", "tsne", "umap"], default="tsne",
                   help="pca is exact and fast; tsne (default) shows local neighbourhood "
                        "structure; umap needs umap-learn installed.")
    p.add_argument("--color-by", default=None,
                   help="Metadata column to colour points by (e.g. discipline, expertise). "
                        "Rows with a missing/NaN label are drawn grey and labelled 'unlabelled'. "
                        "Omit for a single-colour scatter.")
    p.add_argument("--color-by-neuron", type=int, nargs="+", default=None,
                   help="Colour by SAE neuron instead of metadata. ONE index: the rows it fires on "
                        "vs the rest -- where does that learned subtopic sit in each space? "
                        "SEVERAL: each row takes the colour of whichever of them fires strongest "
                        "(rows where none fire go grey) -- do the clusters correspond to neurons? "
                        "Each legend entry is labelled with that neuron's top TF-IDF terms.")
    p.add_argument("--auto-neurons", type=int, default=None, metavar="M",
                   help="Pick the M neurons to colour by automatically instead of naming them, by "
                        "select_neurons.py's rule: firing rate within [--fr-min, --fr-max] "
                        "(neither dead nor a generic hub), ranked by --rank-by, then "
                        "--diversity-overlap applied.")
    p.add_argument("--fr-min", type=float, default=0.01,
                   help="--auto-neurons firing-rate floor (default 0.01, as select_neurons.py).")
    p.add_argument("--fr-max", type=float, default=0.20,
                   help="--auto-neurons firing-rate ceiling (default 0.20).")
    p.add_argument("--text-col", default="text",
                   help="Text column used for the neurons' top-TF-IDF-term labels (default 'text').")
    p.add_argument("--language", nargs="+", choices=sorted(STOPWORDS_BY_LANGUAGE), default=["en"],
                   help="Corpus language(s), for TF-IDF stopword removal in neuron labels -- pass "
                        "several for mixed-language corpora (e.g. fr en), stopword sets are unioned.")
    p.add_argument("--n-terms", type=int, default=5,
                   help="Top TF-IDF terms per neuron label (default 5).")
    p.add_argument("--rank-by", choices=["mass", "entropy"], default="entropy",
                   help="--auto-neurons ranking, as select_neurons.py: 'entropy' (default) "
                        "puts the lowest in-domain theme entropy first and needs --theme-col; "
                        "'mass' puts the highest activation mass first.")
    p.add_argument("--top-n", type=int, default=100,
                   help="Restrict each neuron to its top-N highest-activation documents (default "
                        "100) -- for the entropy ranking, the top terms, and the points drawn for "
                        "it. select_neurons.py's own --top-n defaults to the full fired set instead. "
                        "Pass 0 for every document it fires on.")
    p.add_argument("--boilerplate-min-doc-frac", type=float, default=None,
                   help="Off by default, as select_neurons.py. If given, drop lines (cookie "
                        "banners, nav templates, citation footers) that verbatim-recur in at "
                        "least this fraction of documents before TF-IDF (e.g. 0.005).")
    p.add_argument("--diversity-overlap", type=float, default=None,
                   help="Off by default. If given, skip a candidate once its top terms' Jaccard "
                        "overlap with an already-kept neuron reaches this (e.g. 0.5) -- stops the "
                        "colours being spent on several neurons for the same concept.")
    p.add_argument("--theme-col", default=None,
                   help="Write each theme's name on the plot at the centroid of its points (e.g. "
                        "source_folder). Colour is already spoken for -- by --color-by or by the "
                        "neurons -- so themes go on as text instead, letting you read both at once: "
                        "which theme a region is, and which neuron fires there.")
    p.add_argument("--where", default=None, metavar="COL=VALUE",
                   help="Keep only rows whose COL equals VALUE, applied before subsampling and "
                        "before --auto-neurons picks anything. Use it to look inside a single "
                        "theme (e.g. --where source_folder=genetics): ranking neurons over the "
                        "whole corpus surfaces the big coarse themes, so same-theme/"
                        "different-subtopic neurons only show up once the theme is held fixed.")
    p.add_argument("--max-rows", type=int, default=4000,
                   help="Subsample to at most this many rows (default 4000) -- t-SNE/UMAP do not "
                        "scale to a full 244k corpus. Every panel uses the same subsample.")
    p.add_argument("--max-classes", type=int, default=8,
                   help="Keep the this-many most frequent --color-by values, pooling the rest into "
                        "'other' (default 8, the palette length).")
    p.add_argument("--pre-dims", type=int, default=50,
                   help="SVD pre-reduction width before t-SNE/UMAP (default 50; ignored for pca).")
    p.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity (default 30).")
    p.add_argument("--no-normalize", action="store_true",
                   help="Skip the default per-row L2 normalization (which matches how every "
                        "retrieval metric here scores).")
    p.add_argument("--hidden", type=int, default=None,
                   help="SAE hidden size for a sparsely-stored 'sae embeddings' column; defaults "
                        "to the pkl's own attrs['sae_hidden'] and is only needed if that is absent.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-name", default=None, help="Checkpoint tag for the figure footer.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--stem", default=None,
                   help="Output filename stem (default: 2d_<method>).")
    args = p.parse_args()

    top_n = args.top_n or None            # 0 reads as "no restriction", not as an empty set
    df = pd.read_pickle(args.sae_file)
    hidden = args.hidden if args.hidden is not None else df.attrs.get("sae_hidden")
    # Read before dedup: df.attrs isn't guaranteed to survive .iloc slicing.
    text_col = embedded_text_col(df, args.text_col)
    _, df, _ = dedup_by_text(None, df, text_col, "corpus")
    print(f"{args.sae_file}: {len(df)} rows, sae_hidden={hidden}")

    if args.where:
        if "=" not in args.where:
            raise SystemExit(f"--where expects COL=VALUE, got {args.where!r}")
        col, val = args.where.split("=", 1)
        if col not in df.columns:
            raise SystemExit(f"--where column {col!r} not in columns: {list(df.columns)}")
        keep = df[col].astype(str) == val
        if not keep.any():
            raise SystemExit(f"--where {args.where}: no rows match "
                             f"(values: {sorted(df[col].astype(str).unique())[:12]})")
        df = df[keep]
        print(f"  --where {args.where}: kept {len(df)} rows")

    # Stats on the full corpus, matching select_neurons.py.
    groups: list[tuple[str, str, np.ndarray, str]] = []
    rank_note = ""
    neurons = list(args.color_by_neuron or [])
    if neurons or args.auto_neurons:
        Z = load_space(df, SPARSE_COL, hidden)
        tfidf = vocab = None
        if text_col in df.columns:
            tfidf, vocab = fit_tfidf(df[text_col].astype(str).values,
                                    stop_words=stopwords_for(args.language),
                                    boilerplate_min_doc_frac=args.boilerplate_min_doc_frac)

        # Same selection as select_neurons.py.
        cols_tfidf = neuron_fired_columns(topn_fired_mask(Z, top_n) if top_n is not None else Z)

        def keyfunc(j):
            return {t.strip() for t in top_terms_for_mask(
                tfidf, vocab, neuron_fired_mask(cols_tfidf, int(j)), args.n_terms).split(",")}

        if args.auto_neurons:
            fr = neuron_firing_stats(Z)
            band = np.flatnonzero((fr >= args.fr_min) & (fr <= args.fr_max))
            if not band.size:
                raise SystemExit(f"no neuron has firing rate in [{args.fr_min}, {args.fr_max}]")
            score, rank_desc = rank_by_score(Z, df, args.rank_by, args.theme_col, top_n)
            pool, keep_n = band[np.argsort(-score[band])], args.auto_neurons
            print(f"  {band.size} candidates (firing rate in "
                  f"[{args.fr_min:.1%}, {args.fr_max:.1%}]), ranked by {rank_desc}")
        else:
            pool, keep_n, rank_desc = np.array(neurons), len(neurons), "--color-by-neuron's order"

        bad = [int(j) for j in pool if not 0 <= int(j) < Z.shape[1]]
        if bad:
            raise SystemExit(f"neuron index/indices {bad} out of range for hidden={Z.shape[1]}")
        if args.diversity_overlap is not None and tfidf is None:
            raise SystemExit(f"--diversity-overlap compares neurons' top TF-IDF terms, but there "
                             f"is no {text_col!r} column to fit them on")
        kept, skipped = select_diverse_neurons(pool, keyfunc, keep_n, args.diversity_overlap)
        report_selection(kept, skipped, len(pool), rank_desc, args.diversity_overlap, "top-terms",
                         cap_flag="--auto-neurons" if args.auto_neurons else "--color-by-neuron")
        neurons = [int(j) for j in kept]

        A = Z[:, neurons]
        A = np.asarray(A.todense() if sp.issparse(A) else A)      # (N, len(neurons)), tiny
        # Points drawn for a neuron are the docs its top terms were computed over.
        eligible = np.column_stack([neuron_fired_mask(cols_tfidf, j) for j in neurons])
        rank_note = f", top-{top_n} activating docs" if top_n is not None else ""
        colors = distinct_colors(len(neurons))

        def label(mask, j):
            terms = (top_terms_for_mask(tfidf, vocab, mask, n_terms=args.n_terms)
                     if tfidf is not None and mask.any() else f"(no {text_col!r} column)")
            print(f"  neuron {j}: {int(mask.sum())}/{len(df)} rows "
                  f"({100 * mask.mean():.1f}%) -- {terms}")
            return terms

        if len(neurons) == 1:                                     # variant 1: fires / doesn't
            fires = eligible[:, 0]
            groups.append((f"n{neurons[0]} fires", f": {label(fires, neurons[0])}", fires, colors[0]))
            groups.append(("does not fire", "", ~fires, "#c8c8c8"))
        else:                                                     # variant 2: strongest-firing
            any_fires = eligible.any(axis=1)
            # argmax over the eligible ones only.
            winner = np.where(eligible, A, -np.inf).argmax(axis=1)
            for i, j in enumerate(neurons):
                mask = any_fires & (winner == i)
                if mask.any():
                    groups.append((f"n{j}", f": {label(mask, j)}", mask, colors[i]))
            if (~any_fires).any():
                groups.append(("none of these fire", "", ~any_fires, "#c8c8c8"))
    elif args.color_by:
        if args.color_by not in df.columns:
            raise SystemExit(f"--color-by {args.color_by!r} not in columns: {list(df.columns)}")
        lab = np.array([("unlabelled" if is_missing_label(v) else str(v))
                        for v in df[args.color_by].values])
        counts = {v: int((lab == v).sum()) for v in set(lab) - {"unlabelled"}}
        keep = [v for v, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:args.max_classes]]
        colors = distinct_colors(len(keep))
        for i, v in enumerate(keep):
            groups.append((str(v), "", lab == v, colors[i]))
        rest = ~np.isin(lab, keep)
        if rest.any():
            name = "unlabelled" if set(lab[rest]) == {"unlabelled"} else "other"
            groups.append((name, "", rest, "#c8c8c8"))
    else:
        groups.append(("", "", np.ones(len(df), dtype=bool), PALETTE[1]))

    theme_vals_full = None
    if args.theme_col:
        if args.theme_col not in df.columns:
            raise SystemExit(f"--theme-col {args.theme_col!r} not in columns: {list(df.columns)}")
        theme_vals_full = df[args.theme_col].astype(str).values

    # Subsample only now, for the projection.
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(df))
    if len(idx) > args.max_rows:
        idx = np.sort(rng.choice(idx, args.max_rows, replace=False))
        print(f"  subsampled to {len(idx)} rows for the projection (--max-rows; "
              f"stats above are over all {len(df)})")
    df_sub = df.iloc[idx]
    # "<drawn> of <full corpus>".
    groups = [("" if not head else f"{head} ({int(mask[idx].sum())} of {int(mask.sum())}){tail}",
              mask[idx], color) for head, tail, mask, color in groups]
    theme_vals = theme_vals_full[idx] if theme_vals_full is not None else None

    n_legend_rows = -(-len(groups) // (min(len(groups), 5) if args.color_by else 1))
    axes_h = 4.8
    legend_h = 0.32 * n_legend_rows + 0.55 if len(groups) > 1 or theme_vals is not None else 0.0
    fig_h = axes_h + legend_h + 0.25              # +0.25in: dedicated strip for the model footer

    apply_style()
    fig, axes = plt.subplots(1, len(args.emb_type),
                             figsize=(4.6 * len(args.emb_type), fig_h), squeeze=False)
    for ax, emb in zip(axes[0], args.emb_type):
        X = load_space(df_sub, emb, hidden)
        if not args.no_normalize:
            X = normalize(X)                      # works on CSR in place of densifying
        XY = project(X, args.method, args.pre_dims, args.seed, args.perplexity)
        for name, mask, color in groups:
            if mask.any():
                ax.scatter(XY[mask, 0], XY[mask, 1], s=4, c=color, linewidths=0,
                           alpha=0.65, label=name or None)
        if theme_vals is not None:
            # Median, not mean.
            for t in np.unique(theme_vals):
                m = theme_vals == t
                ax.annotate(t, np.median(XY[m], axis=0), ha="center", va="center", fontsize=7,
                            color=INK, zorder=6,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
        dim = X.shape[1]
        ax.set_title(f"{emb}\n({dim} dims)", color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        print(f"  projected {emb!r}: {X.shape} -> 2D via {args.method}")

    # Legend/footer get reserved inches (legend_h, fig_h), not a fraction of a fixed figure.
    h, l = axes[0][0].get_legend_handles_labels()
    if l:
        title = (args.color_by if args.color_by else
                 f"SAE neuron{rank_note} (top-{args.n_terms} TF-IDF terms of the rows it fires on)"
                 if (neurons or args.auto_neurons) else None)
        ncol = min(len(l), 5) if args.color_by else 1
        fig.tight_layout(rect=(0, (legend_h + 0.25) / fig_h, 1, 1))
        fig.legend(h, l, loc="lower center", ncol=ncol, markerscale=3,
                   bbox_to_anchor=(0.5, 0.25 / fig_h), title=title)
    else:
        fig.tight_layout(rect=(0, 0.25 / fig_h, 1, 1))
    norm_note = "" if args.no_normalize else ", L2-normalized rows"
    fig.suptitle(f"{args.method.upper()} projection, {len(df_sub)} documents{norm_note}",
                 color=INK, y=1.02)
    add_model_footer(fig, args.model_name)
    save(fig, args.stem or f"2d_{args.method}", args.output_dir)


if __name__ == "__main__":
    main()
