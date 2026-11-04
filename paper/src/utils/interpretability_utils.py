"""Reusable interpretability computations shared by src/interpretability's report scripts: loading
a SAE-encoded corpus (convert_embeddings.py's "*_with_sae.pkl", dense or sparse), theme assignment
from a native category column, per-neuron theme entropy and per-neuron firing rate.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

STOPWORDS_EN = sorted(ENGLISH_STOP_WORDS)

# scikit-learn ships no French stopword list; mixed-language corpora union several lists.
STOPWORDS_FR = sorted({
    "au", "aux", "avec", "ce", "ces", "cette", "cet", "dans", "de", "des", "du", "elle",
    "elles", "en", "es", "est", "et", "eu", "eux", "il", "ils", "je", "j", "l", "la", "le",
    "les", "leur", "leurs", "lui", "ma", "mais", "me", "mes", "moi", "mon", "ne", "nos",
    "notre", "nous", "on", "ont", "ou", "par", "pas", "plus", "pour", "qu", "que", "qui",
    "sa", "se", "ses", "son", "sont", "sur", "ta", "te", "tes", "toi", "ton", "tu", "un",
    "une", "vos", "votre", "vous", "c", "d", "n", "s", "t", "y", "été", "être", "avoir",
    "fait", "faire", "cela", "ceci", "alors", "aussi", "ainsi", "donc", "comme", "si", "quoi",
})

STOPWORDS_BY_LANGUAGE = {"en": STOPWORDS_EN, "fr": STOPWORDS_FR}

# Chunks of one document that may appear together in a neuron's evidence or the eval pool.
MAX_CHUNKS_PER_DOC = 2


def stopwords_for(languages: list[str]) -> list[str]:
    """Union of the stopword lists for the given language codes, for mixed-language corpora."""
    return sorted(set().union(*(STOPWORDS_BY_LANGUAGE[lang] for lang in languages)))


def sparse_codes_from_column(col: np.ndarray, hidden: int) -> sp.csr_matrix:
    """(indices, values) tuples per row (convert_embeddings.py's default TopK storage) -> CSR."""
    idx = [np.asarray(t[0], dtype=np.int32) for t in col]
    val = [np.asarray(t[1], dtype=np.float32) for t in col]
    indptr = np.concatenate([[0], np.cumsum([len(i) for i in idx])])
    return sp.csr_matrix((np.concatenate(val), np.concatenate(idx), indptr), shape=(len(col), hidden))


def dense_codes_from_column(col) -> np.ndarray:
    return np.vstack(col.apply(np.asarray).values).astype(np.float32)


def load_sae_codes(sae_corpus_file: Path, hidden: int | None = None):
    """Load a convert_embeddings.py '*_with_sae.pkl' — returns (Zc, df). Zc is (N, H): dense, or CSR
    for the sparse per-row storage used by TopK checkpoints. H comes from
    ``df.attrs["sae_hidden"]`` or an explicit ``hidden``, never from the largest fired index.
    """
    df = pd.read_pickle(sae_corpus_file)
    if "sae embeddings" not in df.columns:
        raise ValueError(f"{sae_corpus_file} has no 'sae embeddings' column (found: {list(df.columns)})")
    col = df["sae embeddings"]
    if not isinstance(col.iloc[0], tuple):
        return dense_codes_from_column(col), df
    hidden = hidden if hidden is not None else df.attrs.get("sae_hidden")
    if hidden is None:
        raise ValueError(f"{sae_corpus_file} stores sparse SAE codes but carries no "
                         f"attrs['sae_hidden'] — re-run src.training.convert_embeddings on it, "
                         f"or pass --hidden explicitly")
    return sparse_codes_from_column(col.values, hidden), df


def _fired_and_n_fired(Zc):
    """(fired mask, n_fired per neuron) — Zc > 0, dense or sparse. Not != 0, since topknorelu keeps
    signed pre-activations and a selected slot can be negative."""
    if sp.issparse(Zc):
        fired = Zc > 0
        return fired, np.asarray(fired.sum(axis=0)).ravel().astype(np.int64)
    fired = (Zc > 0)
    return fired, fired.sum(0).astype(np.int64)


def neuron_firing_stats(Zc) -> np.ndarray:
    """Per-neuron firing rate: fraction of the N documents each neuron fires on (Zc > 0; see
    _fired_and_n_fired)."""
    N = Zc.shape[0]
    _, n_fired = _fired_and_n_fired(Zc)
    return n_fired / N


def neuron_fired_counts(Zc) -> np.ndarray:
    """Per-neuron count of the documents each neuron fires on (Zc > 0; see _fired_and_n_fired)."""
    return _fired_and_n_fired(Zc)[1]


def neuron_activation_mass(Zc) -> np.ndarray:
    """Per-neuron activation mass: sum of raw activation values across all documents, dense or
    sparse. Sensitive to activation magnitude, unlike firing rate."""
    if sp.issparse(Zc):
        return np.asarray(Zc.sum(axis=0)).ravel().astype(np.float64)
    return Zc.sum(0).astype(np.float64)


def is_missing_label(label) -> bool:
    """True for a label value that means "no theme/category" — None, NaN, or an empty list
    (a list-valued theme column represents an unlabelled doc as ``[]``, not None/NaN)."""
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return True
    return isinstance(label, (list, tuple, set)) and len(label) == 0


def neuron_fired_columns(Zc):
    """Zc in a column-sliceable form for repeated per-neuron firing-mask queries via
    neuron_fired_mask (CSC for sparse; dense arrays are already column-sliceable)."""
    return Zc.tocsc() if sp.issparse(Zc) else Zc


def neuron_fired_mask(Zc_cols, neuron_idx: int) -> np.ndarray:
    """Boolean (N,) mask of which docs fire on neuron `neuron_idx` (> 0). Pass Zc through
    neuron_fired_columns() first when querying many neurons."""
    if sp.issparse(Zc_cols):
        return np.asarray(Zc_cols[:, neuron_idx].todense()).ravel() > 0
    return Zc_cols[:, neuron_idx] > 0


def neuron_top_activating(Zc_cols, neuron_idx: int, n: int) -> np.ndarray:
    """Indices of the (up to) n documents with the highest activation for a neuron, descending,
    capped at how many are > 0. Pass Zc through neuron_fired_columns() first.
    """
    col = (np.asarray(Zc_cols[:, neuron_idx].todense()).ravel()
          if sp.issparse(Zc_cols) else Zc_cols[:, neuron_idx])
    k = min(n, int((col > 0).sum()))
    if k == 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-col, k - 1)[:k]
    return idx[np.argsort(-col[idx])]


def topn_fired_mask(Zc, top_n: int):
    """Per-neuron top-N-by-activation fired mask (N, H), same shape and sparsity kind as Zc, keeping
    only each neuron's strongest activations.
    """
    N, H = Zc.shape
    Zc_cols = neuron_fired_columns(Zc)
    rows_out, cols_out = [], []
    for j in range(H):
        idx = neuron_top_activating(Zc_cols, j, top_n)
        rows_out.append(idx)
        cols_out.append(np.full(len(idx), j, dtype=idx.dtype))
    rows = np.concatenate(rows_out) if rows_out else np.array([], dtype=int)
    cols = np.concatenate(cols_out) if cols_out else np.array([], dtype=int)
    if sp.issparse(Zc):
        data = np.ones(len(rows), dtype=bool)
        return sp.csr_matrix((data, (rows, cols)), shape=(N, H))
    mask = np.zeros((N, H), dtype=bool)
    mask[rows, cols] = True
    return mask


def neuron_entropy_score(Zc: np.ndarray, df: pd.DataFrame, theme_col: str | None,
                         top_n: int | None = None) -> np.ndarray:
    """Negative theme entropy per neuron: higher = more concept-pure. ``top_n`` restricts each
    neuron's fired set to its strongest activations. Raises ValueError if ``theme_col`` is
    missing.
    """
    if not theme_col or theme_col not in df.columns:
        raise ValueError(f"--rank-by entropy requires --theme-col, naming one of {list(df.columns)} "
                         f"— got {theme_col!r}")
    raw_labels = df[theme_col].tolist()
    matched = np.array([not is_missing_label(l) for l in raw_labels])
    onehot, _ = onehot_from_labels([l for l, m in zip(raw_labels, matched) if m])
    _, ent_norm, _, _, _ = neuron_theme_entropy(Zc[matched], onehot, top_n=top_n)
    return -ent_norm


def rank_by_score(Zc: np.ndarray, df: pd.DataFrame, rank_by: str, theme_col: str | None,
                  top_n: int | None = None) -> tuple[np.ndarray, str]:
    """(per-neuron score, human-readable description) for --rank-by mass/entropy, higher = better.
    `top_n` affects the entropy mode only; mass ranks on every fired doc."""
    if rank_by == "mass":
        return neuron_activation_mass(Zc), "highest in-domain activation mass"
    if rank_by == "entropy":
        return neuron_entropy_score(Zc, df, theme_col, top_n), "lowest in-domain theme entropy"
    raise ValueError(f"--rank-by must be 'mass' or 'entropy', got {rank_by!r}")


def select_diverse_neurons(neurons_sorted: np.ndarray, keyfunc, max_neurons: int | None,
                           diversity_overlap: float | None) -> tuple[list[int], list[tuple[int, int, float]]]:
    """Greedy top-`max_neurons` pick from a score-sorted neuron list (None keeps all). A candidate is
    skipped once its Jaccard similarity to a kept neuron's `keyfunc` set reaches
    `diversity_overlap`. Returns (kept ids, skipped tuples).
    """
    if diversity_overlap is None:
        return neurons_sorted[:max_neurons].tolist(), []
    kept, keys, skipped = [], [], []
    for n in neurons_sorted:
        key = keyfunc(n)
        dup = next(((kn, len(key & kk) / max(1, len(key | kk)))
                    for kn, kk in zip(kept, keys)
                    if len(key & kk) / max(1, len(key | kk)) >= diversity_overlap), None)
        if dup is not None:
            skipped.append((n, *dup))
            continue
        kept.append(n)
        keys.append(key)
        if len(kept) == max_neurons:
            break
    return kept, skipped


def warn_if_tied_ranking(score: np.ndarray, neurons_sorted: np.ndarray, max_neurons: int | None,
                         score_desc: str = "score", knob_hint: str | None = None,
                         decimals: int = 6) -> int:
    """Print (don't raise) when score can't separate the neurons at the max_neurons cut. ``decimals``
    rounds before comparing. Returns how many candidates share the boundary score.
    """
    if len(neurons_sorted) == 0:
        return 0
    hint = f" Change {knob_hint} first." if knob_hint else ""
    rounded = np.round(np.asarray(score)[neurons_sorted], decimals)
    cut = (len(rounded) if max_neurons is None else min(max_neurons, len(rounded))) - 1
    kept = rounded[:cut + 1]

    # Failure mode 1: the tie straddles the cut.
    tied_at_cut = int((rounded == rounded[cut]).sum())
    if tied_at_cut > int((kept == rounded[cut]).sum()):
        print(f"WARNING: {score_desc} ties {tied_at_cut} candidates at {rounded[cut] + 0.0:.6g}, more than "
              f"fit under --max-neurons {max_neurons} — which of the tied neurons you get is "
              f"arbitrary, not a ranking.{hint}")

    # Failure mode 2: the score saturates inside the cut.
    values, counts = np.unique(kept, return_counts=True)
    biggest = int(counts.max())
    if biggest > 1 and biggest >= max(2, len(kept) // 2):
        print(f"WARNING: {biggest} of the {len(kept)} kept neurons share one {score_desc} value "
              f"({values[int(counts.argmax())] + 0.0:.6g}) — the score saturated and did not rank them."
              f"{hint}")
    return tied_at_cut


def report_selection(kept: list[int], skipped: list[tuple[int, int, float]], n_candidates: int,
                     rank_desc: str, diversity_overlap: float | None, dup_label: str,
                     cap_flag: str = "--max-neurons") -> None:
    """No-silent-truncation summary for select_diverse_neurons' result: what was kept, what was
    skipped as a near-duplicate, and what was never reached because of --max-neurons."""
    for n, kn, overlap in skipped:
        print(f"  #{n} skipped as near-duplicate of #{kn} ({dup_label} Jaccard={overlap:.2f})")
    n_not_reached = n_candidates - len(kept) - len(skipped)
    bits = ([f"{len(skipped)} near-duplicates ({dup_label} Jaccard >= {diversity_overlap:.0%})"]
           if skipped else []) + \
          ([f"{n_not_reached} not reached (raise {cap_flag})"] if n_not_reached > 0 else [])
    if bits:
        print(f"keeping {len(kept)} candidates with {rank_desc}; dropped " + " and ".join(bits))


def strip_boilerplate_lines(texts: np.ndarray, min_doc_frac: float) -> np.ndarray:
    """Drop lines that verbatim-recur across many documents (banners, nav templates, citation
    footers) before TF-IDF. A line counts once per document regardless of in-doc repeats."""
    n = len(texts)
    threshold = max(2, int(min_doc_frac * n))
    line_doc_count = Counter()
    for t in texts:
        line_doc_count.update(set(line.strip() for line in str(t).split("\n") if line.strip()))
    boilerplate = {line for line, c in line_doc_count.items() if c >= threshold}
    if boilerplate:
        n_affected = sum(1 for t in texts
                         if boilerplate & set(line.strip() for line in str(t).split("\n")))
        print(f"[fit_tfidf] stripped {len(boilerplate)} recurring boilerplate line(s) "
              f"(>= {min_doc_frac:.1%} of {n} docs) from {n_affected} docs")
    return np.array(["\n".join(line for line in str(t).split("\n") if line.strip() not in boilerplate)
                     for t in texts])


def fit_tfidf(texts: np.ndarray, max_features: int = 20000, min_df: int = 3,
              stop_words="english", boilerplate_min_doc_frac: float | None = None):
    """Fit a TF-IDF vectorizer once over a corpus, returning (tfidf matrix, vocab array) for repeated
    top_terms_for_mask calls. `stop_words` is anything TfidfVectorizer accepts;
    `boilerplate_min_doc_frac` enables strip_boilerplate_lines."""
    if boilerplate_min_doc_frac is not None:
        texts = strip_boilerplate_lines(texts, boilerplate_min_doc_frac)
    vec = TfidfVectorizer(max_features=max_features, stop_words=stop_words, ngram_range=(1, 2),
                          token_pattern=r"(?u)\b[^\d\W][^\d\W]+\b", min_df=min_df)
    tfidf = vec.fit_transform(texts)
    return tfidf, np.array(vec.get_feature_names_out())


def top_terms_for_mask(tfidf, vocab: np.ndarray, mask: np.ndarray, n_terms: int = 6,
                       empty_label: str = "(no docs)") -> str:
    """Top n_terms TF-IDF terms (by mean weight) among the rows selected by `mask`."""
    rows = tfidf[mask]
    if rows.shape[0] == 0:
        return empty_label
    mean_tfidf = np.asarray(rows.mean(axis=0)).ravel()
    return ", ".join(vocab[np.argsort(-mean_tfidf)[:n_terms]])


def onehot_from_labels(labels: list) -> tuple[np.ndarray, np.ndarray]:
    """Native theme/category labels -> (N, K) one-hot, K theme names. A multi-label doc splits its
    weight evenly; an unlabelled doc gets an all-zero row.
    """
    names = sorted({l for row in labels for l in (row if isinstance(row, (list, tuple, set)) else [row])
                    if row is not None})
    idx = {n: i for i, n in enumerate(names)}
    onehot = np.zeros((len(labels), len(names)), dtype=np.float32)
    for i, row in enumerate(labels):
        row_labels = [l for l in (row if isinstance(row, (list, tuple, set)) else [row]) if l is not None]
        for l in row_labels:
            onehot[i, idx[l]] = 1.0 / len(row_labels)
    return onehot, np.array(names, dtype=object)


def dedup_by_text(Zc, df: pd.DataFrame, text_col: str, msg_prefix: str):
    """Drop exact-duplicate text_col rows. Zc may be None if not loaded yet. Returns (Zc, df,
    n_dropped), df re-indexed."""
    texts = df[text_col].astype(str).values
    N0 = len(df)
    _, keep = np.unique(texts, return_index=True)
    keep = np.sort(keep)
    df = df.iloc[keep].reset_index(drop=True)
    if Zc is not None:
        Zc = Zc[keep]
    print(f"{msg_prefix} docs {N0} -> {len(keep)} after dropping {N0 - len(keep)} "
          f"exact-duplicate passages")
    return Zc, df, int(N0 - len(keep))


def theme_entropy_from_labels(Zc, df: pd.DataFrame, theme_col: str, msg_prefix: str,
                              top_n: int | None = None):
    """Per-neuron theme entropy from a native theme_col, dropping unlabelled docs first. Expects Zc/df
    already deduplicated. Returns (ent, ent_norm, n_fired, firing_rate, theme_dist, theme_sizes,
    theme_labels, n_docs) — see neuron_theme_entropy for the first five.
    """
    raw_labels = df[theme_col].tolist()
    matched = np.array([not is_missing_label(l) for l in raw_labels])
    onehot, theme_names = onehot_from_labels([l for l, m in zip(raw_labels, matched) if m])
    Zc = Zc[matched]
    print(f"{msg_prefix} docs {matched.size} -> {matched.sum()} after dropping "
          f"{int((~matched).sum())} with no native '{theme_col}' label")
    ent, ent_norm, n_fired, firing_rate, theme_dist = neuron_theme_entropy(Zc, onehot, top_n=top_n)
    return (ent, ent_norm, n_fired, firing_rate, theme_dist, onehot.sum(0), list(theme_names),
            int(onehot.shape[0]))


def neuron_theme_entropy(Zc, onehot: np.ndarray, top_n: int | None = None):
    """Per-neuron entropy over the themes of the documents each neuron fires on (Zc > 0).

    ``Zc`` is (N, H), dense or sparse; ``onehot`` is (N, K). ``top_n`` restricts each neuron's
    fired set to its top_n highest-activation docs, affecting entropy/entropy_norm/theme_dist only.
    Entropy is over P(fires | theme=k), renormalized across k. Low entropy = firing concentrated
    in one theme. Returns (entropy, entropy_norm, n_fired, firing_rate, theme_dist) with theme_dist
    (H, K).
    """
    N, K = onehot.shape
    if K <= 1:
        # K<2 makes every neuron tie at ent=0.
        raise ValueError(f"theme entropy needs > 1 theme, got {K} — check --theme-col's values")
    fired, n_fired = _fired_and_n_fired(Zc)
    if top_n is not None:
        # Restricts entropy/theme_dist only: n_fired and firing_rate stay on the full fired set.
        fired = topn_fired_mask(Zc, top_n)
    theme_counts = np.asarray(fired.T.astype(np.float32) @ onehot)

    # P(fires | theme=k), not P(theme=k | fires).
    theme_sizes = onehot.sum(axis=0).astype(np.float32)
    fire_given_theme = theme_counts / np.where(theme_sizes > 0, theme_sizes, 1)[None, :]
    row_sum = fire_given_theme.sum(1)

    live = (n_fired > 0) & (row_sum > 0)
    p = np.zeros_like(fire_given_theme)
    p[live] = fire_given_theme[live] / row_sum[live, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.where(p > 0, p * np.log(p), 0.0).sum(1)
    ent[~live] = np.nan
    # K=1: ent is already 0 for live neurons and nan for dead ones, so copy it as-is.
    ent_norm = ent.copy() if K == 1 else ent / np.log(K)

    firing_rate = n_fired / N
    return ent, ent_norm, n_fired, firing_rate, p
