"""Loaders that rebuild the benchmarks of the paper from their Hugging Face Hub files, and BEIR folders.

Every Hub loader downloads raw files at a pinned revision with :func:`huggingface_hub.hf_hub_download`
(cached, so it works offline once downloaded) and applies the recipe of the paper code
(``paper/src/data_prep``):

* the embedded document string is ``strip(strip(title) + " " + strip(text))`` when the source has a title
  field and the raw text otherwise (``resolve_embed_column`` with its default ``--prepend-col title``);
* queries are embedded from their raw text;
* every id is a string, and a repeated document id keeps its first row, as the paper evaluation does;
* split names are ``train``, ``val`` and ``test`` (``dev``, ``validation``, ``valid`` and ``eval`` become
  ``val``).

Every corpus frame has the columns ``doc_id``, ``text`` (the embedded string), ``title`` (empty when the
source has none) and ``body`` (the source text), plus theme columns such as ``source_folder`` for CURE.
"""

from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from doctosae.data.dataset import RetrievalDataset
from doctosae.data.splits import ids_sha256, split_corpus, split_queries

if TYPE_CHECKING:
    from doctosae.data.registry import DatasetInfo

__all__ = ["CURE_DISCIPLINES", "load_beir_folder", "load_cure", "load_miriad", "load_nfcorpus", "load_r2med"]

logger = logging.getLogger(__name__)

_SPLIT_ALIASES = {"dev": "val", "validation": "val", "valid": "val", "eval": "val"}

# The ten CURE disciplines in the order of the dataset card (alphabetical). This order fixes the row order of
# the corpus, the queries and the qrels, hence the query split: it is hard coded, not read from a library.
CURE_DISCIPLINES = (
    "dentistry_and_oral_health",
    "dermatology",
    "gastroenterology",
    "genetics",
    "neuroscience_and_neurology",
    "orthopedic_surgery",
    "otorhinolaryngology",
    "plastic_surgery",
    "psychiatry_and_psychology",
    "pulmonology",
)

# Fingerprints (doctosae.data.splits.ids_sha256) of the CURE split the paper trained and evaluated on,
# recomputed from the raw files at the pinned revision. The split rests on the default_rng stream of numpy,
# which NumPy does not promise to keep across releases, so every load at the paper settings is checked.
_CURE_PAPER_SPLIT = {
    "queries train": "358745a89f499b1ba50d7a0886a19eee4d8d62bd26b2744afc2ab7a3a3c15914",
    "queries val": "9f3b59ff7d545b24d10b28f7d7d466e9ba9e6176df8b1b805dff20cd27e9501c",
    "queries test": "e81ded71676166bfc3bda7d3b18028b29323417ed910f31fd42c5fbb612092bf",
    "corpus train": "33602e0ffd342bedfb07406044d6f2a8b982da930a969bfb1dd4daf72b082bd0",
    "corpus val": "74439a9cbcbf108d83e8f039bdc2b82749f36111eb180fef6f2dfeac87ac8141",
}

_NFCORPUS_CORPUS_FILE = "corpus/corpus-00000-of-00001.parquet"
_NFCORPUS_QUERIES_FILE = "queries/queries-00000-of-00001.parquet"
_NFCORPUS_QRELS_FILES = (("train", "train.tsv"), ("val", "dev.tsv"), ("test", "test.tsv"))

_MIRIAD_TEST_FILE = "data/test-00000-of-00001.parquet"
# The first train shard holds 106,371 rows and the train split starts with them (data files resolve in
# lexicographic order), so the 30,000 background passages never need the 41 other shards (12 GB).
_MIRIAD_TRAIN_SHARD = "data/train-00000-of-00042.parquet"

_QRELS_COLUMNS = {
    "query_id": ("query-id", "query_id", "qid", "q_id", "query"),
    "doc_id": ("corpus-id", "corpus_id", "doc-id", "doc_id", "docid", "doc", "p_id", "pid", "passage"),
    "score": ("score", "relevance", "rel", "grade", "label"),
}


# Generic helpers.


def _hub_file(
    repo_id: str, filename: str, revision: str, cache_dir: str | Path | None, token: str | bool | None
) -> Path:
    """Local path of one file of a Hub dataset repository at ``revision`` (downloaded once, then cached)."""
    return Path(
        hf_hub_download(
            repo_id, filename, repo_type="dataset", revision=revision, cache_dir=cache_dir, token=token
        )
    )


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _text(value: Any) -> str:
    """A missing text becomes the empty string (the paper's ``fillna("")``)."""
    return "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)


def _with_title(title: Any, body: Any) -> str:
    """The string the paper embedded for a document with a title field (``resolve_embed_column``)."""
    return f"{_text(title).strip()} {_text(body).strip()}".strip()


def _split_name(name: str) -> str:
    name = name.strip().lower()
    return _SPLIT_ALIASES.get(name, name)


def _corpus_frame(
    name: str,
    doc_ids: Sequence[Any],
    bodies: Sequence[Any],
    titles: Sequence[Any] | None = None,
    extra: Mapping[str, Sequence[Any]] | None = None,
) -> pd.DataFrame:
    """Corpus frame holding the embedded string in ``text``; ``titles=None`` means no title field."""
    bodies = [_text(body) for body in bodies]
    if titles is None:
        titles = [""] * len(bodies)
        texts = list(bodies)
    else:
        titles = [_text(title) for title in titles]
        texts = [_with_title(title, body) for title, body in zip(titles, bodies, strict=True)]
    frame = pd.DataFrame(
        {
            "doc_id": [str(doc_id) for doc_id in doc_ids],
            "text": texts,
            "title": titles,
            "body": bodies,
            **(extra or {}),
        }
    )
    return _first_rows(frame, "doc_id", f"{name} corpus")


def _first_rows(frame: pd.DataFrame, id_column: str, what: str) -> pd.DataFrame:
    """Keep the first row of every id: the paper evaluation scores one row per id and drops the others."""
    repeated = frame[id_column].duplicated(keep="first")
    if repeated.any():
        logger.warning("%s: dropping %d rows whose id repeats an earlier row", what, repeated.sum())
        frame = frame[~repeated].reset_index(drop=True)
    return frame


def _qrels_frame(rows: Sequence[tuple[Any, Any, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": [str(row[0]) for row in rows],
            "doc_id": [str(row[1]) for row in rows],
            "score": np.asarray([int(row[2]) for row in rows], dtype=np.int64),
        }
    )


def _read_qrels_tsv(path: Path) -> pd.DataFrame:
    """Qrels with columns ``query_id``, ``doc_id`` and ``score`` from a BEIR style TSV.

    The header may use the BEIR names (``query-id``, ``corpus-id``, ``score``), ``relevance`` for the grade
    (as the paper's MIRIAD files do) or a few common variants. A file without a recognised header is read
    as ``query, doc, score`` (three columns) or as TREC ``query, iteration, doc, score`` (four columns).
    """
    frame = _read_table(path, header=0)
    lowered = {str(column).strip().lower(): column for column in frame.columns}
    found = {
        target: next((lowered[alias] for alias in aliases if alias in lowered), None)
        for target, aliases in _QRELS_COLUMNS.items()
    }
    if all(found.values()):
        frame = frame[[found["query_id"], found["doc_id"], found["score"]]]
    else:
        frame = _read_table(path, header=None)
        if frame.shape[1] not in (3, 4):
            raise ValueError(f"{path}: expected a query-id, corpus-id, score header or 3 to 4 columns")
        frame = frame[[0, 1, 2]] if frame.shape[1] == 3 else frame[[0, 2, 3]]
        if len(frame) and pd.isna(pd.to_numeric(frame.iloc[0, 2], errors="coerce")):
            accepted = "; ".join(
                f"{target}: {', '.join(aliases)}" for target, aliases in _QRELS_COLUMNS.items()
            )
            raise ValueError(
                f"{path}: the header {list(lowered)} is not recognised; accepted names are {accepted}"
            )
    frame.columns = ["query_id", "doc_id", "score"]
    grades = pd.to_numeric(frame["score"])
    fractional = grades[grades != np.floor(grades)]
    if len(fractional):
        raise ValueError(
            f"{path}: relevance grades must be integers, got {float(fractional.iloc[0])!r}; round them "
            "before loading rather than have them truncated"
        )
    frame["score"] = grades.astype(np.int64)
    return frame.reset_index(drop=True)


def _read_table(path: Path, header: int | None) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", header=header, dtype=str, keep_default_na=False)
    if frame.shape[1] < 3:  # whitespace separated, retried as the paper's load_qrels_matrix does
        frame = pd.read_csv(
            path, sep=r"\s+", header=header, dtype=str, keep_default_na=False, engine="python"
        )
    return frame


def _parquet_head(path: Path, n_rows: int, columns: list[str]) -> pd.DataFrame:
    """The first ``n_rows`` rows of a parquet file, reading only the row groups and columns needed."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows < n_rows:
        raise ValueError(f"{path.name} has {parquet.metadata.num_rows:,} rows, {n_rows:,} requested")
    if n_rows == 0:
        return pd.DataFrame({column: pd.Series(dtype=object) for column in columns})
    batches, total = [], 0
    for batch in parquet.iter_batches(batch_size=min(n_rows, 10_000), columns=columns):
        batches.append(batch)
        total += batch.num_rows
        if total >= n_rows:
            break
    return pa.Table.from_batches(batches).slice(0, n_rows).to_pandas()


def _scalar_columns(frame: pd.DataFrame, skip: set[str]) -> list[str]:
    """Columns not in ``skip`` holding only scalars (themes, expertise...); nested fields are dropped."""
    scalar = (str, int, float, bool, type(None))
    return [
        column
        for column in frame.columns
        if column not in skip and frame[column].map(lambda value: isinstance(value, scalar)).all()
    ]


# BEIR folders.


def _beir_file(root: Path, stem: str) -> Path:
    for candidate in (root / f"{stem}.jsonl", root / f"{stem}.jsonl.gz"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{root} has no {stem}.jsonl (BEIR layout: corpus.jsonl, queries.jsonl, qrels/)")


def _id_key(rows: list[dict[str, Any]], path: Path) -> str:
    for key in ("_id", "id"):
        if rows and key in rows[0]:
            return key
    raise ValueError(f"{path}: rows need an '_id' (BEIR) or 'id' field")


def load_beir_folder(path: str | Path, name: str | None = None) -> RetrievalDataset:
    """Load a retrieval dataset stored in the BEIR layout.

    Expected files: ``corpus.jsonl`` (``_id``, optional ``title``, ``text``), ``queries.jsonl`` (``_id``,
    ``text``) and one ``qrels/<split>.tsv`` per split (``query-id``, ``corpus-id``, ``score``); ``.jsonl.gz``
    files work too. Split names are free; ``dev``, ``validation``, ``valid`` and ``eval`` become ``val``. As
    in the paper code, a split holds the queries listed in its qrels file
    (``nfc/filter_queries_by_qrels.py``), documents with a title field are embedded as title and text joined
    by a space, and scalar extra fields (themes...) are kept as columns.

    Args:
        path: Folder holding the files.
        name: Dataset name, the folder name by default.
    """
    root = Path(path).expanduser()
    name = name or root.resolve().name

    corpus_path = _beir_file(root, "corpus")
    corpus_rows = list(_iter_jsonl(corpus_path))
    id_key = _id_key(corpus_rows, corpus_path)
    records = pd.DataFrame(corpus_rows)
    corpus = _corpus_frame(
        name,
        records[id_key].tolist(),
        records["text"].tolist() if "text" in records.columns else [""] * len(records),
        titles=records["title"].tolist() if "title" in records.columns else None,
        extra={
            column: records[column].tolist()
            for column in _scalar_columns(records, {id_key, "title", "text", "doc_id", "body"})
        },
    )

    queries_path = _beir_file(root, "queries")
    query_rows = list(_iter_jsonl(queries_path))
    query_key = _id_key(query_rows, queries_path)
    records = pd.DataFrame(query_rows)
    texts = records["text"].map(_text).tolist()
    if "title" in records.columns:  # the paper's embedding scripts prepend a title column to queries too
        texts = [_with_title(title, text) for title, text in zip(records["title"], texts, strict=True)]
    all_queries = pd.DataFrame({"query_id": records[query_key].astype(str), "text": texts})
    for column in _scalar_columns(records, {query_key, "title", "text", "query_id"}):
        all_queries[column] = records[column]
    all_queries = _first_rows(all_queries, "query_id", f"{name} queries")

    qrels_files = sorted((root / "qrels").glob("*.tsv"))
    if not qrels_files:
        raise FileNotFoundError(f"{root / 'qrels'} holds no <split>.tsv file")
    queries: dict[str, pd.DataFrame] = {}
    qrels: dict[str, pd.DataFrame] = {}
    for qrels_path in qrels_files:
        split = _split_name(qrels_path.stem)
        if split in qrels:
            raise ValueError(f"{root / 'qrels'} has two files for the {split!r} split")
        judged = _read_qrels_tsv(qrels_path)
        members = all_queries[all_queries["query_id"].isin(set(judged["query_id"]))]
        if members.empty:
            logger.warning(
                "%s: no query of %s is in %s, split skipped", name, qrels_path.name, queries_path.name
            )
            continue
        queries[split], qrels[split] = members, judged
    return RetrievalDataset(name=name, corpus=corpus, queries=queries, qrels=qrels)


# Benchmarks of the paper.


def load_cure(
    info: DatasetInfo,
    cache_dir: str | Path | None = None,
    *,
    language: str = "en",
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    corpus_val_frac: float = 0.1,
    seed: int = 42,
    revision: str | None = None,
    token: str | bool | None = None,
) -> RetrievalDataset:
    """CURE, the training set and in domain test set of the paper.

    Recipe of ``cure/prepare_data.py``, ``cure/split_multilang_queries.py`` and ``split_corpus.py``: the
    ten discipline files are read in :data:`CURE_DISCIPLINES` order; the corpus keeps the first row of every
    passage id (244,600 passages) and records that discipline in ``source_folder``; every judgment is kept
    (grades 1 and 2). The queries are split 0.7 / 0.1 / 0.2 by shuffling the English ids in file order with
    ``default_rng(seed)`` (1,400 / 200 / 400). The corpus is split with the positives of the training queries
    forced into train and 10% of the other passages sent to val (223,425 / 21,175 rows, stored in
    ``corpus_splits``). Retrieval always scores the full corpus.

    Args:
        info: Registry entry.
        cache_dir: Hugging Face cache folder.
        language: Query language, ``en`` (the paper), ``es`` or ``fr``. The split is computed on the English
            file order, so every language gets the same query ids in each split.
        train_frac: Fraction of the queries in train.
        val_frac: Fraction of the queries in val; test gets the rest.
        corpus_val_frac: Fraction of the passages not forced into train that go to val.
        seed: Seed of both splits.
        revision: Dataset revision, the pinned one by default.
        token: Hugging Face token, if needed.
    """
    if language not in ("en", "es", "fr"):
        raise ValueError(f"CURE queries exist in en, es and fr, not {language!r}")
    revision = revision or info.revision

    def rows(kind: str) -> Iterator[tuple[str, dict[str, Any]]]:
        for discipline in CURE_DISCIPLINES:
            path = _hub_file(info.hf_id, f"{discipline}/{kind}.jsonl", revision, cache_dir, token)
            for row in _iter_jsonl(path):
                yield discipline, row

    first: dict[str, tuple[Any, Any, str]] = {}
    for discipline, row in rows("corpus"):
        # A passage listed under several disciplines has the same title and text in each; the first wins.
        first.setdefault(str(row["_id"]), (row["title"], row["text"], discipline))
    corpus = _corpus_frame(
        info.name,
        list(first),
        [body for _, body, _ in first.values()],
        titles=[title for title, _, _ in first.values()],
        extra={"source_folder": [folder for _, _, folder in first.values()]},
    )

    all_qrels = _qrels_frame([(row["query-id"], row["corpus-id"], row["score"]) for _, row in rows("qrels")])
    query_rows = list(rows(f"queries-{language}"))
    all_queries = pd.DataFrame(
        {
            "query_id": [str(row["_id"]) for _, row in query_rows],
            "text": [_text(row["text"]) for _, row in query_rows],
            "expertise": [row.get("expertise") for _, row in query_rows],
            "discipline": [discipline for discipline, _ in query_rows],
        }
    )
    if language == "en":
        english_ids = all_queries["query_id"].tolist()
    else:
        english_ids = [str(row["_id"]) for _, row in rows("queries-en")]
    split_rows = split_queries(english_ids, train_frac, val_frac, seed)
    split_ids = {
        split: {english_ids[i] for i in members} for split, members in split_rows.items() if len(members)
    }
    queries = {split: all_queries[all_queries["query_id"].isin(ids)] for split, ids in split_ids.items()}
    qrels = {split: all_qrels[all_qrels["query_id"].isin(ids)] for split, ids in split_ids.items()}

    train_qrels = qrels.get("train", all_qrels.iloc[:0])
    train_positives = train_qrels.loc[train_qrels["score"] > 0, "doc_id"]  # the paper's _required_corpus_ids
    corpus_splits = split_corpus(corpus["doc_id"], train_positives, val_frac=corpus_val_frac, seed=seed)
    if (train_frac, val_frac, corpus_val_frac, seed, revision) == (0.7, 0.1, 0.1, 42, info.revision):
        _check_cure_split(split_ids, corpus["doc_id"].to_numpy(), corpus_splits)
    return RetrievalDataset(
        name=info.name,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        corpus_splits=corpus_splits,
        license=info.license,
        hf_ids=list(info.hf_ids),
    )


def _check_cure_split(
    split_ids: dict[str, set[str]], doc_ids: np.ndarray, corpus_splits: dict[str, np.ndarray]
) -> None:
    found = {f"queries {split}": ids_sha256(ids) for split, ids in split_ids.items()}
    found.update({f"corpus {split}": ids_sha256(doc_ids[rows]) for split, rows in corpus_splits.items()})
    differing = sorted(key for key, digest in _CURE_PAPER_SPLIT.items() if found.get(key) != digest)
    if differing:
        logger.warning(
            "The CURE split differs from the paper split (%s) with numpy %s: numbers are not comparable "
            "with the published ones.",
            ", ".join(differing),
            np.__version__,
        )
    else:
        logger.info(
            "CURE split identical to the paper split (checksums of the query and corpus splits match)"
        )


def load_nfcorpus(
    info: DatasetInfo,
    cache_dir: str | Path | None = None,
    *,
    revision: str | None = None,
    qrels_revision: str | None = None,
    token: str | bool | None = None,
) -> RetrievalDataset:
    """NFCorpus, an out of domain benchmark of the paper (3,633 documents, 323 test queries).

    Corpus and queries come from ``BeIR/nfcorpus``, judgments from ``BeIR/nfcorpus-qrels``. As in
    ``nfc/filter_queries_by_qrels.py``, a split holds the queries listed in its qrels file, and ``dev``
    becomes ``val``. The paper evaluates ``test`` only; ``train`` and ``val`` make NFCorpus a small dataset
    to train on. Documents are embedded as title and text, queries from their raw text (the BEIR files the
    paper read have no query title).
    """
    (corpus_repo, corpus_revision), (qrels_repo, pinned_qrels_revision) = list(info.revisions.items())
    corpus_revision = revision or corpus_revision
    qrels_revision = qrels_revision or pinned_qrels_revision

    def fetch(repo_id: str, filename: str, at: str) -> Path:
        return _hub_file(repo_id, filename, at, cache_dir, token)

    documents = pd.read_parquet(fetch(corpus_repo, _NFCORPUS_CORPUS_FILE, corpus_revision))
    corpus = _corpus_frame(
        info.name, documents["_id"].tolist(), documents["text"].tolist(), titles=documents["title"].tolist()
    )
    records = pd.read_parquet(fetch(corpus_repo, _NFCORPUS_QUERIES_FILE, corpus_revision))
    all_queries = pd.DataFrame({"query_id": records["_id"].astype(str), "text": records["text"].map(_text)})

    queries: dict[str, pd.DataFrame] = {}
    qrels: dict[str, pd.DataFrame] = {}
    for split, filename in _NFCORPUS_QRELS_FILES:
        qrels[split] = _read_qrels_tsv(fetch(qrels_repo, filename, qrels_revision))
        queries[split] = all_queries[all_queries["query_id"].isin(set(qrels[split]["query_id"]))]
    return RetrievalDataset(
        name=info.name,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        license=info.license,
        hf_ids=list(info.hf_ids),
    )


def load_miriad(
    info: DatasetInfo,
    cache_dir: str | Path | None = None,
    *,
    n_queries: int = 1000,
    n_background: int = 30_000,
    revision: str | None = None,
    token: str | bool | None = None,
) -> RetrievalDataset:
    """MIRIAD, an out of domain benchmark of the paper (1,000 queries over 31,000 passages).

    Recipe of ``miriad/prepare_data.py`` for its ``test`` artifacts: query ``test-i`` is the question of row
    ``i`` of the test split and its single positive is the passage of that row (``test-i``); the corpus holds
    these passages followed by the first ``n_background`` passages of the train split (``train-i``). Only the
    test file (27 MB) and the first train shard (282 MB) are downloaded. Duplicate passage texts under
    different ids are kept, as in the paper.
    """
    revision = revision or info.revision
    test_file = _hub_file(info.hf_id, _MIRIAD_TEST_FILE, revision, cache_dir, token)
    test = _parquet_head(test_file, n_queries, ["question", "passage_text"])
    background = pd.DataFrame({"passage_text": []})
    if n_background:
        shard = _hub_file(info.hf_id, _MIRIAD_TRAIN_SHARD, revision, cache_dir, token)
        background = _parquet_head(shard, n_background, ["passage_text"])
    test_ids = [f"test-{i}" for i in range(n_queries)]
    corpus = _corpus_frame(
        info.name,
        test_ids + [f"train-{i}" for i in range(n_background)],
        test["passage_text"].tolist() + background["passage_text"].tolist(),
        titles=[""] * (n_queries + n_background),  # the paper's MIRIAD files carry an empty title field
    )
    queries = {"test": pd.DataFrame({"query_id": test_ids, "text": test["question"].map(_text).tolist()})}
    qrels = {"test": _qrels_frame([(query_id, query_id, 1) for query_id in test_ids])}
    return RetrievalDataset(
        name=info.name,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        license=info.license,
        hf_ids=list(info.hf_ids),
    )


def load_r2med(
    info: DatasetInfo,
    cache_dir: str | Path | None = None,
    *,
    revision: str | None = None,
    token: str | bool | None = None,
) -> RetrievalDataset:
    """One R2MED subset (Biology, Bioinformatics or Medical Sciences), out of domain benchmarks of the paper.

    Reads ``corpus.jsonl``, ``query.jsonl`` and ``qrels.jsonl`` (``q_id``, ``p_id``, ``score``, converted as
    ``r2med/convert_qrels.py`` does). Documents and queries have no title and are embedded from their raw
    text. Repeated document ids (2 in Bioinformatics, 18 in Medical Sciences, with identical texts) keep
    their first row, as the paper evaluation does. Queries keep ``answer``, ``body_system`` and the positive
    ids listed in the query file (``positive_doc_ids``). The qrels are authoritative for evaluation and
    training pairs: the Biology query file lists 372 of its 374 positives. Test split only.
    """
    revision = revision or info.revision

    def read(filename: str) -> list[dict[str, Any]]:
        return list(_iter_jsonl(_hub_file(info.hf_id, filename, revision, cache_dir, token)))

    documents = read("corpus.jsonl")
    corpus = _corpus_frame(info.name, [row["id"] for row in documents], [row["text"] for row in documents])
    query_rows = read("query.jsonl")
    queries = pd.DataFrame(
        {
            "query_id": [str(row["id"]) for row in query_rows],
            "text": [_text(row["text"]) for row in query_rows],
            "answer": [row.get("answer") for row in query_rows],
            "body_system": [row.get("body_system") for row in query_rows],
            "positive_doc_ids": [[str(doc) for doc in row.get("doc_id") or ()] for row in query_rows],
        }
    )
    qrels = _qrels_frame([(row["q_id"], row["p_id"], row["score"]) for row in read("qrels.jsonl")])
    return RetrievalDataset(
        name=info.name,
        corpus=corpus,
        queries={"test": queries},
        qrels={"test": qrels},
        license=info.license,
        hf_ids=list(info.hf_ids),
    )
