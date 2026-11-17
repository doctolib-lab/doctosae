"""Registry of the six retrieval benchmarks of the paper, pinned to the Hub revisions behind its numbers.

``load_dataset("cure")`` rebuilds a benchmark exactly as the paper code did (see
:mod:`doctosae.data.loaders`). The data are downloaded at run time from their official Hub repositories and
never shipped with the package: each benchmark has its own license, printed once per session when it is
first loaded.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from doctosae.data.dataset import RetrievalDataset
from doctosae.data.loaders import load_beir_folder, load_cure, load_miriad, load_nfcorpus, load_r2med

__all__ = [
    "DATASETS",
    "DatasetInfo",
    "get_dataset_info",
    "list_datasets",
    "load_dataset",
    "resolve_dataset_name",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetInfo:
    """What a benchmark is, where it comes from and under which terms it may be used.

    Attributes:
        name: Registry name (``"cure"``, ``"r2med_biology"``...).
        title: Display name.
        revisions: Hub dataset id to its pinned commit, in download order (the first is the main repository).
        license: Hub license id, copied to the model card of SAEs trained on the benchmark.
        terms: Usage terms, printed once per session when the benchmark is first loaded.
        paper_role: How the paper uses the benchmark.
        splits: Query splits returned by the loader.
        n_documents: Documents in the loaded corpus at the pinned revision.
        n_queries: Queries per split at the pinned revision.
        loader: Function building the :class:`RetrievalDataset`, called as
            ``loader(info, cache_dir, **kwargs)``.
        upstream: Hub ids of the datasets the benchmark was derived from, if any.
    """

    name: str
    title: str
    revisions: Mapping[str, str]
    license: str
    terms: str
    paper_role: str
    splits: tuple[str, ...]
    n_documents: int
    n_queries: Mapping[str, int]
    loader: Callable[..., RetrievalDataset] = field(repr=False, compare=False)
    upstream: tuple[str, ...] = ()

    @property
    def hf_id(self) -> str:
        """Main Hub repository."""
        return next(iter(self.revisions))

    @property
    def revision(self) -> str:
        """Pinned commit of the main Hub repository."""
        return self.revisions[self.hf_id]

    @property
    def hf_ids(self) -> tuple[str, ...]:
        """Every Hub dataset the benchmark is built from, for model card metadata."""
        return (*self.revisions, *self.upstream)


_OOD = "out of domain test set (test split only)"


def _r2med(subset: str, revision: str, n_documents: int, n_queries: int) -> DatasetInfo:
    title = f"R2MED {subset.replace('-', ' ')}"
    return DatasetInfo(
        name=f"r2med_{subset.lower().replace('-', '_')}",
        title=title,
        revisions={f"R2MED/{subset}": revision},
        license="cc-by-4.0",
        terms=(
            f"{title} (R2MED/{subset}) is released under CC BY 4.0. Its queries are StackExchange posts "
            "(CC BY SA): attribute both R2MED and StackExchange."
        ),
        paper_role=_OOD,
        splits=("test",),
        n_documents=n_documents,
        n_queries={"test": n_queries},
        loader=load_r2med,
    )


DATASETS: dict[str, DatasetInfo] = {
    info.name: info
    for info in (
        DatasetInfo(
            name="cure",
            title="CURE",
            revisions={"clinia/CUREv1": "072343ca2353c3dc832cf18cf0fef7c9a5a207b7"},
            license="cc-by-nc-4.0",
            terms=(
                "CURE (clinia/CUREv1, heMTEB © 2024 by Clinia) is released under CC BY NC 4.0: "
                "noncommercial use only. Its additional right allows benchmarking embedding models used "
                "commercially, but not training models that may be used commercially, so SAEs trained on "
                "CURE inherit cc-by-nc-4.0. Attribute Clinia when sharing them, and never use them for "
                "clinical decisions."
            ),
            paper_role="training set (train and val splits) and in domain test set",
            splits=("train", "val", "test"),
            n_documents=244_600,
            n_queries={"train": 1_400, "val": 200, "test": 400},
            loader=load_cure,
        ),
        DatasetInfo(
            name="nfcorpus",
            title="NFCorpus",
            revisions={
                "BeIR/nfcorpus": "b5026a0e96e8a7ac4f95f482a596389289d46269",
                "BeIR/nfcorpus-qrels": "a451b3b26d3ae1358f259c1a3a4dd61fcea35a65",
            },
            license="cc-by-sa-4.0",
            terms=(
                "NFCorpus (BeIR/nfcorpus) is labelled cc-by-sa-4.0 on the Hub, but the original release is "
                "free for academic purposes only: other uses of its NutritionFacts.org content need the "
                "permission of its authors."
            ),
            paper_role=_OOD + "; its train and val splits make it a small dataset to train on",
            splits=("train", "val", "test"),
            n_documents=3_633,
            n_queries={"train": 2_590, "val": 324, "test": 323},
            loader=load_nfcorpus,
        ),
        DatasetInfo(
            name="miriad",
            title="MIRIAD",
            revisions={"tomaarsen/miriad-4.4M-split": "596b9ab305d52cb73644ed5b5004957c7bfaae40"},
            license="odc-by",
            terms=(
                "MIRIAD (tomaarsen/miriad-4.4M-split, from miriad/miriad-4.4M) is released under ODC By 1.0 "
                "and is subject to the OpenAI terms of use, because its questions were generated with OpenAI "
                "models. It must not be used for medical diagnosis or clinical decision making."
            ),
            paper_role=_OOD,
            splits=("test",),
            n_documents=31_000,
            n_queries={"test": 1_000},
            loader=load_miriad,
            upstream=("miriad/miriad-4.4M",),
        ),
        _r2med("Biology", "8b9fec2db9eda4b5742d03732213fbaee8169556", 57_359, 103),
        _r2med("Bioinformatics", "6021fce366892cbfd7837fa85a4128ea93315e18", 47_471, 77),
        _r2med("Medical-Sciences", "1b48911514c80bf9182222d99752ad75e23b4b47", 34_792, 88),
    )
}

# Terms are printed once per session and dataset, not at every load.
_TERMS_SHOWN: set[str] = set()


def list_datasets() -> list[str]:
    """Registry names: CURE, the training set of the paper, then the out of domain benchmarks."""
    return list(DATASETS)


def resolve_dataset_name(name: str) -> str | None:
    """Registry name for ``name``, or ``None``.

    Any case works, as do hyphens or spaces instead of underscores (``"R2MED-Biology"``) and the Hub id of
    the main repository (``"clinia/CUREv1"``).
    """
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key in DATASETS:
        return key
    return next((info.name for info in DATASETS.values() if info.hf_id.lower() == name.strip().lower()), None)


def get_dataset_info(name: str) -> DatasetInfo:
    """Registry entry of a benchmark (see :func:`resolve_dataset_name` for the accepted spellings)."""
    key = resolve_dataset_name(name)
    if key is None:
        raise KeyError(f"Unknown dataset {name!r}. Available: {', '.join(DATASETS)}.")
    return DATASETS[key]


def load_dataset(name: str | Path, cache_dir: str | Path | None = None, **kwargs: Any) -> RetrievalDataset:
    """Load a benchmark of the paper by name, or a BEIR folder by path.

    Args:
        name: Registry name (see :func:`list_datasets`; any case, hyphens accepted) or the path of a folder
            in the BEIR layout, which is then named after the folder (call
            :func:`~doctosae.data.load_beir_folder` to choose another name). A :class:`~pathlib.Path`, or a
            string with a path separator or starting with ``.`` or ``~``, is read as a folder first.
        cache_dir: Hugging Face cache folder for the downloaded files (the default Hub cache when ``None``).
        **kwargs: Options of the dataset loader, for example ``revision``, ``token``, ``language`` (CURE)
            or ``n_background`` (MIRIAD).

    Returns:
        The :class:`RetrievalDataset`, with ``license`` and ``hf_ids`` set for registry benchmarks.
    """
    text = str(name)
    # A Path, or a string that reads as a path, is a folder even when its name is also a registry name: the
    # BEIR zip of NFCorpus extracts to nfcorpus/, which must load from disk.
    as_path = isinstance(name, Path) or text.startswith((".", "~")) or "/" in text or "\\" in text
    folder = Path(text).expanduser()
    key = resolve_dataset_name(text)
    if as_path and folder.is_dir():
        key = None
    elif key is not None and folder.is_dir():
        logger.warning(
            "%s is both a registered dataset and a folder here: the registry wins. Load the folder with "
            "load_dataset(Path(%r)) or load_beir_folder(%r).",
            text,
            text,
            text,
        )
    if key is None:
        if not folder.is_dir():
            raise KeyError(f"Unknown dataset {text!r}. Available: {', '.join(DATASETS)}, or a BEIR folder.")
        if kwargs:
            raise TypeError(f"a BEIR folder takes no loader options, got {sorted(kwargs)}")
        return load_beir_folder(folder)
    info = DATASETS[key]
    if key not in _TERMS_SHOWN:
        _TERMS_SHOWN.add(key)
        logger.warning("%s terms (license %s): %s", info.title, info.license, info.terms)
    dataset = info.loader(info, cache_dir, **kwargs)
    logger.info("%s", dataset.describe())
    return dataset
