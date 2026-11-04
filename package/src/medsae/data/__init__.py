"""Retrieval benchmarks of the paper as :class:`RetrievalDataset` objects.

>>> from medsae.data import list_datasets, load_dataset
>>> list_datasets()
['cure', 'nfcorpus', 'miriad', 'r2med_biology', 'r2med_bioinformatics', 'r2med_medical_sciences']
>>> nfcorpus = load_dataset("nfcorpus")  # doctest: +SKIP

Any folder in the BEIR layout loads the same way: ``load_dataset("path/to/folder")`` or
:func:`load_beir_folder`.
"""

from medsae.data.dataset import SPLIT_ORDER, EmbeddedDataset, RetrievalDataset
from medsae.data.loaders import CURE_DISCIPLINES, load_beir_folder
from medsae.data.registry import (
    DATASETS,
    DatasetInfo,
    get_dataset_info,
    list_datasets,
    load_dataset,
    resolve_dataset_name,
)
from medsae.data.splits import ids_sha256, split_corpus, split_queries

__all__ = [
    "CURE_DISCIPLINES",
    "DATASETS",
    "SPLIT_ORDER",
    "DatasetInfo",
    "EmbeddedDataset",
    "RetrievalDataset",
    "get_dataset_info",
    "ids_sha256",
    "list_datasets",
    "load_beir_folder",
    "load_dataset",
    "resolve_dataset_name",
    "split_corpus",
    "split_queries",
]
