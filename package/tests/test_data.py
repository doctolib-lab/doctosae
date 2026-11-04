"""Dataset registry and loaders.

Offline tests serve tiny synthetic copies of the Hub files through a fake ``_hub_file``; tests marked
``network`` download the real benchmarks and check the sizes the paper reports. The CURE and MIRIAD network
tests download about 300 MB each and also need ``MEDSAE_LARGE_DOWNLOADS=1``.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from medsae.data import (
    CURE_DISCIPLINES,
    DATASETS,
    RetrievalDataset,
    get_dataset_info,
    ids_sha256,
    list_datasets,
    load_beir_folder,
    load_dataset,
    loaders,
    registry,
    resolve_dataset_name,
    split_corpus,
    split_queries,
)

large_downloads = pytest.mark.skipif(
    os.environ.get("MEDSAE_LARGE_DOWNLOADS") != "1",
    reason="about 300 MB to download: MEDSAE_LARGE_DOWNLOADS=1",
)


def write_jsonl(path: Path, rows: list[dict], compress: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row) + "\n" for row in rows)
    if compress:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(text)
    else:
        path.write_text(text, encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    """Serve ``tmp_path/hub/<repo>/<file>`` instead of downloading, and record every requested file."""
    root = tmp_path / "hub"
    requests: list[tuple[str, str, str]] = []

    def hub_file(repo_id, filename, revision, cache_dir, token):
        requests.append((repo_id, filename, revision))
        path = root / repo_id / filename
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    monkeypatch.setattr(loaders, "_hub_file", hub_file)
    monkeypatch.setattr(registry, "_TERMS_SHOWN", set())
    return root, requests


# BEIR folders.


@pytest.fixture
def beir_folder(tmp_path):
    root = tmp_path / "toy-beir"
    write_jsonl(
        root / "corpus.jsonl",
        [
            {
                "_id": 7,
                "title": " Statins ",
                "text": " lower cholesterol. ",
                "theme": "cardio",
                "metadata": {"u": 1},
            },
            {"_id": "d2", "title": "", "text": "  untitled passage"},
            {"_id": "d3", "title": "Diet", "text": "fiber and health", "theme": "nutrition"},
        ],
    )
    write_jsonl(
        root / "queries.jsonl",
        [
            {"_id": "007", "text": " do statins work? ", "metadata": {"url": "x"}},
            {"_id": "q2", "text": "fiber"},
            {"_id": "q3", "text": "unjudged query"},
        ],
    )
    write_text(root / "qrels" / "train.tsv", "query-id\tcorpus-id\tscore\n007\t7\t2\n007\td2\t0\n")
    write_text(root / "qrels" / "dev.tsv", "query-id\tcorpus-id\tscore\nq2\td3\t1\n")
    write_text(root / "qrels" / "test.tsv", "query-id\tcorpus-id\tscore\nq2\td3\t1\n007\td3\t1\n")
    return root


def test_beir_folder_follows_the_paper_text_rule_and_split_membership(beir_folder):
    dataset = load_beir_folder(beir_folder)
    assert dataset.name == "toy-beir"
    assert dataset.splits == ["train", "val", "test"]
    corpus = dataset.corpus.set_index("doc_id")
    assert corpus.loc["7", "text"] == "Statins lower cholesterol."
    assert corpus.loc["d2", "text"] == "untitled passage"
    assert corpus.loc["7", "title"] == " Statins " and corpus.loc["7", "body"] == " lower cholesterol. "
    assert "theme" in corpus.columns and "metadata" not in corpus.columns
    assert dataset.queries["train"]["query_id"].tolist() == ["007"]  # ids stay strings, leading zeros kept
    assert dataset.queries["train"]["text"].tolist() == [" do statins work? "]  # queries keep their raw text
    assert dataset.queries["test"]["query_id"].tolist() == ["007", "q2"]  # file order
    assert dataset.qrels["train"]["score"].tolist() == [2, 0]
    assert dataset.positives("train") == {"007": {"7"}}
    assert "q3" not in set().union(*(set(frame["query_id"]) for frame in dataset.queries.values()))


def test_load_dataset_accepts_a_beir_folder(beir_folder):
    dataset = load_dataset(str(beir_folder))
    assert dataset.name == "toy-beir" and len(dataset.corpus) == 3
    assert load_beir_folder(beir_folder, name="toy").name == "toy"
    with pytest.raises(TypeError, match="no loader options"):
        load_dataset(beir_folder, revision="main")


def test_beir_folder_without_titles_embeds_the_raw_text(tmp_path):
    root = tmp_path / "raw"
    write_jsonl(root / "corpus.jsonl.gz", [{"id": "a", "text": "  kept as is "}], compress=True)
    write_jsonl(root / "queries.jsonl", [{"id": "q", "text": "query", "title": "Topic"}])
    write_text(root / "qrels" / "validation.tsv", "query-id corpus-id score\nq a 1\n")
    dataset = load_beir_folder(root)
    assert dataset.corpus["text"].tolist() == ["  kept as is "]
    assert dataset.corpus["title"].tolist() == [""]
    assert dataset.queries["val"]["text"].tolist() == [
        "Topic query"
    ]  # a query title is prepended, as in the paper


def test_beir_folder_keeps_the_first_row_of_a_repeated_document_id(tmp_path, caplog):
    root = tmp_path / "dups"
    write_jsonl(root / "corpus.jsonl", [{"_id": "a", "text": "first"}, {"_id": "a", "text": "second"}])
    write_jsonl(root / "queries.jsonl", [{"_id": "q", "text": "query"}])
    write_text(root / "qrels" / "test.tsv", "query-id\tcorpus-id\tscore\nq\ta\t1\n")
    with caplog.at_level(logging.WARNING, logger="medsae"):
        dataset = load_beir_folder(root)
    assert dataset.corpus["text"].tolist() == ["first"]
    assert "dropping 1 rows" in caplog.text


@pytest.mark.parametrize(
    "content",
    [
        "query-id\tcorpus-id\trelevance\nq\ta\t1\n",  # the paper's MIRIAD header
        "qid\tdocid\tscore\nq\ta\t1\n",
        "query\tdoc\trel\nq\ta\t1\n",
        "q\ta\t1\n",  # no header
        "q 0 a 1\n",  # TREC
    ],
)
def test_qrels_files_in_several_formats(tmp_path, content):
    root = tmp_path / "formats"
    write_jsonl(root / "corpus.jsonl", [{"_id": "a", "text": "doc"}, {"_id": "b", "text": "other"}])
    write_jsonl(root / "queries.jsonl", [{"_id": "q", "text": "query"}])
    write_text(root / "qrels" / "test.tsv", content)
    qrels = load_beir_folder(root).qrels["test"]
    assert qrels.to_dict("records") == [{"query_id": "q", "doc_id": "a", "score": 1}]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("query-id\tcorpus-id\tscore\nq\ta\t0.5\n", "must be integers, got 0.5"),
        ("qq\tdd\trr\nq\ta\t1\n", "accepted names"),
    ],
)
def test_qrels_files_that_cannot_be_read_exactly_are_refused(tmp_path, content, message):
    root = tmp_path / "bad"
    write_jsonl(root / "corpus.jsonl", [{"_id": "a", "text": "doc"}])
    write_jsonl(root / "queries.jsonl", [{"_id": "q", "text": "query"}])
    write_text(root / "qrels" / "test.tsv", content)
    with pytest.raises(ValueError, match=message):
        load_beir_folder(root)


def test_a_path_loads_the_folder_even_with_a_registry_name(beir_folder, tmp_path, monkeypatch, caplog):
    local = tmp_path / "work"
    local.mkdir()
    (local / "nfcorpus").symlink_to(beir_folder, target_is_directory=True)
    monkeypatch.chdir(local)
    assert len(load_dataset(Path("nfcorpus")).corpus) == 3
    assert len(load_dataset("./nfcorpus").corpus) == 3
    calls = []
    loader = lambda *args, **kwargs: calls.append(1) or load_beir_folder(beir_folder)  # noqa: E731
    monkeypatch.setitem(
        registry.DATASETS, "nfcorpus", dataclasses.replace(DATASETS["nfcorpus"], loader=loader)
    )
    with caplog.at_level(logging.WARNING, logger="medsae.data.registry"):
        load_dataset("nfcorpus")  # a bare name keeps meaning the registry, with a warning
    assert calls == [1] and "both a registered dataset and a folder" in caplog.text


def test_beir_folder_errors(tmp_path):
    root = tmp_path / "broken"
    with pytest.raises(FileNotFoundError, match="corpus.jsonl"):
        load_beir_folder(root)
    write_jsonl(root / "corpus.jsonl", [{"_id": "a", "text": "doc"}])
    write_jsonl(root / "queries.jsonl", [{"_id": "q", "text": "query"}])
    with pytest.raises(FileNotFoundError, match="qrels"):
        load_beir_folder(root)
    write_text(root / "qrels" / "dev.tsv", "query-id\tcorpus-id\tscore\nq\ta\t1\n")
    write_text(root / "qrels" / "validation.tsv", "query-id\tcorpus-id\tscore\nq\ta\t1\n")
    with pytest.raises(ValueError, match="two files"):
        load_beir_folder(root)


# Registry.


def test_registry_names_and_aliases():
    assert list_datasets() == [
        "cure",
        "nfcorpus",
        "miriad",
        "r2med_biology",
        "r2med_bioinformatics",
        "r2med_medical_sciences",
    ]
    assert resolve_dataset_name("CURE") == "cure"
    assert resolve_dataset_name("R2MED-Medical-Sciences") == "r2med_medical_sciences"
    assert resolve_dataset_name("r2med bioinformatics") == "r2med_bioinformatics"
    assert resolve_dataset_name("clinia/CUREv1") == "cure"
    assert resolve_dataset_name("BeIR/nfcorpus") == "nfcorpus"
    assert resolve_dataset_name("scifact") is None
    with pytest.raises(KeyError, match="Available"):
        get_dataset_info("scifact")
    with pytest.raises(KeyError, match="Available"):
        load_dataset("scifact")


@pytest.mark.parametrize("name", list(DATASETS))
def test_registry_entries_are_pinned_and_licensed(name):
    info = get_dataset_info(name)
    assert all(len(revision) == 40 and int(revision, 16) >= 0 for revision in info.revisions.values())
    assert info.license in {"cc-by-nc-4.0", "cc-by-sa-4.0", "odc-by", "cc-by-4.0"}
    assert info.terms and info.paper_role
    assert set(info.n_queries) == set(info.splits)
    assert info.hf_ids[0] == info.hf_id and callable(info.loader)


def test_terms_are_printed_once_per_session(fake_hub, caplog, monkeypatch):
    built = RetrievalDataset("cure", pd.DataFrame({"doc_id": ["a"], "text": ["t"]}), {}, {})
    monkeypatch.setitem(
        DATASETS, "cure", dataclasses.replace(DATASETS["cure"], loader=lambda info, cache: built)
    )
    with caplog.at_level(logging.WARNING, logger="medsae"):
        load_dataset("cure")
        load_dataset("CURE")
    notices = [record for record in caplog.records if "terms" in record.getMessage()]
    assert len(notices) == 1 and "cc-by-nc-4.0" in notices[0].getMessage()


# Hub loaders on synthetic files.


def write_fake_cure(root: Path) -> None:
    base = root / "clinia/CUREv1"
    for index, discipline in enumerate(CURE_DISCIPLINES):
        docs = [
            {"_id": f"doc-{index}-{j}", "title": f" Title {index} {j} ", "text": f" body {index} {j} "}
            for j in range(3)
        ]
        if index in (1, 3):  # one passage listed under two disciplines, with the same title and text
            docs.append({"_id": "shared", "title": "Shared", "text": "passage"})
        write_jsonl(base / discipline / "corpus.jsonl", docs)
        english = [
            {"_id": f"q-{index}-{j}", "text": f"question {index} {j}", "expertise": "expert"}
            for j in range(2)
        ]
        write_jsonl(base / discipline / "queries-en.jsonl", english)
        french = [dict(row, text=f"question fr {row['_id']}") for row in reversed(english)]
        write_jsonl(base / discipline / "queries-fr.jsonl", french)
        qrels = [
            {"query-id": f"q-{index}-{j}", "corpus-id": f"doc-{index}-{j}", "score": 2, "judge": "model"}
            for j in range(2)
        ]
        qrels.append(
            {"query-id": f"q-{index}-0", "corpus-id": f"doc-{index}-2", "score": 1, "explanation": "x"}
        )
        write_jsonl(base / discipline / "qrels.jsonl", qrels)


def test_cure_recipe(fake_hub, caplog):
    root, requests = fake_hub
    write_fake_cure(root)
    with caplog.at_level(logging.WARNING, logger="medsae"):
        dataset = load_dataset("cure")
    assert {revision for _, _, revision in requests} == {DATASETS["cure"].revision}
    corpus = dataset.corpus.set_index("doc_id")
    assert len(corpus) == 31  # 32 rows, one repeated id
    assert corpus.loc["shared", "source_folder"] == "dermatology"
    assert corpus.loc["doc-0-1", "text"] == "Title 0 1 body 0 1"
    english_ids = [f"q-{index}-{j}" for index in range(10) for j in range(2)]
    expected = split_queries(english_ids)
    for split in ("train", "val", "test"):
        assert dataset.queries[split]["query_id"].tolist() == [english_ids[i] for i in expected[split]]
        assert set(dataset.qrels[split]["query_id"]) <= set(dataset.queries[split]["query_id"])
    assert [len(dataset.queries[split]) for split in ("train", "val", "test")] == [14, 2, 4]
    assert sum(len(frame) for frame in dataset.qrels.values()) == 30
    assert set(dataset.queries["train"].columns) >= {"expertise", "discipline"}
    train_positives = set(dataset.qrels["train"]["doc_id"])
    expected_corpus = split_corpus(dataset.corpus["doc_id"], train_positives)
    assert all(
        np.array_equal(dataset.corpus_splits[name], expected_corpus[name]) for name in ("train", "val")
    )
    assert train_positives <= set(dataset.corpus["doc_id"].to_numpy()[dataset.corpus_splits["train"]])
    assert dataset.license == "cc-by-nc-4.0" and dataset.hf_ids == ["clinia/CUREv1"]
    assert "differs from the paper split" in caplog.text  # synthetic data cannot match the paper checksums


def test_cure_other_languages_reuse_the_english_split(fake_hub):
    root, _ = fake_hub
    write_fake_cure(root)
    english = load_dataset("cure")
    french = load_dataset("cure", language="fr")
    for split in ("train", "val", "test"):
        assert set(french.queries[split]["query_id"]) == set(english.queries[split]["query_id"])
    assert french.queries["test"]["text"].str.startswith("question fr").all()
    with pytest.raises(ValueError, match="en, es and fr"):
        load_dataset("cure", language="de")


def test_cure_checksum_guard_is_skipped_off_the_paper_settings(fake_hub, caplog):
    root, requests = fake_hub
    write_fake_cure(root)
    with caplog.at_level(logging.WARNING, logger="medsae.data.loaders"):
        dataset = load_dataset("cure", seed=1, revision="main")
    assert "differs" not in caplog.text
    assert {revision for _, _, revision in requests} == {"main"}
    assert sorted(len(frame) for frame in dataset.queries.values()) == [2, 4, 14]


def test_nfcorpus_recipe(fake_hub):
    root, _ = fake_hub
    corpus_dir = root / "BeIR/nfcorpus"
    corpus_dir.joinpath("corpus").mkdir(parents=True)
    corpus_dir.joinpath("queries").mkdir(parents=True)
    pd.DataFrame(
        {"_id": ["MED-1", "MED-2"], "title": ["Statins ", "Fiber"], "text": [" help", "and you"]}
    ).to_parquet(corpus_dir / "corpus/corpus-00000-of-00001.parquet")
    pd.DataFrame(
        {"_id": ["PLAIN-1", "PLAIN-2", "PLAIN-3"], "title": ["", "", ""], "text": ["a ", "b", "c"]}
    ).to_parquet(corpus_dir / "queries/queries-00000-of-00001.parquet")
    qrels_dir = root / "BeIR/nfcorpus-qrels"
    write_text(qrels_dir / "train.tsv", "query-id\tcorpus-id\tscore\nPLAIN-1\tMED-1\t1\n")
    write_text(qrels_dir / "dev.tsv", "query-id\tcorpus-id\tscore\nPLAIN-2\tMED-2\t2\n")
    write_text(qrels_dir / "test.tsv", "query-id\tcorpus-id\tscore\nPLAIN-3\tMED-1\t2\nPLAIN-3\tMED-2\t1\n")
    dataset = load_dataset("nfcorpus")
    assert dataset.splits == ["train", "val", "test"]
    assert dataset.corpus["text"].tolist() == ["Statins help", "Fiber and you"]
    assert dataset.queries["train"]["text"].tolist() == ["a "]
    assert dataset.queries["test"]["query_id"].tolist() == ["PLAIN-3"]
    assert dataset.qrels["test"]["score"].tolist() == [2, 1]
    assert dataset.hf_ids == ["BeIR/nfcorpus", "BeIR/nfcorpus-qrels"] and dataset.license == "cc-by-sa-4.0"


def test_miriad_recipe(fake_hub):
    root, requests = fake_hub
    data = root / "tomaarsen/miriad-4.4M-split/data"
    data.mkdir(parents=True)
    pd.DataFrame(
        {
            "question": [f"question {i}?" for i in range(5)],
            "passage_text": [f" passage {i} " for i in range(5)],
        }
    ).to_parquet(data / "test-00000-of-00001.parquet", row_group_size=2)
    pd.DataFrame(
        {"question": ["unused"] * 8, "passage_text": [f"background {i}" for i in range(8)]}
    ).to_parquet(data / "train-00000-of-00042.parquet", row_group_size=3)
    dataset = load_dataset("miriad", n_queries=3, n_background=4)
    assert dataset.corpus["doc_id"].tolist() == [
        "test-0",
        "test-1",
        "test-2",
        "train-0",
        "train-1",
        "train-2",
        "train-3",
    ]
    assert dataset.corpus["text"].tolist()[:2] == ["passage 0", "passage 1"]
    assert dataset.queries["test"]["text"].tolist() == ["question 0?", "question 1?", "question 2?"]
    assert dataset.qrels["test"].to_dict("records")[1] == {
        "query_id": "test-1",
        "doc_id": "test-1",
        "score": 1,
    }
    assert dataset.hf_ids == ["tomaarsen/miriad-4.4M-split", "miriad/miriad-4.4M"]

    requests.clear()
    assert len(load_dataset("miriad", n_queries=2, n_background=0).corpus) == 2
    assert all("train-" not in filename for _, filename, _ in requests)
    with pytest.raises(ValueError, match="requested"):
        load_dataset("miriad", n_queries=3, n_background=100)


def test_r2med_recipe(fake_hub, caplog):
    root, _ = fake_hub
    base = root / "R2MED/Biology"
    write_jsonl(
        base / "corpus.jsonl",
        [
            {"id": "a.txt", "text": " leading space"},
            {"id": "b.txt", "text": "b"},
            {"id": "a.txt", "text": " leading space"},
        ],
    )
    write_jsonl(
        base / "query.jsonl",
        [
            {"id": "0", "text": "Why? ", "doc_id": ["a.txt"], "answer": "Because.", "body_system": ""},
            {
                "id": "1",
                "text": "How?",
                "doc_id": ["b.txt", "a.txt"],
                "answer": "So.",
                "body_system": "cells",
            },
        ],
    )
    write_jsonl(
        base / "qrels.jsonl",
        [{"q_id": "0", "p_id": "a.txt", "score": 1}, {"q_id": "1", "p_id": "b.txt", "score": 1}],
    )
    with caplog.at_level(logging.WARNING, logger="medsae"):
        dataset = load_dataset("R2MED-Biology")
    assert dataset.name == "r2med_biology" and dataset.splits == ["test"]
    assert dataset.corpus["text"].tolist() == [" leading space", "b"]  # raw text, first row of a repeated id
    assert "dropping 1 rows" in caplog.text
    queries = dataset.queries["test"]
    assert queries["query_id"].tolist() == ["0", "1"] and queries["text"].tolist() == ["Why? ", "How?"]
    assert queries["positive_doc_ids"].tolist() == [["a.txt"], ["b.txt", "a.txt"]]
    assert dataset.qrels["test"]["query_id"].tolist() == ["0", "1"]
    assert dataset.license == "cc-by-4.0"


# Real benchmarks (network).


@pytest.mark.network
def test_nfcorpus_sizes_match_the_paper():
    dataset = load_dataset("nfcorpus")
    assert len(dataset.corpus) == 3_633
    assert {split: len(frame) for split, frame in dataset.queries.items()} == {
        "train": 2_590,
        "val": 324,
        "test": 323,
    }
    assert {split: len(frame) for split, frame in dataset.qrels.items()} == {
        "train": 110_575,
        "val": 11_385,
        "test": 12_334,
    }
    assert dataset.qrels["test"]["score"].value_counts().to_dict() == {1: 11_758, 2: 576}
    assert dataset.qrels["test"]["doc_id"].isin(dataset.doc_index).all()
    assert dataset.corpus["doc_id"].is_unique


@pytest.mark.network
@pytest.mark.parametrize(
    ("name", "n_qrels"),
    [("r2med_biology", 374), ("r2med_bioinformatics", 227), ("r2med_medical_sciences", 244)],
)
def test_r2med_sizes_match_the_paper(name, n_qrels):
    dataset = load_dataset(name)
    info = get_dataset_info(name)
    assert len(dataset.corpus) == info.n_documents
    assert len(dataset.queries["test"]) == info.n_queries["test"]
    assert len(dataset.qrels["test"]) == n_qrels


@pytest.mark.network
@large_downloads
def test_cure_split_matches_the_paper():
    dataset = load_dataset("cure")
    assert len(dataset.corpus) == 244_600
    assert {split: len(frame) for split, frame in dataset.queries.items()} == {
        "train": 1_400,
        "val": 200,
        "test": 400,
    }
    assert {split: len(frame) for split, frame in dataset.qrels.items()} == {
        "train": 57_430,
        "val": 7_581,
        "test": 15_705,
    }
    assert {split: len(rows) for split, rows in dataset.corpus_splits.items()} == {
        "train": 223_425,
        "val": 21_175,
    }
    assert ids_sha256(dataset.queries["test"]["query_id"]) == (
        "e81ded71676166bfc3bda7d3b18028b29323417ed910f31fd42c5fbb612092bf"
    )
    doc_ids = dataset.corpus["doc_id"].to_numpy()
    assert ids_sha256(doc_ids[dataset.corpus_splits["val"]]) == (
        "74439a9cbcbf108d83e8f039bdc2b82749f36111eb180fef6f2dfeac87ac8141"
    )


@pytest.mark.network
@large_downloads
def test_miriad_matches_the_paper():
    dataset = load_dataset("miriad")
    assert len(dataset.corpus) == 31_000 and len(dataset.queries["test"]) == 1_000
    assert dataset.corpus["doc_id"].iloc[[0, 999, 1000, -1]].tolist() == [
        "test-0",
        "test-999",
        "train-0",
        "train-29999",
    ]
    assert dataset.corpus["text"].nunique() == 30_792  # duplicate passage texts are kept, as in the paper
