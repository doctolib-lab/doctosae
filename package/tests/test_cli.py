"""Tests of the ``doctosae`` command line.

Offline: the help of every command, the dataset registry listing, the conversion of a synthetic paper
checkpoint, the arguments of a Hub upload (the upload itself is replaced, nothing leaves the machine) and the
train, evaluate, interpret and explain commands on tiny synthetic embeddings. The ``network`` test embeds a
tiny dataset with a small sentence transformers model first, as a user would.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import doctosae
from doctosae import cli
from doctosae.data import EmbeddedDataset, load_dataset
from doctosae.interpret import load_cards
from doctosae.sae import DoctoSAE
from doctosae.training import TrainingConfig

COMMANDS = ("datasets", "pretrained", "embed", "train", "evaluate", "interpret", "explain", "push", "convert")
REGISTRY = ("cure", "nfcorpus", "miriad", "r2med_biology", "r2med_bioinformatics", "r2med_medical_sciences")
SRC = Path(__file__).resolve().parents[1] / "src"
TOPICS = {
    "kidney": ["kidney", "stones", "renal", "urine", "calcium", "nephrology", "dialysis", "colic"],
    "heart": ["cardiac", "heart", "arrhythmia", "blood", "pressure", "artery", "stroke", "statin"],
    "skin": ["dermatitis", "eczema", "rash", "itching", "skin", "psoriasis", "lesion", "cream"],
    "lung": ["asthma", "inhaler", "airway", "breathing", "lung", "cough", "spirometry", "wheeze"],
}


@pytest.fixture(autouse=True)
def keep_log_level():
    """`doctosae.cli.main` sets the level of the doctosae logger: give it back to the other test modules."""
    level = logging.getLogger("doctosae").level
    yield
    logging.getLogger("doctosae").setLevel(level)


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    """Run ``doctosae *argv`` in process; return the exit code, stdout and stderr."""
    code = cli.main([str(arg) for arg in argv])
    out, err = capsys.readouterr()
    return code, out, err


def help_text(capsys: pytest.CaptureFixture[str], *argv: str) -> str:
    with pytest.raises(SystemExit) as stop:
        cli.main([*argv, "--help"])
    assert stop.value.code == 0
    return capsys.readouterr().out


# Help, version and import time


def test_help_lists_every_command(capsys):
    text = help_text(capsys)
    for command in COMMANDS:
        assert command in text


@pytest.mark.parametrize("command", COMMANDS)
def test_every_command_has_help(command, capsys):
    text = help_text(capsys, command)
    assert text.startswith(f"usage: doctosae {command}")
    assert "doctosae." in text  # the help names the Python function behind the command


def test_version(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["--version"])
    assert stop.value.code == 0
    assert capsys.readouterr().out.strip() == f"doctosae {doctosae.__version__}"


def test_help_does_not_import_heavy_libraries():
    """``doctosae --help`` must stay fast: no torch, sentence transformers or datasets at import time."""
    code = (
        "import sys\n"
        "import doctosae.cli\n"
        "try:\n"
        "    doctosae.cli.main(['--help'])\n"
        "except SystemExit:\n"
        "    pass\n"
        "heavy = ('torch', 'sentence_transformers', 'datasets', 'transformers', 'pandas')\n"
        "print('loaded:', ','.join(name for name in heavy if name in sys.modules))\n"
    )
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [str(SRC), os.environ.get("PYTHONPATH")])),
    }
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert result.stdout.strip().splitlines()[-1] == "loaded:"


def test_missing_command_is_an_error(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main([])
    assert stop.value.code == 2
    assert "required" in capsys.readouterr().err


# Package exports


def test_public_names_resolve():
    assert set(doctosae.__all__) == {"__version__", *doctosae._EXPORTS}
    for name in doctosae.__all__:
        assert getattr(doctosae, name) is not None
    assert doctosae.EmbeddedDataset is doctosae.embed.EmbeddedDataset is doctosae.data.EmbeddedDataset
    assert doctosae.DoctoSAE is DoctoSAE
    with pytest.raises(AttributeError):
        _ = doctosae.not_a_public_name


def test_paper_values_in_the_help_are_the_training_defaults():
    defaults = TrainingConfig()
    for name, value in cli.PAPER_TRAINING_VALUES.items():
        assert getattr(defaults, name) == value, name


def test_train_help_shows_the_paper_grid_without_exponents(capsys):
    text = " ".join(help_text(capsys, "train").split())
    assert "default 0.00001; the paper selects it per k from 0, 0.000005 and 0.00001" in text
    assert "e-0" not in text and "(paper: 0.0005)" in text


def test_help_never_breaks_an_id_across_lines(capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "80")
    lines = help_text(capsys, "push").splitlines()
    assert any("my-org/doctosae-bge-large-en-v1.5-k64" in line for line in lines)
    assert any("cc-by-nc-4.0" in line for line in lines)


# Datasets and pretrained checkpoints


def test_datasets_lists_the_registry(capsys):
    code, out, _ = run(capsys, "datasets")
    assert code == 0
    for name in REGISTRY:
        assert name in out


def test_pretrained_names_a_checkpoint(capsys, monkeypatch):
    class FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, revision=None):
            assert repo_id == "doctolib-lab/doctosae-bge-large-en-v1.5"
            return [
                "k64_m3072/cfg.json",
                "k64_m3072/sae.safetensors",
                "k256_m3072/cfg.json",
                "k256_m3072/sae.safetensors",
            ]

    monkeypatch.setattr("doctosae.hub.HfApi", FakeApi)
    code, out, _ = run(capsys, "pretrained", "--embedder", "bge", "--k", "256")
    assert code == 0
    assert out.strip() == "doctolib-lab/doctosae-bge-large-en-v1.5/k256_m3072"
    code, out, err = run(capsys, "pretrained", "--embedder", "bge", "--k", "128")
    assert code == 1 and "k=128" in err


@pytest.mark.network
def test_pretrained_loads_a_released_checkpoint():
    from doctosae.hub import load_pretrained

    sae = load_pretrained("bge", k=64)
    assert (sae.d_in, sae.n_latents, sae.k) == (1024, 3072, 64)
    assert sae.metadata["hub_checkpoint"] == "doctolib-lab/doctosae-bge-large-en-v1.5/k64_m3072"


# Paper checkpoint conversion


def write_paper_run(folder: Path, d: int = 8, m: int = 24, k: int = 4) -> dict[str, torch.Tensor]:
    """A run folder as ``paper/src/training/train_sae.py`` writes it, saved from ``torch.nn.DataParallel``."""
    generator = torch.Generator().manual_seed(0)
    state = {
        "encoder.weight": torch.randn(m, d, generator=generator),
        "encoder.bias": torch.randn(m, generator=generator),
        "decoder.weight": torch.randn(d, m, generator=generator),
        "b_pre": torch.randn(d, generator=generator),
    }
    config = {
        "input_dim": d, "hidden": m, "sparsity_type": "topknorelu", "intermediate": None, "k": k,
        "activation_fn": "relu", "init": "default", "beta": 0.005, "lambda": 1e-05, "alpha_q": 0.1,
        "temperature": 0.05, "recon_loss": "mse", "k_aux": None, "aux_coef": None,
    }  # fmt: skip
    folder.mkdir(parents=True)
    (folder / "sae_config.json").write_text(json.dumps(config, indent=2))
    torch.save({f"module.{key}": value for key, value in state.items()}, folder / "best_sae.pth")
    return state


def test_convert_round_trip(tmp_path, capsys):
    state = write_paper_run(tmp_path / "run")
    code, out, err = run(capsys, "convert", tmp_path / "run", tmp_path / "sae", "--embedder", "bge")
    assert code == 0, err
    assert "Converted" in out
    assert {"config.json", "model.safetensors", "README.md"} <= {p.name for p in (tmp_path / "sae").iterdir()}

    sae = DoctoSAE.from_pretrained(tmp_path / "sae")
    for key, value in state.items():
        torch.testing.assert_close(sae.state_dict()[key], value)
    assert (sae.d_in, sae.n_latents, sae.k, sae.activation) == (8, 24, 4, "relu")
    assert sae.embedder == "BAAI/bge-large-en-v1.5"
    assert sae.metadata["license"] == "cc-by-nc-4.0"
    assert sae.metadata["paper_config"]["hidden"] == 24
    assert "license: cc-by-nc-4.0" in (tmp_path / "sae" / "README.md").read_text()

    # The converted model computes the paper's code: ReLU of the k largest pre activations.
    x = torch.randn(5, 8, generator=torch.Generator().manual_seed(1))
    pre = (x - state["b_pre"]) @ state["encoder.weight"].T + state["encoder.bias"]
    top = pre.topk(4, dim=1).indices
    expected = torch.zeros_like(pre).scatter(1, top, pre.gather(1, top).relu())
    with torch.no_grad():
        torch.testing.assert_close(sae.encode(x), expected)


def test_convert_refuses_a_folder_in_use(tmp_path, capsys):
    write_paper_run(tmp_path / "run")
    (tmp_path / "sae").mkdir()
    (tmp_path / "sae" / "README.md").write_text("card of another checkpoint")
    code, _, err = run(capsys, "convert", tmp_path / "run", tmp_path / "sae")
    assert code == 1
    assert "not an empty folder" in err


def test_convert_reports_a_missing_folder(tmp_path, capsys):
    code, _, err = run(capsys, "convert", tmp_path / "nowhere", tmp_path / "sae")
    assert code == 1
    assert err.startswith("doctosae convert: error:")


def test_a_missing_local_sae_is_not_looked_up_on_the_hub(pipeline, tmp_path, capsys):
    code, _, err = run(capsys, "evaluate", *data_args(pipeline), "--sae", tmp_path / "runs" / "missing")
    assert code == 1
    assert "no DoctoSAE folder" in err


# Hub upload (replaced by a recorder)


def test_push_is_private_unless_public(tmp_path, capsys, monkeypatch):
    DoctoSAE(d_in=8, n_latents=24, k=4, metadata={"license": "cc-by-nc-4.0"}).save_pretrained(
        tmp_path / "sae"
    )
    calls = []

    def record(sae, repo_id, **kwargs):
        calls.append((sae, repo_id, kwargs))
        return f"https://huggingface.co/{repo_id}/commit/0"

    monkeypatch.setattr("doctosae.hub.push_to_hub", record)

    code, out, _ = run(capsys, "push", tmp_path / "sae", "someone/doctosae-test")
    assert code == 0
    sae, repo_id, kwargs = calls[-1]
    assert isinstance(sae, DoctoSAE) and repo_id == "someone/doctosae-test"
    assert kwargs == {"private": True}
    assert "(private)" in out

    code, out, _ = run(
        capsys, "push", tmp_path / "sae", "someone/doctosae-test", "--public", "--license", "cc-by-4.0",
        "--datasets", "R2MED/Biology", "--commit-message", "first upload",
    )  # fmt: skip
    assert code == 0
    assert calls[-1][2] == {
        "private": False,
        "license": "cc-by-4.0",
        "datasets": ["R2MED/Biology"],
        "commit_message": "first upload",
    }
    assert "(public)" in out

    code, _, err = run(capsys, "push", tmp_path / "missing", "someone/doctosae-test")
    assert code == 1 and "not a DoctoSAE folder" in err
    assert len(calls) == 2  # nothing was pushed


# Train, evaluate, interpret and explain on tiny synthetic data


def write_beir_folder(folder: Path, seed: int = 0) -> None:
    """Four topics with 15 documents and 12 queries each; a query is relevant to 5 documents of its topic.
    Queries 0 to 7 of a topic are train queries, 8 and 9 dev queries and 10 and 11 test queries."""
    rng = np.random.default_rng(seed)
    corpus, queries, qrels = [], [], {"train": [], "dev": [], "test": []}
    for topic, words in TOPICS.items():
        for i in range(15):
            text = " ".join(rng.choice(words, size=12))
            corpus.append({"_id": f"{topic}_d{i}", "title": f"{topic} study {i}", "text": text})
        for i in range(12):
            query_id = f"{topic}_q{i}"
            queries.append({"_id": query_id, "text": "what about " + " ".join(rng.choice(words, size=4))})
            split = "train" if i < 8 else "dev" if i < 10 else "test"
            for doc in rng.choice(15, size=5, replace=False):
                qrels[split].append(f"{query_id}\t{topic}_d{doc}\t{rng.integers(1, 3)}")
    (folder / "qrels").mkdir(parents=True)
    (folder / "corpus.jsonl").write_text("".join(json.dumps(row) + "\n" for row in corpus))
    (folder / "queries.jsonl").write_text("".join(json.dumps(row) + "\n" for row in queries))
    for split, rows in qrels.items():
        header = "query-id\tcorpus-id\tscore\n"
        (folder / "qrels" / f"{split}.tsv").write_text(header + "\n".join(rows) + "\n")


def write_embeddings(folder: Path, beir: Path, dim: int = 16, seed: int = 1) -> None:
    """Unit vectors around one center per topic, saved as `doctosae embed` saves them."""
    dataset = load_dataset(beir)
    rng = np.random.default_rng(seed)
    centers = {topic: rng.normal(size=dim) for topic in TOPICS}

    def embed(ids: list[str]) -> np.ndarray:
        vectors = np.stack([centers[item.split("_")[0]] + 0.5 * rng.normal(size=dim) for item in ids])
        return (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)

    queries = {split: embed(frame["query_id"].tolist()) for split, frame in dataset.queries.items()}
    corpus = embed(dataset.corpus["doc_id"].tolist())
    EmbeddedDataset(dataset, "synthetic_encoder", corpus, queries).save(folder)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A synthetic BEIR folder, its embeddings and an SAE trained on them with `doctosae train`."""
    root = tmp_path_factory.mktemp("pipeline")
    paths = {"beir": root / "beir", "embeddings": root / "embeddings", "sae": root / "sae"}
    write_beir_folder(paths["beir"])
    write_embeddings(paths["embeddings"], paths["beir"])
    level = logging.getLogger("doctosae").level
    code = cli.main(
        ["train", "--dataset", str(paths["beir"]), "--embeddings", str(paths["embeddings"]),
         "--output-dir", str(paths["sae"]), "--k", "4", "--epochs", "6", "--batch-size", "16",
         "--eval-every", "4", "--device", "cpu", "--no-progress", "-q"]
    )  # fmt: skip
    logging.getLogger("doctosae").setLevel(level)
    assert code == 0
    return paths


def data_args(pipeline: dict[str, Path]) -> list[str]:
    """The dataset and embedding flags shared by every command run on the synthetic pipeline."""
    return ["--dataset", str(pipeline["beir"]), "--embeddings", str(pipeline["embeddings"]), "--device=cpu"]


def test_train_saves_the_selected_checkpoint(pipeline):
    sae_dir = pipeline["sae"]
    names = {p.name for p in sae_dir.iterdir()}
    assert {"config.json", "model.safetensors", "README.md", "history.json", "summary.json"} <= names
    sae = DoctoSAE.from_pretrained(sae_dir)
    assert (sae.d_in, sae.n_latents, sae.k) == (16, 48, 4)
    assert sae.embedder == "synthetic_encoder"
    summary = json.loads((sae_dir / "summary.json").read_text())
    assert summary["selection_split"] == "val"
    assert summary["config"]["epochs"] == 6 and summary["config"]["batch_size"] == 16
    assert summary["config"]["lr"] == TrainingConfig().lr  # untouched options keep the paper defaults


def test_evaluate_sparse_codes_and_dense_embeddings(pipeline, tmp_path, capsys):
    code, out, err = run(capsys, "evaluate", *data_args(pipeline), "--sae", pipeline["sae"], "--output",
                         tmp_path / "sparse.json")  # fmt: skip
    assert code == 0, err
    assert "DoctoSAE sparse codes" in out
    sparse = json.loads((tmp_path / "sparse.json").read_text())
    assert 0 < sparse["ndcg@10"] <= 1 and sparse["n_queries"] == 8

    code, out, err = run(capsys, "evaluate", *data_args(pipeline), "--split", "val", "--cutoff", "5")
    assert code == 0, err
    assert "dense synthetic_encoder embeddings" in out and "ndcg@5" in out


def test_evaluate_rejects_an_sae_of_another_size(pipeline, tmp_path, capsys):
    DoctoSAE(d_in=8, n_latents=24, k=4).save_pretrained(tmp_path / "small")
    code, _, err = run(capsys, "evaluate", *data_args(pipeline), "--sae", tmp_path / "small")
    assert code == 1
    assert "8 dimensional" in err


def test_interpret_writes_cards_and_scores(pipeline, tmp_path, capsys):
    output = tmp_path / "features"
    code, out, err = run(
        capsys, "interpret", *data_args(pipeline), "--sae", pipeline["sae"], "--output", output,
        "--n-features", "3", "--min-documents", "1", "--min-queries", "1", "--llm", "fake", "--score",
        "--no-progress",
    )  # fmt: skip
    assert code == 0, err
    cards = load_cards(output / "cards.json")
    assert 0 < len(cards) <= 3
    assert all(card.tfidf_terms and card.top_documents and card.topic for card in cards)
    assert "### Feature" in (output / "cards.md").read_text()
    scores = json.loads((output / "scores.json").read_text())
    assert set(scores["summary"]) == {"tfidf", "llm", "snippets"}
    assert "Activation Prediction" in out


def test_interpret_scores_out_of_domain(pipeline, tmp_path, capsys):
    write_beir_folder(tmp_path / "other", seed=5)
    write_embeddings(tmp_path / "other_embeddings", tmp_path / "other")
    code, out, err = run(
        capsys, "interpret", *data_args(pipeline), "--sae", pipeline["sae"], "--features", "0", "1", "2",
        "--output", tmp_path / "features", "--llm", "fake", "--score", "--methods", "tfidf", "--no-progress",
        "--score-on", tmp_path / "other", "--score-on-embeddings", tmp_path / "other_embeddings",
    )  # fmt: skip
    assert code == 0, err
    scores = json.loads((tmp_path / "features" / "scores.json").read_text())
    assert scores["out_of_domain"] and scores["dataset"] == "other" and scores["methods"] == ["tfidf"]
    assert "out of domain" in out


def test_interpret_scores_its_own_dataset_in_domain(pipeline, tmp_path, capsys):
    code, out, err = run(
        capsys, "interpret", *data_args(pipeline), "--sae", pipeline["sae"], "--features", "0", "1", "2",
        "--output", tmp_path / "features", "--llm", "fake", "--score", "--methods", "tfidf", "--no-progress",
        "--score-on", pipeline["beir"],
    )  # fmt: skip
    assert code == 0, err
    scores = json.loads((tmp_path / "features" / "scores.json").read_text())
    assert not scores["out_of_domain"] and "in domain" in out


def test_interpret_with_an_openai_server_needs_its_url(pipeline, tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    code, _, err = run(
        capsys, "interpret", *data_args(pipeline), "--sae", pipeline["sae"], "--output", tmp_path,
        "--llm", "openai", "--llm-model", "some-model",
    )  # fmt: skip
    assert code == 1 and "--base-url" in err


def test_interpret_score_needs_an_llm(pipeline, tmp_path, capsys):
    code, _, err = run(
        capsys, "interpret", *data_args(pipeline), "--sae", pipeline["sae"], "--output", tmp_path, "--score"
    )
    assert code == 1
    assert "--llm" in err


def test_explain_two_passages_of_the_dataset(pipeline, tmp_path, capsys):
    dataset = load_dataset(pipeline["beir"])
    query = dataset.queries["test"]["text"].iloc[0]
    doc_a, doc_b = dataset.corpus["text"].iloc[0], dataset.corpus["text"].iloc[20]
    code, out, err = run(
        capsys, "explain", *data_args(pipeline), "--sae", pipeline["sae"], "--query", query, "--doc-a", doc_a,
        "--doc-b", doc_b, "--n-features", "2", "--output", tmp_path / "explanation.json",
    )  # fmt: skip
    assert code == 0, err
    assert "passage A (rank" in out and "of the cosine of passage A" in out
    explanation = json.loads((tmp_path / "explanation.json").read_text())
    assert len(explanation["features"]) <= 2
    assert explanation["rank_a"] <= explanation["rank_b"]


@pytest.mark.network
def test_embed_with_a_small_encoder_then_explain_new_texts(tmp_path, capsys):
    """The path of a user: embed with a real model, train, then explain texts that are not in the dataset."""
    beir, embeddings, sae = tmp_path / "beir", tmp_path / "embeddings", tmp_path / "sae"
    write_beir_folder(beir)
    common = ["--dataset", beir, "--device", "cpu"]
    code, out, err = run(
        capsys, "embed", *common, "--model", "BAAI/bge-small-en-v1.5", "--output", embeddings, "--no-progress"
    )
    assert code == 0, err
    assert "384 dimensions" in out
    code, _, err = run(capsys, "train", *common, "--embeddings", embeddings, "--output-dir", sae, "--k", "8",
                       "--epochs", "3", "--batch-size", "16", "--no-progress")  # fmt: skip
    assert code == 0, err
    code, out, err = run(
        capsys, "explain", *common, "--embeddings", embeddings, "--sae", sae,
        "--query", "Can kidney stones be dissolved?", "--doc-a", "Renal colic from calcium stones.",
        "--doc-b", "Eczema flares respond to emollient creams.",
    )  # fmt: skip
    assert code == 0, err
    assert "passage A" in out


@pytest.mark.network
def test_pretrained_lists_the_hub(capsys):
    code, out, _ = run(capsys, "pretrained")
    assert code == 0
    assert out.strip()
