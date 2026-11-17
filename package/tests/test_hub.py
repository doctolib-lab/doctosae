"""Hub helpers without network: checkpoint ids, released and pushed checkpoints, paper conversion, errors."""

import json
import types
from pathlib import Path

import pytest
import torch
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

import doctosae.hub as hub
from doctosae.sae import DoctoSAE


def _sae(seed: int = 0, **kwargs) -> DoctoSAE:
    torch.manual_seed(seed)
    return DoctoSAE(**({"d_in": 8, "n_latents": 24, "k": 4} | kwargs))


def _hub_error(error_class: type[Exception]) -> Exception:
    """An instance of a Hub HTTP error without an HTTP response object (only the type matters here)."""
    error = error_class.__new__(error_class)
    Exception.__init__(error, "simulated Hub error")
    return error


def test_aliases_and_repository_names():
    assert hub.HUB_ORG == "doctolib-lab"
    assert hub.resolve_embedder("BGE") == "BAAI/bge-large-en-v1.5"
    assert hub.resolve_embedder("baai/bge-large-en-v1.5") == "BAAI/bge-large-en-v1.5"  # the Hub ignores case
    assert hub.resolve_embedder("baai/bge-small-en-v1.5") == "BAAI/bge-small-en-v1.5"
    assert hub.resolve_embedder("medembed") == "abhinand/MedEmbed-large-v0.1"
    assert hub.resolve_embedder("some/model") == "some/model"
    assert hub.pretrained_repo_id() == "doctolib-lab/doctosae-bge-large-en-v1.5"
    assert hub.pretrained_repo_id("medembed") == "doctolib-lab/doctosae-medembed-large-v0.1"
    assert hub.pretrained_repo_id("gemini") == "doctolib-lab/doctosae-gemini-embedding-001"
    assert hub.pretrained_repo_id("BAAI/bge-large-en-v1.5", org="me") == "me/doctosae-bge-large-en-v1.5"


def _paper_run(folder: Path, sae: DoctoSAE, *, prefix: str = "", subfolder: str = "", **config) -> Path:
    """A run folder as written by the paper's train_sae.py: sae_config.json and a .pth state dict."""
    folder.mkdir(parents=True, exist_ok=True)
    paper_config = {
        "input_dim": sae.d_in, "hidden": sae.n_latents, "sparsity_type": "topknorelu", "intermediate": None,
        "k": sae.k, "activation_fn": "relu", "init": "default", "beta": 0.005, "lambda": 1e-05,
        "alpha_q": 0.1, "temperature": 0.05, "recon_loss": "mse", "k_aux": None, "aux_coef": None,
    } | config  # fmt: skip
    (folder / "sae_config.json").write_text(json.dumps(paper_config))
    (folder / subfolder).mkdir(exist_ok=True)
    state = {prefix + name: tensor for name, tensor in sae.state_dict().items()}
    torch.save(state, folder / subfolder / "best_sae.pth")
    return folder


def test_from_paper_checkpoint(tmp_path):
    sae = _sae()
    with torch.no_grad():
        sae.b_pre.normal_()
    loaded = hub.from_paper_checkpoint(
        _paper_run(tmp_path / "run", sae), embedder="bge", query_instruction="q: "
    )
    for name, tensor in sae.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor)
    assert (loaded.k, loaded.embedder, loaded.query_instruction) == (4, "BAAI/bge-large-en-v1.5", "q: ")
    assert loaded.metadata["license"] == "cc-by-nc-4.0" and loaded.metadata["datasets"] == ["clinia/CUREv1"]
    assert loaded.metadata["train_dataset"] == "cure" and loaded.metadata["paper_config"]["lambda"] == 1e-05
    x = torch.randn(3, 8)
    assert torch.equal(loaded.encode(x), sae.encode(x))


def test_from_paper_checkpoint_variants(tmp_path):
    sae = _sae()
    data_parallel = _paper_run(
        tmp_path / "dp", sae, prefix="module.", subfolder="checkpoints", input_dim=None
    )
    loaded = hub.from_paper_checkpoint(data_parallel)
    assert loaded.d_in == 8 and torch.equal(loaded.decoder.weight, sae.decoder.weight)
    with pytest.raises(FileNotFoundError, match="best_sae_loss.pth"):
        hub.from_paper_checkpoint(data_parallel, checkpoint="best_sae_loss.pth")
    with pytest.raises(ValueError, match="l1"):
        hub.from_paper_checkpoint(_paper_run(tmp_path / "l1", sae, sparsity_type="l1"))


BGE_REPO = "doctolib-lab/doctosae-bge-large-en-v1.5"
# The layout of the released repositories: one folder per configuration, a stray file at the root.
RELEASED_FILES = [
    ".gitattributes", "README.md", "k256_m24/cfg.json", "k256_m24/sae.safetensors", "k4_m24/cfg.json",
    "k4_m24/sae.safetensors", "k4_m48/cfg.json", "k4_m48/sae.safetensors", "k8_m24/cfg.json",
]  # fmt: skip


class FakeApi:
    """HfApi with fixed repository files; records the calls."""

    files: list[str] = RELEASED_FILES
    calls: list[tuple] = []

    def __init__(self, token=None):
        self.token = token

    def list_repo_files(self, repo_id, revision=None):
        FakeApi.calls.append(("list_repo_files", repo_id, revision, self.token))
        return list(self.files)


@pytest.fixture
def fake_api(monkeypatch):
    FakeApi.files, FakeApi.calls = RELEASED_FILES, []
    monkeypatch.setattr(hub, "HfApi", FakeApi)
    return FakeApi


def _released_folder(folder: Path, sae: DoctoSAE, backbone: str = "BAAI/bge-large-en-v1.5") -> Path:
    """A folder of a released repository: cfg.json and sae.safetensors with the paper parameter names."""
    from safetensors.torch import save_file

    folder.mkdir(parents=True, exist_ok=True)
    config = {
        "architecture": "TopkNoReluSparseAutoencoder", "backbone": backbone, "input_dim": sae.d_in,
        "hidden_dim": sae.n_latents, "k": sae.k, "sparsity_type": "topknorelu", "activation_fn": "relu",
        "metrics": {"cure_test_ndcg_at_10": 0.5},
    }  # fmt: skip
    (folder / "cfg.json").write_text(json.dumps(config))
    save_file(
        {name: tensor.contiguous() for name, tensor in sae.state_dict().items()}, folder / "sae.safetensors"
    )
    return folder


def test_pretrained_checkpoint_id_picks_a_folder_of_the_repository(fake_api):
    assert hub.pretrained_checkpoint_id("bge") == f"{BGE_REPO}/k4_m24"  # smallest k, then smallest m
    assert hub.pretrained_checkpoint_id("bge", k=256, revision="v1") == f"{BGE_REPO}/k256_m24"
    assert fake_api.calls[-1] == ("list_repo_files", BGE_REPO, "v1", None)
    with pytest.raises(FileNotFoundError, match=r"k=8 .*k4_m24, .*k4_m48, .*k256_m24"):
        hub.pretrained_checkpoint_id("bge", k=8)  # k8_m24 has no weights
    calls = len(fake_api.calls)
    assert hub.pretrained_checkpoint_id(repo_id="me/repo/k64_m3072") == "me/repo/k64_m3072"
    assert len(fake_api.calls) == calls  # a full checkpoint id needs no listing
    fake_api.files = ["README.md", "config.json", "model.safetensors"]
    assert hub.pretrained_checkpoint_id(repo_id="me/my-sae", k=4) == "me/my-sae"
    fake_api.files = ["README.md"]
    with pytest.raises(FileNotFoundError, match="no DoctoSAE checkpoint"):
        hub.pretrained_checkpoint_id("bge")


def test_load_pretrained_reads_a_released_checkpoint(tmp_path, monkeypatch, fake_api):
    sae = _sae(n_latents=24, k=4)
    with torch.no_grad():
        sae.b_pre.normal_()
    _released_folder(tmp_path / "k4_m24", sae)
    requested = []

    def fake_download(repo_id, filename, **kwargs):
        requested.append((repo_id, filename, kwargs["library_name"], kwargs["revision"]))
        return str(tmp_path / filename)

    monkeypatch.setattr(hub, "hf_hub_download", fake_download)
    loaded = hub.load_pretrained("bge", k=4, revision="main")
    assert requested == [
        (BGE_REPO, "k4_m24/cfg.json", "doctosae", "main"),
        (BGE_REPO, "k4_m24/sae.safetensors", "doctosae", "main"),
    ]
    for name, tensor in sae.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor)
    assert (loaded.k, loaded.embedder, loaded.training) == (4, "BAAI/bge-large-en-v1.5", False)
    assert loaded.query_instruction == "Represent this sentence for searching relevant passages: "
    assert loaded.metadata["hub_checkpoint"] == f"{BGE_REPO}/k4_m24"
    assert loaded.metadata["license"] == "cc-by-nc-4.0" and loaded.metadata["datasets"] == ["clinia/CUREv1"]
    assert loaded.metadata["paper_config"]["metrics"] == {"cure_test_ndcg_at_10": 0.5}
    x = torch.randn(3, 8)
    assert torch.equal(loaded.encode(x), sae.encode(x))
    overridden = hub.load_pretrained(repo_id=f"{BGE_REPO}/k4_m24", query_instruction="")
    assert overridden.query_instruction == ""
    with pytest.raises(ValueError, match="k=4, not k=8"):
        hub.load_pretrained(repo_id=f"{BGE_REPO}/k4_m24", k=8)


def test_released_checkpoints_of_other_encoders_get_their_own_instruction(tmp_path, monkeypatch):
    _released_folder(tmp_path / "k4_m24", _sae(n_latents=24, k=4), backbone="abhinand/MedEmbed-large-v0.1")
    monkeypatch.setattr(hub, "hf_hub_download", lambda repo_id, filename, **kwargs: str(tmp_path / filename))
    loaded = hub.load_pretrained(repo_id="doctolib-lab/doctosae-medembed-large-v0.1/k4_m24")
    assert (loaded.embedder, loaded.query_instruction) == ("abhinand/MedEmbed-large-v0.1", "")


def test_load_pretrained_reads_a_repository_written_by_push_to_hub(tmp_path, monkeypatch, fake_api):
    sae = _sae(embedder="BAAI/bge-large-en-v1.5")
    sae.save_pretrained(tmp_path)
    fake_api.files = [".gitattributes", "README.md", "config.json", "model.safetensors"]
    requested = []

    def fake_download(repo_id, filename, **kwargs):
        requested.append((repo_id, filename))
        return str(tmp_path / filename)

    monkeypatch.setattr(hub, "hf_hub_download", fake_download)
    loaded = hub.load_pretrained(repo_id="me/my-sae", query_instruction="query: ")
    assert requested == [("me/my-sae", "config.json"), ("me/my-sae", "model.safetensors")]
    assert torch.equal(loaded.encoder.weight, sae.encoder.weight) and loaded.query_instruction == "query: "
    assert hub.load_pretrained(repo_id="me/my-sae", k=4).k == 4
    with pytest.raises(ValueError, match="k=4, not k=8"):
        hub.load_pretrained(repo_id="me/my-sae", k=8)


@pytest.mark.parametrize(
    ("error_class", "expected"),
    [(GatedRepoError, PermissionError), (RepositoryNotFoundError, FileNotFoundError),
     (RevisionNotFoundError, FileNotFoundError), (EntryNotFoundError, FileNotFoundError)],
)  # fmt: skip
def test_load_pretrained_maps_hub_errors(monkeypatch, fake_api, error_class, expected):
    def failing_download(*args, **kwargs):
        raise _hub_error(error_class)

    monkeypatch.setattr(hub, "hf_hub_download", failing_download)
    monkeypatch.setattr(hub, "list_pretrained", lambda: [f"{BGE_REPO}/k64_m3072"])
    with pytest.raises(expected):
        hub.load_pretrained("bge", k=4)


def test_listing_errors_are_mapped_too(monkeypatch):
    class MissingRepo:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, revision=None):
            raise _hub_error(RepositoryNotFoundError)

    monkeypatch.setattr(hub, "HfApi", MissingRepo)
    monkeypatch.setattr(hub, "list_pretrained", lambda: [])
    with pytest.raises(FileNotFoundError, match="none yet"):
        hub.load_pretrained("some/encoder")


def test_list_pretrained(monkeypatch):
    calls = {}

    def model(model_id, *files):
        return types.SimpleNamespace(
            id=model_id, siblings=[types.SimpleNamespace(rfilename=f) for f in files]
        )

    class ListingApi:
        def __init__(self, token=None):
            calls["token"] = token

        def list_models(self, **kwargs):
            calls.update(kwargs)
            return [
                model("doctolib-lab/doctosae-gemini", "k256_m9/cfg.json", "k256_m9/sae.safetensors",
                      "k128_m9/cfg.json", "k128_m9/sae.safetensors"),
                model("doctolib-lab/doctosae-mine", "README.md", "config.json", "model.safetensors"),
                model("doctolib-lab/doctosae-notes", "README.md"),
            ]  # fmt: skip

    monkeypatch.setattr(hub, "HfApi", ListingApi)
    assert hub.list_pretrained() == [
        "doctolib-lab/doctosae-gemini/k128_m9",
        "doctolib-lab/doctosae-gemini/k256_m9",
        "doctolib-lab/doctosae-mine",
    ]
    assert calls == {"token": None, "author": "doctolib-lab", "search": "doctosae", "expand": ["siblings"]}


def test_push_to_hub_is_private_and_carries_the_data_terms(monkeypatch):
    sae = _sae(metadata={"license": "cc-by-nc-4.0", "datasets": ["clinia/CUREv1"]})
    pushed = {}
    monkeypatch.setattr(
        sae, "push_to_hub", lambda repo_id, **kwargs: pushed.update(repo_id=repo_id, **kwargs)
    )
    hub.push_to_hub(sae, "me/my-sae")
    assert pushed["repo_id"] == "me/my-sae" and pushed["private"] is True
    assert sae.metadata["license"] == "cc-by-nc-4.0"
    hub.push_to_hub(sae, "me/my-sae", private=False, license="cc-by-4.0", datasets=["BeIR/nfcorpus"])
    assert pushed["private"] is False
    assert sae.metadata["license"] == "cc-by-4.0" and sae.metadata["datasets"] == ["BeIR/nfcorpus"]


def test_push_to_hub_refuses_a_local_embedder_path(monkeypatch, tmp_path):
    sae = _sae(embedder=str(tmp_path))
    monkeypatch.setattr(sae, "push_to_hub", lambda *args, **kwargs: pytest.fail("nothing may be uploaded"))
    with pytest.raises(ValueError, match="Hub id"):
        hub.push_to_hub(sae, "me/my-sae")
