"""Hub helpers without network: repository names, paper checkpoint conversion and error mapping."""

import json
import types
from pathlib import Path

import pytest
import torch
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError

import medsae.hub as hub
from medsae.sae import MedSAE


def _sae(seed: int = 0, **kwargs) -> MedSAE:
    torch.manual_seed(seed)
    return MedSAE(**({"d_in": 8, "n_latents": 24, "k": 4} | kwargs))


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
    assert hub.pretrained_repo_id() == "doctolib-lab/medsae-bge-large-en-v1.5-k64"
    assert hub.pretrained_repo_id("medembed", k=128) == "doctolib-lab/medsae-medembed-large-v0.1-k128"
    assert hub.pretrained_repo_id("gemini", 256) == "doctolib-lab/medsae-gemini-embedding-001-k256"
    assert hub.pretrained_repo_id("BAAI/bge-large-en-v1.5", 64, org="me") == "me/medsae-bge-large-en-v1.5-k64"


def _paper_run(folder: Path, sae: MedSAE, *, prefix: str = "", subfolder: str = "", **config) -> Path:
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


def test_load_pretrained_reads_the_downloaded_files(tmp_path, monkeypatch):
    sae = _sae(embedder="BAAI/bge-large-en-v1.5")
    sae.save_pretrained(tmp_path)
    requested = []

    def fake_download(repo_id, filename, **kwargs):
        requested.append((repo_id, filename, kwargs["library_name"]))
        return str(tmp_path / filename)

    monkeypatch.setattr(hub, "hf_hub_download", fake_download)
    loaded = hub.load_pretrained("bge", k=4)
    assert [name for _, name, _ in requested] == ["config.json", "model.safetensors"]
    assert {repo for repo, _, _ in requested} == {"doctolib-lab/medsae-bge-large-en-v1.5-k4"}
    assert {library for _, _, library in requested} == {"medsae"}
    assert torch.equal(loaded.encoder.weight, sae.encoder.weight)
    overridden = hub.load_pretrained(repo_id="me/my-sae", query_instruction="query: ")
    assert requested[-1][0] == "me/my-sae" and overridden.query_instruction == "query: "


@pytest.mark.parametrize(
    ("error_class", "expected"),
    [(GatedRepoError, PermissionError), (RepositoryNotFoundError, FileNotFoundError),
     (RevisionNotFoundError, FileNotFoundError)],
)  # fmt: skip
def test_load_pretrained_maps_hub_errors(monkeypatch, error_class, expected):
    def failing_download(*args, **kwargs):
        raise _hub_error(error_class)

    monkeypatch.setattr(hub, "hf_hub_download", failing_download)
    monkeypatch.setattr(hub, "list_pretrained", lambda: ["doctolib-lab/medsae-bge-large-en-v1.5-k64"])
    with pytest.raises(expected):
        hub.load_pretrained("bge", k=32)


def test_list_pretrained(monkeypatch):
    calls = {}

    class FakeApi:
        def __init__(self, token=None):
            calls["token"] = token

        def list_models(self, **kwargs):
            calls.update(kwargs)
            return [types.SimpleNamespace(id="doctolib-lab/b"), types.SimpleNamespace(id="doctolib-lab/a")]

    monkeypatch.setattr(hub, "HfApi", FakeApi)
    assert hub.list_pretrained() == ["doctolib-lab/a", "doctolib-lab/b"]
    assert calls == {"token": None, "author": "doctolib-lab", "filter": "medsae"}


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
