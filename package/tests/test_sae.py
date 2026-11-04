"""MedSAE model: TopK encoding, decoder constraints, b_pre initialisation and Hub serialisation (offline)."""

import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from medsae.sae import MedSAE, geometric_median


def _sae(d_in: int = 8, n_latents: int = 24, k: int = 4, seed: int = 0, **kwargs) -> MedSAE:
    torch.manual_seed(seed)
    return MedSAE(d_in=d_in, n_latents=n_latents, k=k, **kwargs)


def test_parameters_are_named_like_the_paper_code():
    sae = _sae()
    shapes = {name: tuple(tensor.shape) for name, tensor in sae.state_dict().items()}
    assert shapes == {
        "encoder.weight": (24, 8),
        "encoder.bias": (24,),
        "decoder.weight": (8, 24),
        "b_pre": (8,),
    }
    assert (sae.d_in, sae.n_latents, sae.k) == (8, 24, 4)
    assert sae.device == torch.device("cpu") and "k=4" in sae.extra_repr()


def test_initial_decoder_is_the_normalised_encoder_transpose():
    sae = _sae()
    expected = sae.encoder.weight.t() / sae.encoder.weight.t().norm(dim=0, keepdim=True)
    assert torch.allclose(sae.decoder.weight, expected, atol=1e-6)
    assert torch.allclose(sae.decoder.weight.norm(dim=0), torch.ones(24), atol=1e-6)
    assert sae.encoder.bias.abs().sum() > 0  # the paper keeps PyTorch's default bias (not zeroed)
    assert not sae.b_pre.any()


def test_encode_is_relu_of_the_top_k_pre_activations():
    sae, x = _sae(), torch.randn(16, 8)
    z = sae.encode(x)
    pre = sae.pre_acts(x)
    assert torch.allclose(pre, sae.encoder(x - sae.b_pre))
    expected = torch.zeros_like(pre)
    top = pre.topk(4, dim=1)
    expected.scatter_(1, top.indices, torch.relu(top.values))
    assert torch.equal(z, expected)
    assert ((z != 0).sum(dim=1) <= 4).all()


def test_selection_happens_before_the_relu():
    sae = _sae()
    with torch.no_grad():
        sae.encoder.bias.fill_(-10.0)
        sae.encoder.bias[0] = 100.0  # one feature is always active, the other selected ones are negative
    z = sae.encode(torch.randn(5, 8))
    assert ((z != 0).sum(dim=1) == 1).all()  # fewer than k non zeros per row


def test_encode_k_override_and_linear_activation():
    sae, x = _sae(), torch.randn(6, 8)
    assert ((sae.encode(x, k=2) != 0).sum(dim=1) <= 2).all()
    linear = _sae(activation="none")
    with torch.no_grad():
        linear.encoder.bias.fill_(-10.0)
    z = linear.encode(x)
    assert ((z != 0).sum(dim=1) == 4).all() and (z <= 0).all()


@pytest.mark.parametrize("kwargs", [{"activation": "gelu"}, {"k": 0}, {"k": 25}])
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        _sae(**kwargs)


def test_forward_and_decode():
    sae, x = _sae(), torch.randn(3, 8)
    with torch.no_grad():
        sae.b_pre.fill_(0.5)
    x_hat, z = sae(x)
    assert torch.equal(z, sae.encode(x))
    assert torch.allclose(x_hat, sae.decoder(z) + 0.5)
    assert torch.allclose(sae.decode(torch.zeros(1, 24)), torch.full((1, 8), 0.5))


def test_decoder_renormalisation_and_gradient_projection():
    sae = _sae()
    sae.project_decoder_gradient()  # no gradient yet: nothing to do
    with torch.no_grad():
        sae.decoder.weight.mul_(torch.linspace(0.5, 3.0, 24))
    sae.normalize_decoder()
    assert torch.allclose(sae.decoder.weight.norm(dim=0), torch.ones(24), atol=1e-6)
    sae.decoder.weight.grad = torch.randn(8, 24)
    sae.project_decoder_gradient()
    parallel = (sae.decoder.weight.grad * sae.decoder.weight).sum(dim=0)
    assert torch.allclose(parallel, torch.zeros(24), atol=1e-6)


def test_init_b_pre_is_the_geometric_median():
    sae = _sae()
    x = np.random.default_rng(0).normal(size=(50, 8)).astype(np.float32)
    sae.init_b_pre(x, seed=3)
    assert np.allclose(sae.b_pre.detach().numpy(), geometric_median(x, seed=3))


def test_geometric_median_properties():
    line = np.array([[0.0], [1.0], [2.0], [3.0], [100.0]])
    assert geometric_median(line)[0] == pytest.approx(2.0, abs=1e-3)  # robust to the outlier, unlike the mean
    rng = np.random.default_rng(1)
    cloud = rng.normal(size=(400, 3)).astype(np.float32)
    median = geometric_median(cloud)
    assert median.dtype == np.float32
    np.testing.assert_allclose(geometric_median(cloud + 5.0), median + 5.0, atol=1e-3)
    np.testing.assert_array_equal(geometric_median(torch.from_numpy(cloud)), median)
    subsampled = geometric_median(cloud, max_samples=50, seed=4)
    np.testing.assert_array_equal(subsampled, geometric_median(cloud, max_samples=50, seed=4))
    assert not np.array_equal(subsampled, geometric_median(cloud, max_samples=50, seed=5))


def test_aux_recon_uses_dead_features_only():
    sae, x = _sae(), torch.randn(4, 8)
    dead = torch.zeros(24, dtype=torch.bool)
    dead[[1, 5, 7]] = True
    codes = sae.topk(sae.pre_acts(x), 2, allowed=dead)
    assert (codes[:, ~dead] == 0).all()
    assert torch.allclose(sae.aux_recon(x, dead, k_aux=2), sae.decoder(codes))  # no b_pre
    with pytest.raises(ValueError):
        sae.aux_recon(x, torch.zeros(24, dtype=torch.bool), k_aux=2)


def test_config_follows_live_attributes_and_is_json():
    sae = _sae(embedder="BAAI/bge-large-en-v1.5", metadata={"score": np.float32(0.5), "n": np.int64(3)})
    sae.k = 2
    sae.metadata["license"] = "cc-by-4.0"
    config = sae.get_config()
    assert config["k"] == 2 and config["metadata"] == {"score": 0.5, "n": 3, "license": "cc-by-4.0"}
    json.dumps(config)


def test_save_and_load_round_trip(tmp_path):
    sae = _sae(
        embedder="BAAI/bge-large-en-v1.5", query_instruction="q: ", metadata={"license": "cc-by-nc-4.0"}
    )
    with torch.no_grad():
        sae.b_pre.normal_()
    sae.metadata["datasets"] = ["clinia/CUREv1"]
    sae.save_pretrained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["d_in"] == 8 and config["n_latents"] == 24 and config["query_instruction"] == "q: "
    loaded = MedSAE.from_pretrained(tmp_path)
    assert (loaded.k, loaded.embedder, loaded.metadata) == (4, sae.embedder, sae.metadata)
    for name, tensor in sae.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor)
    assert MedSAE.from_pretrained(tmp_path, k=2).k == 2
    card = (tmp_path / "README.md").read_text()
    assert "license: cc-by-nc-4.0" in card and "clinia/CUREv1" in card
    assert "base_model: BAAI/bge-large-en-v1.5" in card and "base_model_relation: adapter" in card
    assert "k4" in card


def test_model_card_text_carries_the_terms_of_the_training_data():
    sae = _sae(
        embedder="bge",
        metadata={"train_dataset": "cure", "datasets": ["clinia/CUREv1"], "license": "cc-by-nc-4.0"},
    )
    card = str(sae.generate_model_card())
    assert "`BAAI/bge-large-en-v1.5`" in card and "base_model_relation: adapter" in card
    assert "cure (`clinia/CUREv1`), license `cc-by-nc-4.0`" in card and "Clinia" in card
    assert "noncommercial" in card and "## Responsible use" in card and "clinical decision" in card
    assert "Paper:" not in card and "More Information Needed" not in card
    local = _sae(embedder="/home/someone/models/bge").generate_model_card()
    assert local.data.base_model is None and "a text encoder" in str(local)


def test_loading_a_local_folder_prints_nothing(tmp_path, capsys):
    sae = _sae()
    sae.save_pretrained(tmp_path)
    loaded = MedSAE.from_pretrained(tmp_path)
    assert capsys.readouterr().out == "" and not loaded.training
    assert torch.equal(loaded.encoder.weight, sae.encoder.weight)


def test_model_card_defaults_to_the_package_license():
    card = _sae().generate_model_card()
    assert card.data.license == "apache-2.0" and card.data.library_name == "medsae"


def test_loading_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.json"):
        MedSAE.from_pretrained(tmp_path)
    sae = _sae()
    sae.save_pretrained(tmp_path)
    state = {name: tensor for name, tensor in sae.state_dict().items() if name != "b_pre"}
    save_file(state, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="b_pre"):
        MedSAE.from_pretrained(tmp_path)
