"""Loss terms: reductions, the alpha_q weighting, positive masks and InfoNCE."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from doctosae.training.losses import (
    info_nce_loss,
    positive_mask,
    reconstruction_loss,
    squared_error,
    weighted_loss,
)


def test_reconstruction_loss_averages_over_rows_and_dimensions():
    x = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    x_hat = torch.tensor([[1.0, 0.0, 3.0], [1.0, 1.0, 1.0]])
    assert reconstruction_loss(x, x_hat).item() == pytest.approx((4 + 1 + 1 + 1) / 6)


def test_squared_error_sums_dimensions_and_averages_rows():
    x = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    x_hat = torch.tensor([[1.0, 0.0, 3.0], [1.0, 1.0, 1.0]])
    assert squared_error(x, x_hat).item() == pytest.approx((4 + 3) / 2)


@pytest.mark.parametrize(
    ("alpha_q", "expected"),
    [(0.0, 2.0), (1.0, (2.0 * 4 + 10.0 * 2) / 6), (0.1, (2.0 * 4 + 0.1 * 10.0 * 2) / (4 + 0.1 * 2))],
)
def test_weighted_loss(alpha_q, expected):
    value = weighted_loss(torch.tensor(2.0), torch.tensor(10.0), 4, 2, alpha_q)
    assert value.item() == pytest.approx(expected)


def test_weighted_loss_with_equal_batches_is_the_paper_formula():
    """With N_c = N_q, the paper loss is (MSE_c + 0.1 MSE_q) / 1.1."""
    value = weighted_loss(torch.tensor(0.3), torch.tensor(0.7), 512, 512, 0.1)
    assert value.item() == pytest.approx((0.3 + 0.1 * 0.7) / 1.1)


def test_positive_mask_marks_every_column_of_a_positive_document():
    positives = {0: frozenset({5}), 1: frozenset({6, 9})}
    mask = positive_mask([0, 1, 2], [5, 6, 5, 7], positives)
    expected = torch.tensor(
        [[True, False, True, False], [False, True, False, False], [False, False, False, False]]
    )
    assert torch.equal(mask, expected)


def test_positive_mask_skip_diagonal_keeps_each_anchor_positive():
    positives = {0: frozenset({10, 11}), 1: frozenset({10, 11})}
    mask = positive_mask(np.array([0, 1]), torch.tensor([10, 11]), positives, skip_diagonal=True)
    assert torch.equal(mask, torch.tensor([[False, True], [True, False]]))


def test_positive_mask_without_positives_is_empty():
    mask = positive_mask([3, 4], [1, 2, 3], {})
    assert mask.shape == (2, 3) and not mask.any()


def _manual_info_nce(zq, zd, zn, mask, negative_mask, temperature):
    """Reference computation, written independently of the implementation."""
    losses = []
    for i in range(len(zq)):
        candidates = [(zd[j], mask[i, j]) for j in range(len(zd))]
        candidates += [(zn[j], negative_mask[i, j]) for j in range(len(zn))]
        logits = []
        for j, (code, masked) in enumerate(candidates):
            cosine = F.cosine_similarity(zq[i], code, dim=0, eps=1e-12).item() if code.norm() > 0 else 0.0
            logits.append(-math.inf if (masked and j != i) else cosine / temperature)
        log_norm = math.log(sum(math.exp(value) for value in logits if value != -math.inf))
        losses.append(log_norm - logits[i])
    return sum(losses) / len(losses)


def test_info_nce_matches_a_direct_computation():
    torch.manual_seed(0)
    zq, zd, zn = torch.randn(4, 6), torch.randn(4, 6), torch.randn(5, 6)
    mask = torch.zeros(4, 4, dtype=torch.bool)
    mask[0, 2] = True
    negative_mask = torch.zeros(4, 5, dtype=torch.bool)
    negative_mask[1, 3] = negative_mask[3, 0] = True
    loss = info_nce_loss(zq, zd, zn, mask, negative_mask, temperature=0.5)
    assert loss.item() == pytest.approx(_manual_info_nce(zq, zd, zn, mask, negative_mask, 0.5), rel=1e-5)


def test_info_nce_ignores_masked_documents():
    torch.manual_seed(1)
    zq, zd, zn = torch.randn(3, 8), torch.randn(3, 8), torch.randn(4, 8)
    mask = torch.zeros(3, 3, dtype=torch.bool)
    negative_mask = torch.zeros(3, 4, dtype=torch.bool)
    negative_mask[:, 1] = True  # document 1 of the negatives is a positive of every anchor
    reference = info_nce_loss(zq, zd, zn, mask, negative_mask)
    changed = zn.clone()
    changed[1] = torch.randn(8) * 100
    assert info_nce_loss(zq, zd, changed, mask, negative_mask).item() == pytest.approx(reference.item())


def test_info_nce_handles_zero_codes_and_propagates_gradients():
    zq = torch.randn(3, 5, requires_grad=True)
    zd = torch.randn(3, 5, requires_grad=True)
    zn = torch.zeros(2, 5, requires_grad=True)  # empty codes: cosine 0 with everything
    loss = info_nce_loss(zq, zd, zn)
    assert torch.isfinite(loss)
    loss.backward()
    assert zq.grad.abs().sum() > 0 and zd.grad.abs().sum() > 0


def test_info_nce_without_negatives_is_in_batch_cross_entropy():
    torch.manual_seed(2)
    zq, zd = torch.randn(4, 3), torch.randn(4, 3)
    logits = F.normalize(zq, dim=1) @ F.normalize(zd, dim=1).t() / 0.1
    expected = F.cross_entropy(logits, torch.arange(4))
    assert info_nce_loss(zq, zd, temperature=0.1).item() == pytest.approx(expected.item(), rel=1e-6)
