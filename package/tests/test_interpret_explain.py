"""Retrieval explanations must reproduce the four panels printed in the paper (Figures 2, 11 and 12)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp

from doctosae.interpret.explain import explain_codes, unit

# Printed values: (features, query, passage A, passage B, favoured passage per feature), L2 normalised codes.
PANELS = {
    "kidney stones": (
        [1473, 206, 2181, 2677],
        [0.581, 0.502, 0.255, 0.226],
        [0.514, 0.486, 0.078, 0.139],
        [0.317, 0.340, 0.270, 0.000],
        ["A", "A", "B", "A"],
    ),
    "oral retinoids": (
        [1912, 2617, 2399, 1460],
        [0.183, 0.400, 0.220, 0.400],
        [0.306, 0.375, 0.229, 0.000],
        [0.000, 0.248, 0.000, 0.109],
        ["A", "A", "A", "B"],
    ),
    "sun allergy": (
        [2746, 1573, 1939, 411],
        [0.395, 0.183, 0.239, 0.190],
        [0.380, 0.000, 0.000, 0.319],
        [0.097, 0.516, 0.327, 0.000],
        ["A", "B", "B", "A"],
    ),
    "aortic aneurysm": (
        [1512, 1016, 1163, 924],
        [0.386, 0.477, 0.343, 0.347],
        [0.000, 0.261, 0.310, 0.290],
        [0.331, 0.000, 0.000, 0.000],
        ["B", "A", "A", "A"],
    ),
}
N_LATENTS = 3072


def unit_code(features: list[int], values: list[float], filler: int) -> np.ndarray:
    """The printed values plus one private filler feature completing the vector to unit norm."""
    code = np.zeros(N_LATENTS)
    code[features] = values
    code[filler] = math.sqrt(1.0 - sum(v * v for v in values))
    return code


@pytest.mark.parametrize("panel", sorted(PANELS))
def test_explanation_reproduces_the_printed_panels(panel):
    features, query, passage_a, passage_b, favours = PANELS[panel]
    # Feature ids are not in the printed order: the explanation must recover it from the codes alone.
    explanation = explain_codes(
        unit_code(features, query, 3000),
        unit_code(features, passage_a, 3001),
        unit_code(features, passage_b, 3002),
    )
    assert explanation.features == features
    assert explanation.favours == favours
    assert explanation.query_values == pytest.approx(query)
    assert explanation.passage_a_values == pytest.approx(passage_a)
    assert explanation.passage_b_values == pytest.approx(passage_b)
    # The fillers never meet, so the four printed features make up the whole cosine.
    assert explanation.share_a == pytest.approx(1.0) and explanation.share_b == pytest.approx(1.0)
    assert explanation.cosine_a == pytest.approx(sum(q * a for q, a in zip(query, passage_a, strict=True)))


def test_renormalising_only_the_printed_entries_would_change_the_order():
    features, query, passage_a, passage_b, _ = PANELS["kidney stones"]
    code = np.zeros(N_LATENTS)
    truncated = [code.copy() for _ in range(3)]
    for vector, values in zip(truncated, (query, passage_a, passage_b), strict=True):
        vector[features] = values
    assert explain_codes(*truncated).features != features


def test_share_of_cosine_counts_only_the_shown_features():
    q = np.array([0.6, 0.6, 0.4, 0.3])
    a = np.array([0.5, 0.1, 0.5, 0.7])
    b = np.array([0.1, 0.6, 0.5, 0.6])
    explanation = explain_codes(q, a, b, n_features=2)
    qn, an, bn = unit(q), unit(a), unit(b)
    delta = qn * (an - bn)
    expected = list(np.argsort(-np.abs(delta), kind="stable")[:2])
    assert explanation.features == expected
    assert explanation.cosine_a == pytest.approx(float(qn @ an))
    assert explanation.share_a == pytest.approx(float((qn * an)[expected].sum() / (qn @ an)))
    assert explanation.share_b == pytest.approx(float((qn * bn)[expected].sum() / (qn @ bn)))
    assert explanation.delta == pytest.approx(delta[expected].tolist())


def test_features_inactive_on_the_query_never_appear_and_zero_codes_are_safe():
    q = sp.csr_matrix(np.array([[0.0, 1.0, 0.0]]))
    explanation = explain_codes(q, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    assert explanation.features == []
    assert explanation.cosine_a == 0.0 and explanation.share_a == 0.0
    assert unit(np.zeros(3)).tolist() == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="sizes"):
        explain_codes(np.ones(3), np.ones(4), np.ones(3))


def test_frame_text_and_names():
    features, query, passage_a, passage_b, _ = PANELS["kidney stones"]
    explanation = explain_codes(
        unit_code(features, query, 3000),
        unit_code(features, passage_a, 3001),
        unit_code(features, passage_b, 3002),
    )
    named = explanation.with_names({1473: "quality of life", 206: "kidney stones"})
    assert named.names == ["quality of life", "kidney stones", None, None]
    assert explanation.names == []
    frame = named.to_frame()
    assert frame["feature"].tolist() == features
    assert frame["favours"].tolist() == ["A", "A", "B", "A"]
    assert frame["contribution_a"].tolist() == pytest.approx(
        [q * a for q, a in zip(query, passage_a, strict=True)]
    )
    named.query, named.passage_a, named.rank_a, named.rank_b = (
        "kidney stones and quality of life?",
        "x " * 300,
        1,
        6,
    )
    text = named.to_text()
    assert "passage A (rank 1)" in text and "quality of life" in text and "0.581" in text
    assert "These 4 features make up 100% of the cosine of passage A" in text
    assert str(named) == text
    assert named.to_dict()["features"] == features
