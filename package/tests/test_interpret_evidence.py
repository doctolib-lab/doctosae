"""Evidence of a feature: firing counts, top documents per article, negatives, selection and TF IDF terms."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from medsae.interpret.activations import (
    ActivationIndex,
    CodedTexts,
    article_keys,
    dedup_first,
    firing_counts,
    negative_rows,
    snippet_titles,
    top_activating,
    top_per_article,
)
from medsae.interpret.selection import select_features, theme_entropy
from medsae.interpret.tfidf import TfidfTerms


def column_codes(values: list[float], n_latents: int = 3, feature: int = 1) -> sp.csc_matrix:
    dense = np.zeros((len(values), n_latents), dtype=np.float32)
    dense[:, feature] = values
    return sp.csc_matrix(dense)


def test_dedup_keeps_first_occurrences_in_order():
    assert dedup_first(["a", "b", "a", "c", "b"]).tolist() == [0, 1, 3]


def test_firing_counts_only_count_strictly_positive_values():
    codes = sp.csr_matrix(np.array([[0.5, -0.2, 0.0], [0.1, 0.3, 0.0]], dtype=np.float32))
    assert firing_counts(codes).tolist() == [2, 1, 0]


def test_top_activating_is_strongest_first_among_firing_rows():
    codes = column_codes([0.1, 0.0, 0.7, -0.4, 0.3])
    assert top_activating(codes, 1, 10).tolist() == [2, 4, 0]
    assert top_activating(codes, 1, 2).tolist() == [2, 4]
    assert top_activating(codes, 0, 5).tolist() == []


def test_top_documents_keep_at_most_two_passages_per_article():
    codes = column_codes([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    articles = ["a", "a", "a", "b", "", "", "c"]
    # The third passage of article "a" is skipped; untitled rows are never capped.
    assert top_per_article(codes, 1, 5, articles).tolist() == [0, 1, 3, 4, 5]
    assert top_per_article(codes, 1, 5, None).tolist() == [0, 1, 2, 3, 4]
    assert top_per_article(codes, 1, 0, articles).tolist() == []


def test_negatives_use_a_fresh_generator_per_feature():
    dense = np.zeros((50, 2), dtype=np.float32)
    dense[::3, 0] = 1.0
    dense[::5, 1] = 1.0
    codes = sp.csc_matrix(dense)
    for feature in (0, 1):
        candidates = np.flatnonzero(dense[:, feature] <= 0)
        expected = np.sort(np.random.default_rng(0).choice(candidates, 2, replace=False))
        assert negative_rows(codes, feature, 2).tolist() == expected.tolist()
    assert negative_rows(codes, 0, 2).tolist() == negative_rows(codes, 0, 2).tolist()
    assert negative_rows(sp.csc_matrix(np.ones((3, 1))), 0, 2).tolist() == []


def test_snippet_titles_require_the_title_to_rebuild_the_text():
    frame = pd.DataFrame(
        {
            "text": ["Title A body a", "Title B body b", "body only", "Title D  spaced", "nan x"],
            "title": ["Title A", "Title B ", None, "Title D", float("nan")],
            "body": ["body a", "other body", "body only", "spaced", "x"],
        }
    )
    assert snippet_titles(frame, frame["text"].tolist()).tolist() == ["Title A", None, None, None, None]
    without_body = frame.drop(columns="body")
    titles = snippet_titles(without_body, without_body["text"].tolist()).tolist()
    assert titles == ["Title A", "Title B", None, None, None]


def test_article_keys_prefer_the_group_column():
    frame = pd.DataFrame({"title": [" T1 ", None, "T2"], "group": ["g7", "g7", None]})
    assert article_keys(frame).tolist() == ["g7", "g7", ""]
    assert article_keys(frame.drop(columns="group")).tolist() == ["T1", "", "T2"]
    assert article_keys(frame.drop(columns=["group", "title"])).tolist() == ["", "", ""]


def make_index() -> ActivationIndex:
    frame = pd.DataFrame(
        {
            "doc_id": [f"d{i}" for i in range(8)],
            "text": ["A one", "A two", "A three", "A four", "A five", "B one", "A one", "C one"],
            "title": ["A", "A", "A", "A", "A", "B", "A", "C"],
            "body": ["one", "two", "three", "four", "five", "one", "one", "one"],
        }
    )
    dense = np.zeros((8, 4), dtype=np.float32)
    dense[:5, 0] = [0.9, 0.8, 0.7, 0.6, 0.5]  # five passages of one article
    dense[5, 0] = 0.4
    dense[6, 0] = 0.95  # duplicate text of row 0: dropped by deduplication
    dense[:, 1] = 0.2  # fires everywhere
    dense[7, 2] = 0.3
    documents = CodedTexts.from_frame(frame, sp.csr_matrix(dense))
    queries_frame = pd.DataFrame(
        {"query_id": [f"q{i}" for i in range(6)], "text": [f"query {i}" for i in range(6)]}
    )
    query_dense = np.zeros((6, 4), dtype=np.float32)
    query_dense[:, 0] = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    query_dense[:4, 1] = 0.1
    queries = CodedTexts.from_frame(queries_frame, sp.csr_matrix(query_dense), id_column="query_id")
    return ActivationIndex(documents, queries)


def test_coded_texts_deduplicate_and_keep_ids_titles_rows():
    index = make_index()
    assert len(index.documents) == 7
    assert index.documents.rows.tolist() == [0, 1, 2, 3, 4, 5, 7]
    assert index.documents.ids.tolist()[:2] == ["d0", "d1"]
    assert index.documents.titles.tolist()[0] == "A"
    assert index.queries.titles.tolist() == [None] * 6
    snippets = index.documents.snippets(0, [0, 5])
    assert snippets == [
        {"id": "d0", "text": "A one", "title": "A", "activation": pytest.approx(0.9)},
        {"id": "d5", "text": "B one", "title": "B", "activation": pytest.approx(0.4)},
    ]
    assert "activation" not in index.documents.snippets(0, [6], with_activation=False)[0]


def test_coded_texts_follow_the_given_row_order():
    frame = pd.DataFrame({"doc_id": ["a", "b", "c"], "text": ["x", "y", "z"]})
    texts = CodedTexts.from_frame(frame, sp.csr_matrix(np.eye(3, dtype=np.float32)), rows=[2, 0])
    assert texts.texts.tolist() == ["z", "x"]
    assert texts.codes.toarray().tolist() == [[0, 0, 1], [1, 0, 0]]


def test_minimum_counts_are_checked_before_the_article_cap():
    index = make_index()
    assert index.document_counts.tolist() == [6, 7, 1, 0]
    # Feature 0 fires on 6 deduplicated documents, 5 of them from one article: it is selected...
    assert select_features(index, min_documents=5, min_queries=5) == [0]
    # ...but shows only two passages of that article.
    rows = index.top_documents(0, 5)
    assert index.documents.texts[rows].tolist() == ["A one", "A two", "B one"]


def test_selection_band_queries_minimum_and_n():
    index = make_index()
    assert select_features(index, min_documents=1, min_queries=0) == [0, 1, 2]
    assert select_features(index, min_documents=1, min_queries=5) == [0]
    assert select_features(index, min_documents=1, min_queries=0, firing_rate_band=(0.0, 6 / 7)) == [0, 2]
    assert select_features(index, min_documents=1, min_queries=0, firing_rate_band=(1.0, 1.0)) == [1]
    assert select_features(index, min_documents=1, min_queries=0, n=2) == [0, 1]
    no_queries = ActivationIndex(index.documents)
    assert select_features(no_queries, min_documents=1, min_queries=100) == [0, 1, 2]


def test_theme_entropy_matches_a_hand_computation():
    codes = sp.csr_matrix(
        np.array(
            [
                [1, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 1],
            ],
            dtype=np.float32,
        )
    )
    labels = ["x", "x", "y", None]
    entropy, p, names = theme_entropy(codes, labels)
    assert names == ["x", "y"]
    # Feature 0 fires on both "x" rows only: pure. Feature 1 fires on one of two "x" rows and the "y" row:
    # P(fires | x) = 0.5, P(fires | y) = 1, renormalised to (1/3, 2/3).
    assert entropy[0] == pytest.approx(0.0)
    expected = -(1 / 3 * math.log(1 / 3) + 2 / 3 * math.log(2 / 3)) / math.log(2)
    assert entropy[1] == pytest.approx(expected)
    assert p[1].tolist() == pytest.approx([1 / 3, 2 / 3])
    # Feature 2 never fires and feature 3 fires only on the row without a theme.
    assert math.isnan(entropy[2]) and math.isnan(entropy[3])


def test_theme_entropy_splits_multi_label_rows_and_needs_two_themes():
    codes = sp.csr_matrix(np.array([[1.0], [0.0], [0.0]], dtype=np.float32))
    entropy, p, names = theme_entropy(codes, [["x", "y"], "x", []])
    assert names == ["x", "y"]
    # The first row counts half for each theme: P(fires | x) = 0.5 / 1.5 and P(fires | y) = 0.5 / 0.5.
    assert p[0].tolist() == pytest.approx([0.25, 0.75])
    with pytest.raises(ValueError, match="more than one theme"):
        theme_entropy(codes, ["x", "x", "x"])


def test_selection_orders_by_ascending_theme_entropy():
    frame = pd.DataFrame({"doc_id": list("abcd"), "text": list("abcd")})
    dense = np.array([[1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 1, 1]], dtype=np.float32)
    index = ActivationIndex(CodedTexts.from_frame(frame, sp.csr_matrix(dense)))
    themes = ["x", "x", "y", "y"]
    assert select_features(index, min_documents=1, themes=themes) == [0, 1, 2]
    with pytest.raises(ValueError, match="themes"):
        select_features(index, min_documents=1, themes=["x"])


def test_tfidf_drops_stop_words_before_bigrams_and_digits():
    documents = [
        "quality of life after covid-19",
        "quality of life and COVID19 care",
        "the quality of life of patients",
        "covid-19 outcomes",
        "covid-19 wards",
    ]
    tfidf = TfidfTerms.fit(documents, ["quality of life question"])
    vocabulary = set(tfidf.vocabulary.tolist())
    assert "quality life" in vocabulary
    assert "covid" in vocabulary
    assert not any("19" in term or "covid19" in term for term in vocabulary)


def test_tfidf_terms_are_the_top_of_the_mean_row():
    documents = ["heart failure drugs"] * 3 + ["kidney stones pain"] * 3 + ["heart kidney"] * 3
    queries = ["heart failure question"] * 3
    tfidf = TfidfTerms.fit(documents, queries)
    terms = tfidf.terms([0, 1], [0], n_terms=2)
    assert set(terms) <= {"heart", "failure", "heart failure", "drugs", "failure drugs"}
    assert len(tfidf.terms([3], n_terms=5)) == 5  # n_terms terms even when fewer have weight
    assert tfidf.terms([], []) == []


def test_tfidf_on_a_corpus_too_small_for_min_df_is_disabled():
    tfidf = TfidfTerms.fit(["one text", "another text"])
    assert tfidf.terms([0, 1]) == []
