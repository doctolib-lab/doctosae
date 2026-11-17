"""Activation Prediction: evaluation pool, rows, pairs, metrics and the score report."""

from __future__ import annotations

import json
import math
import random

import numpy as np
import pytest
import scipy.sparse as sp

from doctosae.interpret.cards import FeatureCard
from doctosae.interpret.llm import FakeLLM
from doctosae.interpret.prompts import OOD_NOTE, SCORE_INSTRUCTION
from doctosae.interpret.scoring import (
    EvalPool,
    PoolSource,
    build_eval_rows,
    entry_ranks,
    exemplars,
    feature_aurocs,
    feature_metrics,
    pearson_r,
    positive_matrix,
    rank_band_stats,
    run_activation_prediction,
    sample_pairs,
    sample_pool_rows,
)


def test_pool_deduplicates_excludes_exemplars_and_caps_articles():
    first = PoolSource(
        name="train",
        codes=sp.csr_matrix(np.arange(6, dtype=np.float32).reshape(6, 1) + 1),
        texts=["t0", "t1", "t2", "t3", "t4", "t5"],
        titles=["A", "A", "A", "B", "C", ""],
        articles=["A", "A", "A", "B", "C", ""],
    )
    second = PoolSource(
        name="val",
        codes=sp.csr_matrix(np.full((3, 1), 9, dtype=np.float32)),
        texts=["t1", "t6", "t7"],  # "t1" repeats a train text: the train row is kept
        titles=["A", "", "D"],
        articles=["A", "", "D"],
    )
    pool = EvalPool.build([first, second], exemplar_texts={"t4"}, exemplar_titles={"B"})
    # t2 is the third passage of article A, t3 shares the title of a shown passage, t4 was shown.
    assert pool.texts.tolist() == ["t0", "t1", "t5", "t6", "t7"]
    assert pool.sources.tolist() == ["train", "train", "train", "val", "val"]
    assert pool.codes.toarray().ravel().tolist() == [1, 2, 6, 9, 9]
    uncapped = EvalPool.build([first], cap_articles=False)
    assert uncapped.texts.tolist() == ["t0", "t1", "t2", "t3", "t4", "t5"]


def test_article_cap_is_counted_before_the_exclusions():
    source = PoolSource(
        name="corpus", codes=sp.csr_matrix(np.ones((3, 1))), texts=["a", "b", "c"], articles=["X", "X", "X"]
    )
    # The first two passages of X are shown: the third is still beyond the cap, so nothing remains.
    assert len(EvalPool.build([source], exemplar_texts={"a", "b"})) == 0


def test_exemplars_collect_texts_and_titles_of_every_shown_snippet():
    card = FeatureCard(
        feature=0,
        n_documents=1,
        n_queries=1,
        firing_rate=0.1,
        top_documents=[{"text": "T body", "title": "T", "activation": 1.0}],
        top_queries=[{"text": "q", "title": None, "activation": 1.0}],
        negatives=[{"text": "N other", "title": "N"}],
        tfidf_terms=[],
    )
    assert exemplars([card]) == ({"T body", "q", "N other"}, {"T", "N"})


def test_entry_ranks_and_positive_band():
    codes = sp.csr_matrix(np.array([[0.1, 0.5, 0.0, 0.3], [0.0, 0.2, 0.4, 0.0]], dtype=np.float32))
    assert entry_ranks(codes).tolist() == [3, 1, 2, 2, 1]
    positives = positive_matrix(codes, [1, 3, 0], rank_band=(1, 2))
    assert positives.toarray().tolist() == [[True, True, False], [True, False, False]]
    positives = positive_matrix(codes, [1, 2], rank_band=(2, 2))
    assert positives.toarray().tolist() == [[False, False], [True, False]]


def random_codes(n_rows: int = 60, n_latents: int = 12, k: int = 4, seed: int = 0) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    dense = np.zeros((n_rows, n_latents), dtype=np.float32)
    for row in range(n_rows):
        dense[row, rng.choice(n_latents, k, replace=False)] = rng.random(k).astype(np.float32) + 0.01
    return sp.csr_matrix(dense)


def test_pool_rows_follow_the_paper_sampling():
    codes = random_codes()
    features = [0, 3, 7]
    positives = positive_matrix(codes, features)
    rng = np.random.default_rng(42)
    expected = []
    dense_positives = positives.toarray()
    for j in range(len(features)):
        strong = np.where(dense_positives[:, j])[0]
        expected.append(rng.choice(strong, min(5, len(strong)), replace=False))
    fires_any = (codes.toarray()[:, features] > 0).any(axis=1)
    none_fire = np.where(~fires_any)[0]
    expected.append(rng.choice(none_fire, min(5, len(none_fire)), replace=False))
    assert (
        sample_pool_rows(codes, features, positives).tolist() == np.unique(np.concatenate(expected)).tolist()
    )


def test_eval_rows_record_activations_positives_and_ranks():
    codes = random_codes()
    pool = EvalPool(
        codes=codes,
        texts=np.array([f"text {i}" for i in range(60)], dtype=object),
        sources=np.array(["corpus"] * 60, dtype=object),
    )
    features = [0, 3, 7, 11]
    kept, rows = build_eval_rows(pool, features, rank_band=(1, 2))
    dense = codes.toarray()
    assert set(kept) <= set(features)
    for row in rows:
        values = dense[row["row"]]
        order = np.argsort(-values)
        assert row["text"] == f"text {row['row']}"
        assert row["activations"] == {str(f): pytest.approx(float(values[f])) for f in kept if values[f] > 0}
        assert set(row["ranks"]) == set(row["activations"])
        for f in row["positive_for"]:
            assert row["ranks"][str(f)] <= 2 and f in kept
            assert int(np.flatnonzero(order == f)[0]) + 1 == row["ranks"][str(f)]
    with pytest.raises(ValueError, match="unique"):
        build_eval_rows(pool, [0, 0])


def test_pairs_share_one_generator_across_features():
    rows = [
        {"activations": {"1": 0.5}, "positive_for": [1]},
        {"activations": {"1": 0.2, "2": 0.3}, "positive_for": [2]},
        {"activations": {}, "positive_for": []},
        {"activations": {"2": 0.1}, "positive_for": [2]},
        {"activations": {}, "positive_for": []},
    ]
    pairs = sample_pairs(rows, [1, 2], n_positive=3, n_negative=3, seed=42)
    rng = random.Random(42)
    expected = [(1, i, True) for i in rng.sample([0], 1)]
    expected += [(1, i, False) for i in rng.sample([2, 3, 4], 3)]
    expected += [(2, i, True) for i in rng.sample([1, 3], 2)]
    expected += [(2, i, False) for i in rng.sample([0, 2, 4], 3)]
    assert pairs == expected
    # A row where the feature fires below the positive band is neither positive nor negative.
    assert (1, 1, False) not in pairs


def test_pearson_r_conventions():
    assert pearson_r([0.9, 0.1], [True, False]) == pytest.approx(1.0)
    assert pearson_r([0.5, 0.5, 0.5], [True, False, True]) == 0.0
    assert math.isnan(pearson_r([0.2, 0.8], [True, True]))


def predictions_for(scores: dict[int, list[tuple[bool, float]]]) -> list[dict]:
    return [
        {
            "feature": f,
            "ground_truth": truth,
            "score": score,
            "parse_error": False,
            "rank": 1 if truth else None,
        }
        for f, items in scores.items()
        for truth, score in items
    ]


def test_feature_metrics_and_aurocs_skip_parse_errors():
    predictions = predictions_for(
        {
            1: [(True, 0.9), (True, 0.6), (False, 0.1)],
            2: [(True, 0.4), (False, 0.4)],
            3: [(True, 0.2), (True, 0.7)],
        }
    )
    predictions.append({"feature": 1, "ground_truth": False, "score": 0.5, "parse_error": True, "rank": None})
    metrics = feature_metrics(predictions, [1, 2, 3, 4])
    assert metrics[1]["support"] == 2 and metrics[1]["n_pairs"] == 3
    assert metrics[1]["pearson"] == pytest.approx(np.corrcoef([0.9, 0.6, 0.1], [1, 1, -1])[0, 1])
    assert metrics[2]["pearson"] == 0.0
    assert math.isnan(metrics[3]["pearson"]) and math.isnan(metrics[4]["pearson"])
    assert feature_aurocs(predictions, [1, 2, 3]) == {1: 1.0, 2: 0.5}


def test_rank_band_stats():
    predictions = [
        {"ground_truth": True, "rank": 1, "score": 0.9, "parse_error": False},
        {"ground_truth": True, "rank": 7, "score": 0.5, "parse_error": False},
        {"ground_truth": True, "rank": 18, "score": 0.1, "parse_error": False},
        {"ground_truth": False, "rank": None, "score": 0.2, "parse_error": False},
        {"ground_truth": False, "rank": None, "score": 0.9, "parse_error": True},
    ]
    stats = rank_band_stats(predictions)
    assert stats["1-5"] == pytest.approx(0.9) and stats["6-10"] == pytest.approx(0.5)
    assert math.isnan(stats["11-15"]) and stats["16-20"] == pytest.approx(0.1)
    assert stats["inactive"] == pytest.approx(0.2)
    assert stats["spearman"] == pytest.approx(-1.0)


WORDS = [
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
]


def make_cards(features: list[int], labelled: bool = True) -> list[FeatureCard]:
    return [
        FeatureCard(
            feature=f,
            n_documents=5,
            n_queries=5,
            firing_rate=0.1,
            top_documents=[{"id": "x", "text": f"shown passage {f}", "title": None, "activation": 0.5}],
            top_queries=[],
            negatives=[],
            tfidf_terms=[WORDS[f]],
            topic=f"topic {f}" if labelled else None,
            description="Something.",
            keywords=["kw"],
        )
        for f in features
    ]


def make_pool(n_rows: int = 80) -> EvalPool:
    codes = random_codes(n_rows=n_rows, seed=3)
    texts = [
        f"text {i} " + " ".join(WORDS[j] for j in np.flatnonzero(codes[i].toarray())) for i in range(n_rows)
    ]
    return EvalPool(codes=codes, texts=np.array(texts, dtype=object), sources=np.array(["corpus"] * n_rows))


def test_activation_prediction_end_to_end_with_fake_llm():
    llm = FakeLLM()
    report = run_activation_prediction(
        make_cards([0, 2, 5, 9]), llm, make_pool(), dataset="toy", batch_size=4, show_progress=False
    )
    assert report.methods == ["tfidf", "llm", "snippets"]
    assert report.shared_features == report.features
    frame = report.to_frame()
    assert list(frame.index) == ["tfidf", "llm", "snippets"]
    assert set(frame.columns) == {
        "pearson_r",
        "auroc",
        "n_features",
        "n_calls",
        "n_parse_errors",
        "n_errors",
        "abstained",
    }
    # Term overlap separates positives from negatives perfectly for the TF IDF description.
    assert frame.loc["tfidf", "pearson_r"] == pytest.approx(1.0)
    assert frame.loc["tfidf", "auroc"] == pytest.approx(1.0)
    assert all(user.endswith(SCORE_INSTRUCTION) and OOD_NOTE not in user for _, user in llm.calls)
    for method, predictions in report.predictions.items():
        assert len(predictions) == frame.loc[method, "n_calls"]
        assert all(p["ground_truth"] == (p["rank"] is not None) for p in predictions)
    assert set(report.rank_frame().columns) == {"1-5", "6-10", "11-15", "16-20", "inactive", "spearman"}
    assert len(report.feature_frame()) == 3 * len(report.features)
    json.dumps(report.to_dict())


def test_out_of_domain_prompts_carry_the_note():
    llm = FakeLLM()
    run_activation_prediction(
        make_cards([0, 2]),
        llm,
        make_pool(),
        dataset="other",
        out_of_domain=True,
        methods=["tfidf"],
        show_progress=False,
    )
    assert llm.calls and all(OOD_NOTE in user for _, user in llm.calls)


def test_parse_errors_are_recorded_and_excluded():
    report = run_activation_prediction(
        make_cards([0, 2]),
        FakeLLM(lambda system, user: "no idea"),
        make_pool(),
        dataset="toy",
        methods=["tfidf"],
        show_progress=False,
    )
    predictions = report.predictions["tfidf"]
    assert all(p["parse_error"] and p["score"] == 0.5 and p["raw_reply"] == "no idea" for p in predictions)
    assert report.summary["tfidf"]["n_parse_errors"] == len(predictions)
    assert math.isnan(report.summary["tfidf"]["pearson_r"])


def test_clamped_scores_keep_the_raw_value_and_abstentions_are_counted():
    report = run_activation_prediction(
        make_cards([0]),
        FakeLLM(lambda system, user: '{"score": 1.5}'),
        make_pool(),
        dataset="toy",
        methods=["tfidf"],
        show_progress=False,
    )
    assert all(
        p["score"] == 1.0 and p["score_raw"] == 1.5 and p["fires"] for p in report.predictions["tfidf"]
    )
    report = run_activation_prediction(
        make_cards([0]),
        FakeLLM(lambda system, user: '{"score": 0}'),
        make_pool(),
        dataset="toy",
        methods=["tfidf"],
        show_progress=False,
    )
    assert report.summary["tfidf"]["abstained"] == 1.0


def test_llm_method_needs_labels_and_skips_unlabelled_features():
    with pytest.raises(ValueError, match="labelled"):
        run_activation_prediction(
            make_cards([0, 2], labelled=False),
            FakeLLM(),
            make_pool(),
            dataset="toy",
            methods=["llm"],
            show_progress=False,
        )
    cards = make_cards([0, 2, 5])
    cards[1].topic = "PARSE_ERROR"
    report = run_activation_prediction(cards, FakeLLM(), make_pool(), dataset="toy", show_progress=False)
    assert 2 not in report.scored["llm"] and 2 in report.scored["tfidf"]
    assert 2 not in report.shared_features
    with pytest.raises(ValueError, match="unknown methods"):
        run_activation_prediction(cards, FakeLLM(), make_pool(), dataset="toy", methods=["words"])


def test_batches_follow_batch_size():
    sizes = []

    class Recorder(FakeLLM):
        def generate(self, system, users):
            sizes.append(len(users))
            return super().generate(system, users)

    report = run_activation_prediction(
        make_cards([0, 2, 5]),
        Recorder(),
        make_pool(),
        dataset="toy",
        methods=["tfidf"],
        batch_size=8,
        show_progress=False,
    )
    n_calls = len(report.predictions["tfidf"])
    assert sizes == [8] * (n_calls // 8) + ([n_calls % 8] if n_calls % 8 else [])


def test_a_failed_request_costs_only_its_own_prediction():
    """One prompt that fails even alone becomes an error record; the chunk and the other methods go on."""
    failing = None

    def reply(system, user):
        nonlocal failing
        failing = failing or user
        if user == failing:
            raise TimeoutError("read timeout")
        return FakeLLM().reply(system, user)

    report = run_activation_prediction(
        make_cards([0, 2, 5]), FakeLLM(reply), make_pool(), dataset="toy", batch_size=4, show_progress=False
    )
    assert report.methods == ["tfidf", "llm", "snippets"] and not report.failed
    errors = [p for p in report.predictions["tfidf"] if p.get("error")]
    assert (
        len(errors) == 1 and errors[0]["error"] == "TimeoutError: read timeout" and errors[0]["score"] == 0.5
    )
    assert report.summary["tfidf"]["n_errors"] == 1 and report.summary["llm"]["n_errors"] == 0
    reference = run_activation_prediction(
        make_cards([0, 2, 5]), FakeLLM(), make_pool(), dataset="toy", batch_size=4, show_progress=False
    )
    others = [p for p in report.predictions["tfidf"] if not p.get("error")]
    assert others == [p for p in reference.predictions["tfidf"] if p["text"] != errors[0]["text"]
                      or p["feature"] != errors[0]["feature"]]  # fmt: skip
    json.dumps(report.to_dict())


def test_an_unavailable_llm_stops_the_method_and_keeps_the_others():
    calls = []

    class Outage(FakeLLM):
        def generate(self, system, users):
            calls.append(len(users))
            if "terms" not in system:  # the snippets and llm methods meet a server outage
                raise ConnectionError("server down")
            return super().generate(system, users)

    report = run_activation_prediction(
        make_cards([0, 2, 5]), Outage(), make_pool(), dataset="toy", batch_size=4, show_progress=False
    )
    assert report.methods == ["tfidf"] and set(report.failed) == {"llm", "snippets"}
    assert "3 chunks in a row" in report.failed["llm"] and set(report.summary) == {"tfidf"}
    assert report.to_dict()["failed"] == report.failed


def test_tfidf_needs_terms_and_a_feature_without_evidence_reads_no_docs():
    cards = make_cards([0, 2])
    for card in cards:
        card.tfidf_terms = []
    with pytest.raises(ValueError, match="TF IDF"):
        run_activation_prediction(cards, FakeLLM(), make_pool(), dataset="toy", methods=["tfidf"])
    cards[0].tfidf_terms = ["alpha"]
    llm = FakeLLM()
    run_activation_prediction(cards, llm, make_pool(), dataset="toy", methods=["tfidf"], show_progress=False)
    assert any("Neuron 2:\n  (no docs)\n" in user for _, user in llm.calls)
