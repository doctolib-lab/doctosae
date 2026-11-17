"""scripts/build_cases.py on small synthetic inputs: the three input modes, ranking, the draw, the output."""

from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import pytest

from cases import CASE_KEYS, EXTRA_KEYS, judge_input, validate_cases

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_cases.py"
_spec = importlib.util.spec_from_file_location("build_cases", _SCRIPT)
build_cases = importlib.util.module_from_spec(_spec)
sys.modules["build_cases"] = build_cases  # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(build_cases)


def write_json(path: Path, data) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ── Markdown review files ────────────────────────────────────────────


def review_file(n: int = 8) -> str:
    blocks = [
        f"### {rank}. Feature {100 + rank}\n\n"
        f"**Description.** topic {rank}: Synthetic description {rank}. (keywords: alpha {rank}, beta)\n\n"
        f"**{'Positive' if rank % 2 else 'Negative'}**\n\n> Synthetic passage {rank}."
        for rank in range(1, n + 1)
    ]
    return "\n\n---\n\n".join([f"# Top {n} SAE features by r\n\nSynthetic review file.", *blocks]) + "\n"


def study_draw(features: list[dict], n_cases: int, seed: int) -> list[dict]:
    """The sampling code of the original study app, kept verbatim as the reference."""
    rng = random.Random(seed)
    half = n_cases // 2
    pos = rng.sample([f for f in features if f["ground_truth"] == 1], half)
    neg = rng.sample([f for f in features if f["ground_truth"] == 0], n_cases - half)
    return sorted(pos + neg, key=lambda f: f["rank"])


def test_review_file_parse_and_draw_match_the_study_code():
    features = build_cases.parse_markdown(review_file())
    assert [f["rank"] for f in features] == list(range(1, 9))
    first = features[0]
    assert (first["feature_id"], first["topic"]) == (101, "topic 1")
    assert first["description"] == "Synthetic description 1."
    assert first["keywords"] == ["alpha 1", "beta"]
    assert first["judge_input"] == "topic 1: Synthetic description 1. (keywords: alpha 1, beta)"
    assert (first["passage"], first["ground_truth"]) == ("Synthetic passage 1.", 1)

    cases = build_cases.sample_cases(features, n_cases=4, seed=42)
    assert [c["rank"] for c in cases] == [f["rank"] for f in study_draw(features, 4, 42)]
    assert [c["case_id"] for c in cases] == ["C01", "C02", "C03", "C04"]
    assert all(list(c) == list(CASE_KEYS) for c in cases)  # no extra key: displayed as in the study
    assert sum(c["ground_truth"] for c in cases) == 2
    validate_cases(cases)


def test_split_description_without_keywords_or_topic():
    assert build_cases.split_description("asthma: Airways. (keywords: )") == {
        "topic": "asthma", "description": "Airways.", "keywords": []}
    assert build_cases.split_description("no colon here") == {
        "topic": "", "description": "no colon here", "keywords": []}


def test_judge_input_is_the_paper_string():
    assert judge_input("asthma", "Airways.", ["wheeze", "ICS"]) == "asthma: Airways. (keywords: wheeze, ICS)"
    assert judge_input("asthma", "Airways.", []) == "asthma: Airways. (keywords: )"


# ── Paper code outputs ───────────────────────────────────────────────


def paper_outputs(tmp_path: Path, **overrides) -> tuple[Path, Path]:
    """labels.json and pred_binary_llm.json in the paper format, with one feature per filtering rule."""
    neurons, predictions = [], []
    metrics: dict = {"micro avg": {"support": 1, "n_pairs": 2, "pearson": 0.9}}

    def add(feature, topic, scores, labels, description=None, parse_errors=(), texts=None):
        if description is None:
            description = f"Synthetic feature {feature}."
        neurons.append({"neuron": feature, "dominant_theme": "synthetic", "topic": topic,
                        "description": description, "keywords": [f"kw{feature}", "shared"]})
        for i, (score, positive) in enumerate(zip(scores, labels, strict=True)):
            text = (texts or {}).get(i, f"Synthetic passage {i} judged for feature {feature}.")
            failed = i in parse_errors
            predictions.append({"neuron": feature, "row": 100 * feature + i, "source": "synthetic",
                                "text": text[:300], "ground_truth": positive, "activation": None,
                                "rank": None, "parse_error": failed, "score": 0.5 if failed else score,
                                "fires": score > 0.5})

    t, f = True, False
    add(1, "cardiology", [0.9, 0.8, 0.1, 0.0], [t, t, f, f])
    add(2, "LABEL_ERROR", [0.9, 0.1], [t, f])  # failed label: dropped although it scores well
    add(3, "dermatology", [0.9, 0.2], [t, t])  # positives only: r undefined
    metrics["3"] = {"support": 2, "n_pairs": 2, "pearson": math.nan}
    add(4, "oncology", [0.9, 0.8, 0.1, 0.0, 0.3], [t, t, f, f, f], parse_errors={4})  # same r as 1, cleaned
    add(5, "nephrology", [0.6, 0.2, 0.4, 0.1], [t, t, f, f])
    metrics["5"] = {"support": 2, "n_pairs": 4, "pearson": 0.25}  # stored r wins over a recomputation
    add(6, "endocrinology", [0.3, 0.3], [t, f], texts={0: "y" * 120, 1: "x" * 400})  # constant: r = 0
    add(7, "neurology", [0.7, 0.1], [t, f])
    metrics["7"] = {"support": 1, "n_pairs": 2, "pearson": math.nan}  # stored r not finite: dropped
    add(8, "hepatology", [0.7, 0.1], [t, f], description="  ")  # empty description: dropped
    for feature in range(10, 22):  # ordinary features, r between 0.5 and 0.6, decreasing with the feature id
        gap = (22 - feature) / 30
        add(feature, f"topic {feature}", [0.5 + gap, 0.45, 0.4, 0.3], [t, f, t, f])

    labels = {"neuron_profiles_file": "profiles.json", "n_labelled": len(neurons), "neurons": neurons}
    preds = {"description_type": "llm", "shuffle_descriptions": False, "max_example_chars": 1000,
             "predictions": predictions, "metrics": metrics, **overrides}
    return write_json(tmp_path / "labels.json", labels), write_json(tmp_path / "pred_binary_llm.json", preds)


def test_paper_outputs_are_filtered_and_ranked(tmp_path):
    inputs = build_cases.load_paper(*paper_outputs(tmp_path))
    ranked = build_cases.rank_features(inputs, top=60)
    ids = [f["feature"] for f in ranked]
    assert ids[:2] == [1, 4]  # equal r, ties broken by feature id
    assert ranked[0]["r"] == ranked[1]["r"] > 0.99
    assert not {2, 3, 7, 8} & set(ids)
    by_id = {f["feature"]: f for f in ranked}
    assert by_id[5]["r"] == 0.25
    assert by_id[6]["r"] == 0.0 and ids[-1] == 6
    assert all(not p["parse_error"] for p in by_id[4]["pairs"]) and len(by_id[4]["pairs"]) == 4
    rs = [f["r"] for f in ranked]
    assert rs == sorted(rs, reverse=True)
    assert [f["feature"] for f in build_cases.rank_features(inputs, top=3)] == ids[:3]


@pytest.mark.parametrize(("overrides", "message"), [
    ({"shuffle_descriptions": True}, "shuffled control"),
    ({"description_type": "tfidf"}, "'tfidf' descriptions"),
])
def test_scores_of_the_wrong_kind_are_refused(tmp_path, overrides, message):
    with pytest.raises(build_cases.BuildError, match=message):
        build_cases.load_paper(*paper_outputs(tmp_path, **overrides))


def test_pick_passages_keeps_the_judged_pair(tmp_path):
    inputs = build_cases.load_paper(*paper_outputs(tmp_path))
    ranked = build_cases.rank_features(inputs, top=60)
    picks = build_cases.pick_passages(ranked, seed=42, max_chars=300, text_cap=300)
    assert [p["rank"] for p in picks] == list(range(1, len(ranked) + 1))
    for pick, feature in zip(picks, ranked, strict=True):
        pair = next(p for p in feature["pairs"] if p["text"][:300].split() == pick["passage"].split())
        assert pick["ground_truth"] == int(pair["ground_truth"])
        assert pick["judge_probability"] == pair["probability"]
        assert pick["feature_r"] == feature["r"] and pick["source"] == "synthetic"
        expected = judge_input(feature["topic"], feature["description"], feature["keywords"])
        assert pick["judge_input"] == expected
    assert picks == build_cases.pick_passages(ranked, seed=42, max_chars=300, text_cap=300)  # deterministic


def pick_with_label(feature: dict, positive: bool, **kwargs) -> dict:
    """The pick of the first seed that draws the wanted label (the label itself is drawn at random)."""
    for seed in range(50):
        pick = build_cases.pick_passages([feature], seed=seed, **kwargs)[0]
        if pick["ground_truth"] == int(positive):
            return pick
    raise AssertionError("no seed drew that label")


def test_truncation_flags_and_whitespace():
    def pair(text: str, positive: bool) -> dict:
        return {"feature": 1, "row": None, "text": text, "ground_truth": positive, "probability": 0.5,
                "parse_error": False, "source": None}

    feature = {"feature": 1, "r": 0.5, "topic": "t", "description": "d", "keywords": [],
               "pairs": [pair("a  b\n\nc " * 50, True), pair("short text", False)]}
    cut = pick_with_label(feature, True, max_chars=40, text_cap=None)
    assert cut["truncated"] and len(cut["passage"]) <= 40 and "\n" not in cut["passage"]
    assert cut["passage"].startswith("a b c a b c")
    kept = pick_with_label(feature, False, max_chars=40, text_cap=None)
    assert not kept["truncated"] and kept["passage"] == "short text"
    # a scorer that stored only the first 300 characters may have cut a longer original
    feature["pairs"][1] = pair("z" * 300, False)
    capped = pick_with_label(feature, False, max_chars=500, text_cap=300)
    assert capped["truncated"] and capped["passage"] == "z" * 300
    feature["pairs"][1] = pair("z" * 299, False)
    assert not pick_with_label(feature, False, max_chars=500, text_cap=300)["truncated"]


def test_cli_paper_mode_end_to_end(tmp_path, capsys):
    labels, predictions = paper_outputs(tmp_path)
    out, review = tmp_path / "out" / "cases.json", tmp_path / "top.md"
    code = build_cases.main(["--labels", str(labels), "--predictions", str(predictions), "--top", "12",
                             "--n-cases", "6", "--out", str(out), "--markdown", str(review),
                             "--dataset", "CURE"])
    assert code == 0
    cases = json.loads(out.read_text(encoding="utf-8"))
    assert out.read_text(encoding="utf-8").endswith("\n")
    validate_cases(cases)
    assert len(cases) == 6 and sum(c["ground_truth"] for c in cases) == 3
    assert [c["case_id"] for c in cases] == [f"C{i:02d}" for i in range(1, 7)]
    assert [c["rank"] for c in cases] == sorted(c["rank"] for c in cases)
    for case in cases:
        assert list(case)[:len(CASE_KEYS)] == list(CASE_KEYS)
        assert set(case) - set(CASE_KEYS) <= set(EXTRA_KEYS)
        assert {"truncated", "feature_r", "judge_probability", "source"} <= set(case)
        assert case["judge_input"] == judge_input(case["topic"], case["description"], case["keywords"])
    assert "wrote 6 cases (3 positive, 3 negative)" in capsys.readouterr().out

    text = review.read_text(encoding="utf-8")
    assert text.startswith("# Top 12 SAE features by r (Interpreter LLM description, CURE)\n")
    back = build_cases.parse_markdown(text)
    assert len(back) == 12
    again = tmp_path / "again.json"
    assert build_cases.main(["--from-markdown", str(review), "--n-cases", "6", "--out", str(again)]) == 0
    reread = json.loads(again.read_text(encoding="utf-8"))
    for key in CASE_KEYS:
        assert [c[key] for c in reread] == [c[key] for c in cases]


# ── doctosae package outputs ───────────────────────────────────────────


def judged(feature: int, row: int, positive: bool, score: float, parse_error: bool = False) -> dict:
    """One prediction record as doctosae.interpret.scoring writes it."""
    return {"feature": feature, "row": row, "source": "corpus", "text": f"Synthetic text {row} of {feature}.",
            "ground_truth": positive, "activation": 0.2 if positive else None,
            "rank": 4 if positive else None, "parse_error": parse_error,
            "score": 0.5 if parse_error else score, "fires": score > 0.5}


def doctosae_outputs(tmp_path: Path, methods: tuple[str, ...] = ("tfidf", "llm")) -> tuple[Path, Path]:
    """cards.json and scores.json in the layout `doctosae interpret --score` writes them."""
    cards = [{"feature": 9, "n_documents": 6, "n_queries": 5, "firing_rate": 0.01, "top_documents": [],
              "top_queries": [], "negatives": [], "tfidf_terms": ["term"], "topic": None, "description": None,
              "keywords": []}]  # never labelled
    llm = [judged(9, 0, True, 0.9), judged(9, 1, False, 0.1), judged(1, 5, False, 0.0, parse_error=True)]
    tfidf = []
    for feature in range(1, 9):
        cards.append({**cards[0], "feature": feature, "topic": f"topic {feature}",
                      "description": f"Synthetic feature {feature}.", "keywords": ["kw"]})
        for row, (positive, score) in enumerate([(True, 0.6), (True, 0.3), (False, 0.4), (False, 0.2)]):
            llm.append(judged(feature, row, positive, score - feature / 100))
            tfidf.append(judged(feature, row, not positive, 0.9))  # another method: must be ignored
    per_feature = {"llm": {"8": {"support": 2, "n_pairs": 4, "pearson": 0.999, "auroc": 1.0},
                           "9": {"support": 1, "n_pairs": 2, "pearson": 1.0, "auroc": 1.0}},
                   "tfidf": {"1": {"support": 2, "n_pairs": 4, "pearson": 1.0, "auroc": 1.0}}}
    report = {"dataset": "synthetic", "out_of_domain": False, "llm": "fake", "methods": list(methods),
              "features": list(range(1, 10)), "shared_features": list(range(1, 9)), "n_rows": 40,
              "summary": {}, "rank_bands": {},
              "per_feature": {m: per_feature[m] for m in methods},
              "predictions": {m: {"llm": llm, "tfidf": tfidf}[m] for m in methods}}
    payload = {"format": "doctosae.feature_cards", "version": 1, "metadata": {}, "cards": cards}
    return write_json(tmp_path / "cards.json", payload), write_json(tmp_path / "scores.json", report)


def test_doctosae_outputs(tmp_path):
    cards, scores = doctosae_outputs(tmp_path)
    inputs = build_cases.load_doctosae(cards, scores)
    assert inputs.dataset == "synthetic"
    assert len(inputs.pairs) == 3 + 8 * 4  # only the llm predictions
    ranked = build_cases.rank_features(inputs, top=60)
    ids = [f["feature"] for f in ranked]
    assert ids[0] == 8 and 9 not in ids and len(ids) == 8  # stored r first, unlabelled feature dropped
    assert all(p["probability"] < 0.9 for f in ranked for p in f["pairs"])  # no tfidf prediction leaked in
    out, review = tmp_path / "cases.json", tmp_path / "top.md"
    assert build_cases.main(["--cards", str(cards), "--scores", str(scores), "--n-cases", "4",
                             "--out", str(out), "--markdown", str(review)]) == 0
    validate_cases(json.loads(out.read_text(encoding="utf-8")))
    assert review.read_text(encoding="utf-8").startswith("# Top 8 SAE features by r (Interpreter LLM "
                                                         "description, synthetic)")


def test_failed_requests_of_doctosae_are_dropped_like_parse_errors(tmp_path):
    cards, scores = doctosae_outputs(tmp_path)
    report = json.loads(scores.read_text(encoding="utf-8"))
    failed = dict(judged(3, 9, True, 0.5), error="TimeoutError: read timeout")  # score 0.5, parse_error False
    report["predictions"]["llm"].append(failed)
    scores.write_text(json.dumps(report), encoding="utf-8")
    inputs = build_cases.load_doctosae(cards, scores)
    assert inputs.pairs[-1]["parse_error"] and math.isnan(inputs.pairs[-1]["probability"])
    ranked = build_cases.rank_features(inputs, top=60)
    assert all(p["row"] != 9 for f in ranked for p in f["pairs"])


def test_doctosae_scores_without_the_llm_method_are_refused(tmp_path):
    with pytest.raises(build_cases.BuildError, match="no scores of Interpreter LLM descriptions"):
        build_cases.load_doctosae(*doctosae_outputs(tmp_path, methods=("tfidf",)))


# ── The draw and the command line ────────────────────────────────────


def drawn_features(n_pos: int, n_neg: int) -> list[dict]:
    labels = [1] * n_pos + [0] * n_neg
    return [{"rank": rank, "feature_id": rank, "judge_input": f"t: d{rank} (keywords: )", "topic": "t",
             "description": f"d{rank}", "keywords": [], "passage": f"p{rank}", "ground_truth": label}
            for rank, label in enumerate(labels, start=1)]


def test_case_id_width_follows_the_number_of_cases():
    assert [build_cases.case_id_width(n) for n in (1, 9, 99, 100, 1000)] == [2, 2, 2, 3, 4]
    cases = build_cases.sample_cases(drawn_features(60, 60), n_cases=100, seed=42)
    assert cases[0]["case_id"] == "C001" and cases[-1]["case_id"] == "C100"
    assert len({c["case_id"] for c in cases}) == 100


def test_too_few_positives_is_a_clear_error():
    with pytest.raises(build_cases.BuildError, match="need 3 positive and 3 negative passages"):
        build_cases.sample_cases(drawn_features(2, 10), n_cases=6, seed=42)


def test_too_few_described_features_ask_for_more_features(tmp_path, capsys):
    cards, scores = doctosae_outputs(tmp_path)  # 8 usable features, below the default --top 60
    with pytest.raises(SystemExit) as exit_info:
        build_cases.main(["--cards", str(cards), "--scores", str(scores), "--out", str(tmp_path / "c.json")])
    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert "only 8 features" in err and "doctosae interpret" in err and "raise --top" not in err


@pytest.mark.parametrize("argv", [
    [],
    ["--cards", "c.json"],
    ["--from-markdown", "a.md", "--labels", "l.json", "--predictions", "p.json"],
])
def test_cli_needs_exactly_one_complete_input(argv):
    with pytest.raises(SystemExit) as exit_info:
        build_cases.main(argv)
    assert exit_info.value.code == 2


def test_cli_reports_bad_inputs_without_a_traceback(tmp_path, capsys):
    empty = tmp_path / "empty.md"
    empty.write_text("# nothing here\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        build_cases.main(["--from-markdown", str(empty), "--out", str(tmp_path / "cases.json")])
    assert exit_info.value.code == 1
    assert "no feature block found" in capsys.readouterr().err
    assert not (tmp_path / "cases.json").exists()
