"""Equivalence with the paper code: prompts, parsing, evidence, TF IDF terms, eval rows, pairs and metrics.

The paper scripts in ``paper/src/interpretability`` are imported read only (no bytecode written, no ``.env``
file loaded) and run on the same synthetic data as the package. The module is skipped when the paper code is
not next to the package (an sdist) or cannot be imported (its scripts need Python 3.12 and pydantic).
"""

from __future__ import annotations

import importlib
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from medsae.data.dataset import EmbeddedDataset, RetrievalDataset
from medsae.interpret import FakeLLM, Interpreter
from medsae.interpret.activations import encode_sparse
from medsae.interpret.labeling import parse_label
from medsae.interpret.prompts import (
    INTERPRETER_SYSTEM_PROMPT,
    METHOD_DEFINITIONS,
    OOD_NOTE,
    SCORE_INSTRUCTION,
    SCORE_PROMPT,
    USER_PROMPT,
    cap_text,
    feature_description,
    interpreter_system_prompt,
    judge_system_prompt,
    judge_user_prompt,
    label_instruction,
    render_evidence,
)
from medsae.interpret.scoring import feature_aurocs, feature_metrics, parse_score
from medsae.sae import MedSAE

PAPER_ROOT = Path(__file__).resolve().parents[2] / "paper"


@pytest.fixture(scope="module")
def paper():
    if not (PAPER_ROOT / "src" / "interpretability" / "select_neurons.py").is_file():
        pytest.skip("the paper code is not next to the package")
    pytest.importorskip("pydantic")
    # The paper code is the top level package "src": set aside any other "src" for the time of this module.
    saved_modules = {name: sys.modules.pop(name) for name in list(sys.modules) if name.split(".")[0] == "src"}

    def restore_modules() -> None:
        for name in [name for name in sys.modules if name.split(".")[0] == "src"]:
            del sys.modules[name]
        sys.modules.update(saved_modules)

    saved_path, saved_flag = list(sys.path), sys.dont_write_bytecode
    had_dotenv, saved_dotenv = "dotenv" in sys.modules, sys.modules.get("dotenv")
    sys.dont_write_bytecode = True
    sys.modules["dotenv"] = None  # label_neurons.py would otherwise load a .env file at import
    sys.path.insert(0, str(PAPER_ROOT))
    try:
        modules = SimpleNamespace(
            select=importlib.import_module("src.interpretability.select_neurons"),
            label=importlib.import_module("src.interpretability.label_neurons"),
            build=importlib.import_module("src.interpretability.build_interpret_eval"),
            predict=importlib.import_module("src.interpretability.predict_neurons_binary"),
        )
        try:
            modules.auroc = importlib.import_module("src.interpretability.table_auroc")
        except ImportError:  # plotting dependencies of the table scripts
            modules.auroc = None
    except (ImportError, SyntaxError) as error:
        restore_modules()
        pytest.skip(f"the paper code cannot be imported here: {error}")
    finally:
        sys.path[:] = saved_path
        sys.dont_write_bytecode = saved_flag
        if had_dotenv:
            sys.modules["dotenv"] = saved_dotenv
        else:
            sys.modules.pop("dotenv", None)
    yield modules
    restore_modules()


def same(a, b) -> bool:
    """Equality that treats NaN as equal to NaN, element by element for tuples."""
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


# Prompts and renderers.


def random_snippets(rng: random.Random, n: int, titled: bool, activation: bool) -> list[dict]:
    snippets = []
    for _ in range(n):
        body = " ".join(
            rng.choice(["bone", "graft", "cells", "renal", "x" * 30, "é"]) for _ in range(rng.randint(1, 60))
        )
        title = rng.choice([None, "Bone tissue", "Kidney"]) if titled else None
        text = f"{title} {body}" if title and rng.random() < 0.8 else body
        snippet = {"id": rng.randint(0, 99), "text": text, "title": title}
        if activation:
            snippet["activation"] = rng.random()
        snippets.append(snippet)
    return snippets


def test_prompt_constants_are_the_paper_constants(paper):
    assert INTERPRETER_SYSTEM_PROMPT == paper.label.SYSTEM_PROMPT
    assert interpreter_system_prompt(10) == paper.label.build_system_prompt(10)
    assert interpreter_system_prompt(7) == paper.label.build_system_prompt(7)
    assert METHOD_DEFINITIONS == paper.predict.CLASS_DEFN
    assert SCORE_PROMPT == paper.predict.SCORE_PROMPT
    assert USER_PROMPT == paper.predict.USER_PROMPT
    assert OOD_NOTE == paper.predict.OOD_NOTE
    assert SCORE_INSTRUCTION == paper.predict.SCORE_INSTRUCTION
    for method in ("tfidf", "llm", "snippets"):
        assert judge_system_prompt(method) == paper.predict.build_system_prompt(method)
    labeler = paper.label.LocalLabeler.__new__(paper.label.LocalLabeler)
    for n_keywords in (10, 3):
        labeler.n_keywords = n_keywords
        assert label_instruction(n_keywords) == labeler._with_json_instruction("")


def test_renderings_match_on_random_evidence(paper):
    rng = random.Random(0)
    for trial in range(200):
        neuron = {
            "neuron": trial,
            "corpus_snippets": random_snippets(rng, rng.randint(0, 5), titled=True, activation=True),
            "query_snippets": random_snippets(rng, rng.randint(0, 5), titled=False, activation=True),
            "negative_snippets": random_snippets(rng, rng.randint(0, 2), titled=True, activation=False),
            "top_terms": ", ".join(rng.sample(["bone", "cells", "graft", "renal", "kidney stones"], 3)),
        }
        max_chars = rng.choice([None, 0, 5, 40, 1000])
        max_documents, max_queries = rng.choice([None, 1, 3]), rng.choice([None, 2])
        show_negatives = rng.random() < 0.7
        expected = paper.label.build_prompt(
            neuron,
            max_chars=max_chars,
            max_corpus=max_documents,
            max_queries=max_queries,
            show_negatives=show_negatives,
        )
        assert (
            render_evidence(
                neuron["corpus_snippets"],
                neuron["query_snippets"],
                neuron["negative_snippets"],
                max_chars=max_chars,
                max_documents=max_documents,
                max_queries=max_queries,
                show_negatives=show_negatives,
            )
            == expected
        )
        card = SimpleNamespace(
            feature=trial,
            top_documents=neuron["corpus_snippets"],
            top_queries=neuron["query_snippets"],
            negatives=neuron["negative_snippets"],
            tfidf_terms=neuron["top_terms"].split(", "),
            has_label=True,
            interpretation="bone: Reacts to bone. (keywords: bone, graft)",
        )
        label_row = {"topic": "bone", "description": "Reacts to bone.", "keywords": ["bone", "graft"]}
        for method in ("tfidf", "snippets", "llm"):
            assert feature_description(card, method, max_chars) == paper.predict.build_class_description(
                neuron, method, label_row, max_chars
            )
        text = " ".join(rng.choice(["kidney", "stone", "a" * 50]) for _ in range(rng.randint(1, 80)))
        description = feature_description(card, "tfidf", max_chars)
        for ood in (False, True):
            assert judge_user_prompt(
                trial, description, cap_text(text, max_chars), ood
            ) == paper.predict.build_user(True, trial, description, paper.predict._cap(text, max_chars), ood)


# Parsing.


def label_replies() -> list[str]:
    objects = [
        {"topic": "bone", "description": "Bone grafts and healing.", "keywords": ["bone", "Bone", "graft"]},
        {"topic": ["bone", None, 2], "description": ["a", 1], "keywords": "single"},
        {"topic": None, "keywords": [None, 1.5, {"k": "é"}, [1], "x", " X "]},
        {"topic": 5, "description": "number topic"},
        {"description": "no topic", "keywords": []},
        {"topic": "t", "keywords": {"a": 1}},
        {"topic": "t", "description": True},
        {"topic": 'braces {inside} "quoted" \\ text', "keywords": [f"k{i}" for i in range(14)]},
    ]
    replies = []
    for obj in objects:
        payload = json.dumps(obj, ensure_ascii=False)
        replies += [
            payload,
            f"Steps: shared theme {{maybe}}.\n{payload}",
            f"```json\n{payload}\n```",
            f'{payload} and then {{"topic": "later"}}',
            f'{{"topic": "earlier"}} then {payload}',
            f"{payload} trailing words {{",
        ]
        replies += [payload[:cut] for cut in range(len(payload))]
        replies += ["Reasoning first.\n" + payload[:cut] for cut in range(0, len(payload), 3)]
    rng = random.Random(1)
    alphabet = ["{", "}", "[", "]", '"', ",", ":", "\\", "a", " ", "topic", '"topic"', "1", "null"]
    replies += ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 30))) for _ in range(2000)]
    return replies


def test_label_parsing_matches_on_many_replies(paper):
    for reply in label_replies():
        expected = paper.label._parse_json_label(reply)
        label = parse_label(reply)
        assert (label.topic, label.description, label.keywords) == (
            expected.topic,
            expected.description,
            expected.keywords,
        ), reply


def score_replies() -> list[str]:
    replies = []
    for value in [
        "0.3",
        "0",
        "1",
        "1.4",
        "-0.2",
        '"0.7"',
        '"high"',
        "null",
        "true",
        "[0.4]",
        "1e400",
        "NaN",
        "Infinity",
        "-Infinity",
        "9" * 400,
        '{"x": 1}',
    ]:
        payload = f'{{"score": {value}}}'
        replies += [
            payload,
            f"Sure.\n```json\n{payload}\n```",
            f'{payload} then {{"note": 1}}',
            f'{{"note": 1}} then {payload}',
            f'{{"outer": {payload}}}',
        ]
        replies += [payload[:cut] for cut in range(len(payload))]
    replies += ["score: 0.4", "", "{}", '{"Score": 0.4}', '{"score": 0.5, "score": 0.9}']
    rng = random.Random(2)
    alphabet = ["{", "}", '"score"', ":", "0.5", " ", '"', ",", "1", "x"]
    replies += ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 20))) for _ in range(2000)]
    return replies


def test_score_parsing_matches_on_many_replies(paper):
    for reply in score_replies():
        verdict, raw = paper.predict._parse_json(reply)
        expected = (None if verdict is None else verdict.score, raw)
        assert same(parse_score(reply), expected), reply


# Evidence, eval rows, pairs, prompts and metrics, end to end through the Interpreter.

D_IN, N_LATENTS, K = 16, 40, 6
N_SCORED = 8  # features profiled for the scoring comparisons (`--max-neurons` of the paper script)
WORDS = {
    "cardiology": ["heart", "artery", "cardiac", "valve", "rhythm", "pressure", "infarction"],
    "nephrology": ["kidney", "stones", "renal", "dialysis", "urine", "nephron", "bladder"],
    "dermatology": ["skin", "rash", "eczema", "psoriasis", "acne", "lesion", "itching"],
    "neurology": ["brain", "seizure", "stroke", "nerve", "memory", "migraine", "neuron"],
}


def make_data(name: str, seed: int, n_articles: int) -> EmbeddedDataset:
    rng = np.random.default_rng(seed)
    themes = list(WORDS)
    centroids = np.random.default_rng(100).normal(size=(len(themes), D_IN))
    rows, vectors = [], []
    for article in range(n_articles):
        theme = themes[article % len(themes)]
        title = f"Report {name} {article if article % 5 else article - 1}"  # a few titles name two articles
        article_vector = centroids[article % len(themes)] + 0.8 * rng.normal(size=D_IN)
        for chunk in range(int(rng.integers(1, 7))):  # passages of one article fire together: the cap matters
            body = " ".join(rng.choice(WORDS[theme], 5)) + f" {name} note {article}.{chunk}"
            rows.append(
                {
                    "doc_id": f"{name}{article}.{chunk}",
                    "text": f"{title} {body}",
                    "title": title,
                    "body": body,
                    "source_folder": theme,
                }
            )
            vectors.append(article_vector + 0.15 * rng.normal(size=D_IN))
    for position in (0, 5):  # exact duplicate texts
        rows.append(dict(rows[position], doc_id=f"{name}dup{position}"))
        vectors.append(vectors[position])
    corpus = pd.DataFrame(rows)
    n_queries = 90
    query_themes = [themes[i % len(themes)] for i in range(n_queries)]
    queries = pd.DataFrame(
        {
            "query_id": [f"q{i}" for i in range(n_queries)],
            "text": [
                f"question on {' '.join(rng.choice(WORDS[t], 2))}?" for t in query_themes
            ],  # some repeat
        }
    )
    query_vectors = np.stack([centroids[themes.index(t)] + 0.7 * rng.normal(size=D_IN) for t in query_themes])
    qrels = pd.DataFrame({"query_id": ["q0"], "doc_id": [rows[0]["doc_id"]], "score": [1]})
    order = np.random.default_rng(seed + 1).permutation(len(corpus))
    n_val = len(corpus) // 4
    dataset = RetrievalDataset(
        name=name,
        corpus=corpus,
        queries={"train": queries},
        qrels={"train": qrels},
        corpus_splits={"train": order[n_val:], "val": order[:n_val]},
    )
    return EmbeddedDataset(
        dataset=dataset, embedder="toy", corpus=np.stack(vectors), queries={"train": query_vectors}
    )


def write_codes(path: Path, frame: pd.DataFrame, codes) -> Path:
    """A ``*_with_sae.pkl`` file as the paper's convert_embeddings.py writes it (sparse TopK storage)."""
    frame = frame.copy()
    frame["sae embeddings"] = [
        (
            codes.indices[codes.indptr[i] : codes.indptr[i + 1]].astype(np.int32),
            codes.data[codes.indptr[i] : codes.indptr[i + 1]].astype(np.float32),
        )
        for i in range(codes.shape[0])
    ]
    frame.attrs["sae_hidden"] = codes.shape[1]
    frame.to_pickle(path)
    return path


def paper_corpus_frame(corpus: pd.DataFrame) -> pd.DataFrame:
    """The paper's corpus columns: the embedded string in ``_prepended_text``, the body in ``text``."""
    return pd.DataFrame(
        {
            "_prepended_text": corpus["text"].to_numpy(),
            "text": corpus["body"].to_numpy(),
            "title": corpus["title"].to_numpy(),
            "source_folder": corpus["source_folder"].to_numpy(),
        }
    )


def run_build(paper, monkeypatch, tmp_path: Path, profiles: dict, extra: list[str], name: str) -> dict:
    profiles_path = tmp_path / f"profiles_{name}.json"
    profiles_path.write_text(json.dumps(profiles))
    output = tmp_path / f"eval_rows_{name}.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_interpret_eval",
            "--neuron-profiles",
            str(profiles_path),
            "--hidden",
            str(N_LATENTS),
            "--rank-band",
            "1",
            "20",
            "--per-neuron",
            "5",
            "--n-negatives",
            "5",
            "--seed",
            "42",
            "--output",
            str(output),
            *extra,
        ],
    )
    paper.build.main()
    return json.loads(output.read_text())


@pytest.fixture(scope="module")
def setting():
    torch.manual_seed(0)
    sae = MedSAE(d_in=D_IN, n_latents=N_LATENTS, k=K, embedder="toy")
    data = make_data("toy", seed=0, n_articles=120)
    return sae, data, Interpreter(sae, data, show_progress=False)


def test_evidence_matches_select_neurons(paper, setting, tmp_path):
    sae, data, interpreter = setting
    corpus, splits = data.dataset.corpus, data.dataset.corpus_splits
    train = write_codes(
        tmp_path / "train.pkl",
        paper_corpus_frame(corpus.iloc[splits["train"]]),
        interpreter.corpus_codes[splits["train"]],
    )
    queries = write_codes(
        tmp_path / "queries.pkl",
        data.dataset.queries["train"][["text"]],
        encode_sparse(sae, data.queries["train"]),
    )
    profiles = paper.select.select_neurons(
        train,
        "text",
        "source_folder",
        hidden=N_LATENTS,
        queries_file=queries,
        fr_range=(0.0, 1.0),
        entropy_source="corpus",
    )
    features = interpreter.select_features(theme_column="source_folder")
    assert features == [neuron["neuron"] for neuron in profiles["neurons"]]
    assert len(features) >= 10
    for card, neuron in zip(interpreter.describe(features), profiles["neurons"], strict=True):
        assert card.firing_rate == neuron["firing_rate_corpus"]
        assert ", ".join(card.tfidf_terms) == neuron["top_terms"]
        for ours, theirs in (
            (card.top_documents, neuron["corpus_snippets"]),
            (card.top_queries, neuron["query_snippets"]),
        ):
            assert [(s["text"], s["title"], s["activation"]) for s in ours] == [
                (s["text"], s["title"], s["activation"]) for s in theirs
            ]
        assert [(s["text"], s["title"]) for s in card.negatives] == [
            (s["text"], s["title"]) for s in neuron["negative_snippets"]
        ]


def check_scoring(paper, report, cards, profiles_by_feature, eval_rows, llm, ood):
    assert report.features == eval_rows["neurons"]
    assert report.rows == eval_rows["rows"]
    assert len(report.features) >= 6 and len(report.rows) >= 25
    calls = iter(llm.calls)
    for method in report.methods:
        label_rows = {
            card.feature: {"topic": card.topic, "description": card.description, "keywords": card.keywords}
            for card in cards
        }
        descriptions = {
            f: paper.predict.build_class_description(profiles_by_feature[f], method, label_rows[f], 1000)
            for f in eval_rows["neurons"]
        }
        pairs = paper.predict.sample_pairs(
            eval_rows["rows"], eval_rows["neurons"], 3, 3, 42, eval_rows.get("band_thresholds")
        )
        predictions = report.predictions[method]
        assert [(p["feature"], p["row"], p["ground_truth"]) for p in predictions] == [
            (f, eval_rows["rows"][i]["row"], truth) for f, i, truth in pairs
        ]
        system = paper.predict.build_system_prompt(method)
        for f, i, _ in pairs:
            text = paper.predict._cap(eval_rows["rows"][i]["text"], 1000)
            assert next(calls) == (system, paper.predict.build_user(True, f, descriptions[f], text, ood))
        as_paper = [dict(p, neuron=p["feature"]) for p in predictions]
        metrics = paper.predict.score_pairs(as_paper, eval_rows["neurons"])
        ours = feature_metrics(predictions, eval_rows["neurons"])
        for f in eval_rows["neurons"]:
            assert ours[f]["support"] == metrics[str(f)]["support"]
            assert ours[f]["n_pairs"] == metrics[str(f)]["n_pairs"]
            assert same(ours[f]["pearson"], metrics[str(f)].get("pearson", float("nan")))
        if paper.auroc is not None:
            keep = {str(f) for f in eval_rows["neurons"]}
            expected = paper.auroc.neuron_aurocs({"predictions": as_paper}, keep)
            assert {
                str(f): v for f, v in feature_aurocs(predictions, eval_rows["neurons"]).items()
            } == expected


def test_in_domain_scoring_matches_the_paper_scripts(paper, setting, tmp_path, monkeypatch):
    sae, data, interpreter = setting
    corpus, splits = data.dataset.corpus, data.dataset.corpus_splits
    files = [
        write_codes(
            tmp_path / f"{split}.pkl",
            paper_corpus_frame(corpus.iloc[splits[split]]),
            interpreter.corpus_codes[splits[split]],
        )
        for split in ("train", "val")
    ]
    queries = write_codes(
        tmp_path / "queries.pkl",
        data.dataset.queries["train"][["text"]],
        encode_sparse(sae, data.queries["train"]),
    )
    profiles = paper.select.select_neurons(
        files[0],
        "text",
        "source_folder",
        hidden=N_LATENTS,
        queries_file=queries,
        fr_range=(0.0, 1.0),
        entropy_source="corpus",
        max_neurons=N_SCORED,
    )
    cards = interpreter.describe(
        interpreter.select_features(n=N_SCORED, theme_column="source_folder"), llm=FakeLLM()
    )
    eval_rows = run_build(
        paper, monkeypatch, tmp_path, profiles, ["--eval-file", *map(str, files)], "in_domain"
    )
    llm = FakeLLM()
    report = interpreter.score(cards, llm)
    profiles_by_feature = {neuron["neuron"]: neuron for neuron in profiles["neurons"]}
    check_scoring(paper, report, cards, profiles_by_feature, eval_rows, llm, ood=False)


def test_out_of_domain_scoring_matches_the_paper_scripts(paper, setting, tmp_path, monkeypatch):
    sae, data, interpreter = setting
    corpus, splits = data.dataset.corpus, data.dataset.corpus_splits
    train = write_codes(
        tmp_path / "train.pkl",
        paper_corpus_frame(corpus.iloc[splits["train"]]),
        interpreter.corpus_codes[splits["train"]],
    )
    queries = write_codes(
        tmp_path / "queries.pkl",
        data.dataset.queries["train"][["text"]],
        encode_sparse(sae, data.queries["train"]),
    )
    profiles = paper.select.select_neurons(
        train,
        "text",
        "source_folder",
        hidden=N_LATENTS,
        queries_file=queries,
        fr_range=(0.0, 1.0),
        entropy_source="corpus",
        max_neurons=N_SCORED,
    )
    cards = interpreter.describe(
        interpreter.select_features(n=N_SCORED, theme_column="source_folder"), llm=FakeLLM()
    )
    other = make_data("NFCorpus", seed=5, n_articles=60)
    ood_dir = tmp_path / "ood"
    (ood_dir / "NFCorpus").mkdir(parents=True)
    write_codes(
        ood_dir / "NFCorpus" / "corpus_with_sae.pkl",
        paper_corpus_frame(other.dataset.corpus),
        encode_sparse(sae, other.corpus),
    )
    extra = [
        "--ood-dir",
        str(ood_dir),
        "--dataset-name",
        "CURE",
        "--exclude",
        "Bioinformatics",
        "Biology",
        "Medical-Sciences",
        "MIRIAD",
    ]
    eval_rows = run_build(paper, monkeypatch, tmp_path, profiles, extra, "ood")
    assert eval_rows["ood_sources"] == ["NFCorpus"]
    llm = FakeLLM()
    report = interpreter.score(cards, llm, data=other)
    profiles_by_feature = {neuron["neuron"]: neuron for neuron in profiles["neurons"]}
    check_scoring(paper, report, cards, profiles_by_feature, eval_rows, llm, ood=True)
