"""The Interpreter facade end to end on a tiny synthetic dataset and SAE, with a fake LLM."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from doctosae.data.dataset import EmbeddedDataset, RetrievalDataset
from doctosae.interpret import FakeLLM, FeatureCard, Interpreter, load_cards
from doctosae.interpret.cards import LABEL_ERROR
from doctosae.interpret.prompts import (
    OOD_NOTE,
    interpreter_system_prompt,
    interpreter_user_prompt,
    label_instruction,
)
from doctosae.interpret.scoring import exemplars
from doctosae.sae import DoctoSAE

D_IN, N_LATENTS, K = 16, 48, 6
TOPICS = {
    "cardiology": ["heart", "artery", "cardiac", "valve", "rhythm", "pressure", "infarction"],
    "nephrology": ["kidney", "stones", "renal", "dialysis", "urine", "nephron", "bladder"],
    "dermatology": ["skin", "rash", "eczema", "psoriasis", "acne", "lesion", "itching"],
    "neurology": ["brain", "seizure", "stroke", "nerve", "memory", "migraine", "neuron"],
}


def make_dataset(name: str = "toy", seed: int = 0, n_articles: int = 30, n_train_queries: int = 80):
    """Passages of articles on four topics, embeddings near one centroid per topic, train and val splits."""
    rng = np.random.default_rng(seed)
    centroids = np.random.default_rng(100).normal(size=(len(TOPICS), D_IN))
    names = list(TOPICS)
    rows, vectors = [], []
    for article in range(n_articles):
        topic = names[article % len(names)]
        title = f"{topic.title()} study {article}"
        for chunk in range(3):
            body = " ".join(rng.choice(TOPICS[topic], 6)) + f" part {chunk} of {name} article {article}"
            rows.append(
                {
                    "doc_id": f"{name}{article}_{chunk}",
                    "text": f"{title} {body}",
                    "title": title,
                    "body": body,
                    "source_folder": topic,
                }
            )
            vectors.append(centroids[article % len(names)] + 0.6 * rng.normal(size=D_IN))
    rows.append(dict(rows[0], doc_id=f"{name}_duplicate"))  # same text as the first passage
    vectors.append(vectors[0])
    corpus = pd.DataFrame(rows)

    def queries(n: int, offset: int) -> tuple[pd.DataFrame, np.ndarray]:
        topics = [names[i % len(names)] for i in range(n)]
        texts = [
            f"what about {' '.join(rng.choice(TOPICS[t], 3))} case {offset + i}?"
            for i, t in enumerate(topics)
        ]
        embeddings = np.stack([centroids[names.index(t)] + 0.6 * rng.normal(size=D_IN) for t in topics])
        frame = pd.DataFrame({"query_id": [f"q{offset + i}" for i in range(n)], "text": texts})
        return frame, embeddings

    train_queries, train_embeddings = queries(n_train_queries, 0)
    test_queries, test_embeddings = queries(8, 1000)
    qrels = {
        split: pd.DataFrame(
            {"query_id": frame["query_id"], "doc_id": corpus["doc_id"].iloc[: len(frame)], "score": 1}
        )
        for split, frame in (("train", train_queries), ("test", test_queries))
    }
    order = np.random.default_rng(seed + 1).permutation(len(corpus))
    n_val = len(corpus) // 5
    dataset = RetrievalDataset(
        name=name,
        corpus=corpus,
        queries={"train": train_queries, "test": test_queries},
        qrels=qrels,
        corpus_splits={"train": order[n_val:], "val": order[:n_val]},
    )
    return EmbeddedDataset(
        dataset=dataset,
        embedder="toy-encoder",
        corpus=np.stack(vectors),
        queries={"train": train_embeddings, "test": test_embeddings},
    )


@pytest.fixture(scope="module")
def data() -> EmbeddedDataset:
    return make_dataset()


@pytest.fixture(scope="module")
def sae() -> DoctoSAE:
    torch.manual_seed(0)
    return DoctoSAE(d_in=D_IN, n_latents=N_LATENTS, k=K, embedder="toy-encoder")


@pytest.fixture(scope="module")
def interpreter(sae, data) -> Interpreter:
    return Interpreter(sae, data, show_progress=False)


@pytest.fixture(scope="module")
def features(interpreter) -> list[int]:
    selected = interpreter.select_features()
    assert len(selected) >= 4, "the synthetic data must give a few interpretable features"
    return selected[:8]


@pytest.fixture(scope="module")
def cards(interpreter, features) -> list[FeatureCard]:
    return interpreter.describe(features, llm=FakeLLM())


def test_evidence_is_the_train_corpus_split_and_the_train_queries(interpreter, data):
    documents = interpreter.index.documents
    train_rows = data.dataset.corpus_splits["train"]
    texts = data.dataset.corpus["text"].to_numpy()
    expected = [
        row for position, row in enumerate(train_rows) if texts[row] not in set(texts[train_rows[:position]])
    ]
    assert documents.rows.tolist() == expected
    assert len(interpreter.index.queries) == len(data.dataset.queries["train"])
    assert interpreter.corpus_codes.shape == (len(data.dataset.corpus), N_LATENTS)
    assert (interpreter.corpus_codes.getnnz(axis=1) <= K).all()


def test_selected_features_meet_the_minimum_counts(interpreter, features):
    index = interpreter.index
    for feature in interpreter.select_features():
        assert index.document_counts[feature] >= 5 and index.query_counts[feature] >= 5
    assert interpreter.select_features(n=3) == interpreter.select_features()[:3]


def test_theme_column_orders_by_entropy(interpreter):
    ordered = interpreter.select_features(theme_column="source_folder")
    assert sorted(ordered) == interpreter.select_features()
    with pytest.raises(ValueError, match="column"):
        interpreter.select_features(theme_column="missing")


def test_cards_hold_capped_evidence_negatives_and_terms(interpreter, cards):
    index = interpreter.index
    columns = index.documents.columns
    for card in cards:
        assert card.n_documents == index.document_counts[card.feature]
        assert card.firing_rate == pytest.approx(card.n_documents / len(index.documents))
        activations = [s["activation"] for s in card.top_documents]
        assert 0 < len(activations) <= 5 and activations == sorted(activations, reverse=True)
        titles = [s["title"] for s in card.top_documents]
        assert all(titles.count(title) <= 2 for title in titles)
        assert all(
            s["text"] == f"{s['title']} " + s["text"][len(s["title"]) + 1 :] for s in card.top_documents
        )
        assert 0 < len(card.top_queries) <= 5 and all(s["title"] is None for s in card.top_queries)
        assert len(card.negatives) == 2 and all("activation" not in s for s in card.negatives)
        rows = {text: row for row, text in enumerate(index.documents.texts)}
        assert all(columns[rows[s["text"]], card.feature] <= 0 for s in card.negatives)
        assert 0 < len(card.tfidf_terms) <= 5
        assert card.has_label and card.topic and card.keywords


def test_describe_sends_the_paper_prompts_in_batches_of_twenty(interpreter, features):
    sizes = []

    class Recorder(FakeLLM):
        def generate(self, system, users):
            sizes.append(len(users))
            return super().generate(system, users)

    llm = Recorder()
    many = (features * 6)[:45]
    interpreter.describe(many, llm=llm)
    assert sizes == [20, 20, 5]
    assert all(system == interpreter_system_prompt(10) for system, _ in llm.calls)
    assert all(user.endswith(label_instruction(10)) for _, user in llm.calls)


def test_failed_label_calls_become_label_errors(interpreter, features):
    def broken(system, user):
        raise ConnectionError("server down")

    labelled = interpreter.describe(features[:3], llm=FakeLLM(broken))
    assert all(card.topic == LABEL_ERROR and "ConnectionError" in card.description for card in labelled)
    assert not any(card.has_label for card in labelled)


def test_one_failed_card_does_not_cost_its_chunk(interpreter, features):
    failing = features[1]

    def reply(system, user):
        if user == failing_prompt[0]:
            raise TimeoutError("rate limited")
        return FakeLLM().reply(system, user)

    plain = interpreter.describe(features[:4])
    failing_prompt = [interpreter_user_prompt(plain[1])]
    labelled = interpreter.describe(features[:4], llm=FakeLLM(reply))
    assert [card.has_label for card in labelled] == [True, False, True, True]
    assert labelled[1].feature == failing and labelled[1].topic == LABEL_ERROR
    assert "TimeoutError: rate limited" in labelled[1].description


def test_an_unavailable_interpreter_llm_is_not_called_forever(interpreter, features):
    calls = []

    def broken(system, user):
        calls.append(user)
        raise ConnectionError("server down")

    many = (features * 10)[:70]  # 4 chunks of 20 and one of 10
    labelled = interpreter.label(interpreter.describe(many), FakeLLM(broken), batch_size=20)
    assert all(card.topic == LABEL_ERROR for card in labelled)
    # Three chunks: the batch call (FakeLLM stops at its first turn), then each card alone; nothing more.
    assert len(calls) == 3 * (1 + 20)
    assert "not sent" in labelled[-1].description


def test_describe_rejects_unknown_features(interpreter):
    with pytest.raises(ValueError, match="outside"):
        interpreter.describe([N_LATENTS])


def test_in_domain_score_excludes_what_the_interpreter_saw(interpreter, data, cards):
    llm = FakeLLM()
    report = interpreter.score(cards, llm)
    assert not report.out_of_domain and report.dataset == "toy"
    assert set(report.to_frame().index) == {"tfidf", "llm", "snippets"}
    shown_texts, shown_titles = exemplars(cards)
    title_of = dict(zip(data.dataset.corpus["text"], data.dataset.corpus["title"], strict=True))
    for row in report.rows:
        assert row["text"] not in shown_texts and title_of[row["text"]] not in shown_titles
        assert row["source"] in {"train", "val"}
    assert all(OOD_NOTE not in user for _, user in llm.calls)
    json.dumps(report.to_dict())


def test_another_object_of_the_same_dataset_is_scored_in_domain(interpreter, data, cards, caplog):
    with caplog.at_level("WARNING", logger="doctosae.interpret.interpreter"):
        copied = interpreter.score(cards, FakeLLM(), data=copy.copy(data), methods=["tfidf"])
    reference = interpreter.score(cards, FakeLLM(), methods=["tfidf"])
    assert not copied.out_of_domain and "scored in domain" in caplog.text
    assert copied.rows == reference.rows and copied.predictions == reference.predictions


def test_out_of_domain_score_uses_the_other_corpus(interpreter, cards):
    other = make_dataset(name="other", seed=7, n_articles=12)
    llm = FakeLLM()
    report = interpreter.score(cards, llm, data=other, methods=["tfidf"])
    assert report.out_of_domain and report.dataset == "other"
    assert all(row["source"] == "other" and " other article " in row["text"] for row in report.rows)
    assert llm.calls and all(OOD_NOTE in user for _, user in llm.calls)


def test_explain_uses_stored_embeddings_and_ranks_against_the_corpus(interpreter, data, cards):
    corpus = data.dataset.corpus["text"].tolist()
    query = data.dataset.queries["train"]["text"].iloc[0]
    explanation = interpreter.explain(query, corpus[-5], corpus[1], cards=cards)
    assert explanation.rank_a <= explanation.rank_b
    assert {explanation.passage_a, explanation.passage_b} == {corpus[-5], corpus[1]}
    assert explanation.swapped == (explanation.passage_a == corpus[1])
    assert 0 < len(explanation.features) <= 4
    named = {card.feature: card.topic for card in cards}
    assert explanation.names == [named.get(feature) for feature in explanation.features]
    # The ranks are those of a cosine search over the codes of the whole corpus.
    codes = interpreter.corpus_codes.toarray().astype(np.float64)
    codes /= np.maximum(np.linalg.norm(codes, axis=1, keepdims=True), 1e-12)
    query_code = interpreter.sae.encode(torch.from_numpy(data.queries["train"][:1])).detach().numpy().ravel()
    scores = codes @ (query_code / np.linalg.norm(query_code))
    row_a = corpus.index(explanation.passage_a)
    assert explanation.rank_a == 1 + int((scores > scores[row_a] + 1e-9).sum())
    assert explanation.cosine_a == pytest.approx(scores[row_a], abs=1e-5)


def test_explain_accepts_embeddings_and_embeds_unknown_texts(interpreter, data):
    vectors = data.corpus
    explanation = interpreter.explain(data.queries["test"][0], vectors[3], vectors[10], rank=False)
    assert explanation.query is None and explanation.rank_a is None and not explanation.swapped

    class StubEmbedder:
        def __init__(self):
            self.seen = []

        def encode_queries(self, texts):
            self.seen += list(texts)
            return data.queries["test"][:1]

        def encode_documents(self, texts):
            self.seen += list(texts)
            return vectors[:1]

    embedder = StubEmbedder()
    explanation = interpreter.explain(
        "a brand new question", "a brand new passage", data.dataset.corpus["text"][5], embedder=embedder
    )
    assert embedder.seen == ["a brand new question", "a brand new passage"]
    assert explanation.rank_a is not None


def test_cards_round_trip_through_json(interpreter, cards, tmp_path):
    path = tmp_path / "cards.json"
    interpreter.save_cards(cards, path)
    payload = json.loads(path.read_text())
    assert payload["metadata"]["dataset"] == "toy" and payload["metadata"]["n_latents"] == N_LATENTS
    assert Interpreter.load_cards(path) == cards
    assert load_cards(path) == cards
    markdown = cards[0].to_markdown()
    assert markdown.startswith(f"### Feature {cards[0].feature}: {cards[0].topic}")
    assert f"**{cards[0].top_documents[0]['title']}**" in markdown
    (tmp_path / "other.json").write_text("{}")
    with pytest.raises(ValueError, match="feature cards"):
        load_cards(tmp_path / "other.json")


def test_embedder_aliases_are_resolved_before_warning(data, caplog):
    sae = DoctoSAE(d_in=D_IN, n_latents=N_LATENTS, k=K, embedder="bge")
    aliased = EmbeddedDataset(data.dataset, "BAAI/bge-large-en-v1.5", data.corpus, dict(data.queries))
    with caplog.at_level("WARNING", logger="doctosae"):
        Interpreter(sae, aliased, show_progress=False)
    assert "trained on" not in caplog.text
    with caplog.at_level("WARNING", logger="doctosae"):
        Interpreter(sae, data, show_progress=False)
    assert "trained on bge embeddings" in caplog.text


def test_other_embedding_settings_are_flagged(data, caplog):
    sae = DoctoSAE(d_in=D_IN, n_latents=N_LATENTS, k=K, embedder="toy-encoder", query_instruction="query: ")
    other = EmbeddedDataset(
        data.dataset, "toy-encoder", data.corpus, dict(data.queries), query_instruction=""
    )
    with caplog.at_level("WARNING", logger="doctosae"):
        Interpreter(sae, other, show_progress=False)
    assert "query_instruction 'query: ' against ''" in caplog.text


def test_explain_builds_the_sae_encoder_on_the_interpreter_device(sae, data, monkeypatch):
    import doctosae.embed

    built = {}

    class RecordingEmbedder:
        @classmethod
        def from_sae(cls, model, **options):
            built.update(options)
            return cls()

        def encode_queries(self, texts):
            return data.queries["test"][:1]

        def encode_documents(self, texts):
            return data.corpus[:1]

    monkeypatch.setattr(doctosae.embed, "Embedder", RecordingEmbedder)
    interpreter = Interpreter(sae, data, split=None, device="cpu", show_progress=False)
    interpreter.explain("an unseen question", "an unseen passage", data.dataset.corpus["text"][5])
    assert built == {"device": "cpu", "show_progress": False}


def test_input_checks(sae, data):
    with pytest.raises(ValueError, match="query embeddings"):
        Interpreter(sae, data, split="val")
    with pytest.raises(ValueError, match="corpus split"):
        Interpreter(sae, data, corpus_split="test")
    small = DoctoSAE(d_in=D_IN + 1, n_latents=8, k=2)
    with pytest.raises(ValueError, match="dimensional"):
        Interpreter(small, data)
    documents_only = Interpreter(sae, data, split=None, show_progress=False)
    assert documents_only.index.queries is None
    assert all(documents_only.index.document_counts[f] >= 5 for f in documents_only.select_features())
