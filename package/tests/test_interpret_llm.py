"""LLM backends: the fake model, the OpenAI compatible client (stubbed, offline) and a small local model."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from medsae.interpret.cards import FeatureCard
from medsae.interpret.labeling import parse_label
from medsae.interpret.llm import (
    LLM,
    FakeLLM,
    OpenAICompatibleLLM,
    PartialReplies,
    TransformersLLM,
    _dtype_keyword,
    generate_isolated,
)
from medsae.interpret.prompts import (
    feature_description,
    interpreter_system_prompt,
    interpreter_user_prompt,
    judge_system_prompt,
    judge_user_prompt,
)
from medsae.interpret.scoring import parse_score


def make_card() -> FeatureCard:
    return FeatureCard(
        feature=3,
        n_documents=7,
        n_queries=5,
        firing_rate=0.07,
        top_documents=[
            {
                "id": "1",
                "text": "Kidney stones Kidney stones cause renal colic",
                "title": "Kidney stones",
                "activation": 0.4,
            },
            {"id": "2", "text": "Renal colic and kidney stones in adults", "title": None, "activation": 0.3},
        ],
        top_queries=[{"id": "q", "text": "how are kidney stones treated?", "title": None, "activation": 0.2}],
        negatives=[{"id": "9", "text": "Asthma inhalers for children", "title": None}],
        tfidf_terms=["kidney", "stones", "renal", "colic", "kidney stones"],
    )


def test_fake_llm_replies_parse_with_the_paper_parsers():
    llm = FakeLLM()
    assert isinstance(llm, LLM)
    card = make_card()
    [label_reply] = llm.generate(interpreter_system_prompt(), [interpreter_user_prompt(card)])
    label = parse_label(label_reply)
    assert label.topic == "kidney" and "stones" in label.keywords
    description = feature_description(card, "tfidf")
    replies = llm.generate(
        judge_system_prompt("tfidf"),
        [
            judge_user_prompt(3, description, "Kidney stones and renal colic in the emergency room."),
            judge_user_prompt(3, description, "Asthma control in children."),
        ],
    )
    high, low = (parse_score(reply)[0] for reply in replies)
    assert high > low == 0.0
    assert len(llm.calls) == 3


def test_fake_llm_accepts_a_custom_reply():
    llm = FakeLLM(lambda system, user: f"{len(system)}:{user}")
    assert llm.generate("abc", ["x", "y"]) == ["3:x", "3:y"]


def test_fake_llm_repr_has_no_memory_address():
    # The repr is recorded in ScoreReport.llm and scores.json.
    assert repr(FakeLLM()) == "FakeLLM()"
    assert repr(FakeLLM(lambda system, user: "")) == "FakeLLM(reply=custom)"


class StubCompletions:
    """Records requests and answers with the user turn reversed, after a delay inverse to its position."""

    def __init__(self):
        self.requests = []
        self.lock = threading.Lock()

    def create(self, **request):
        with self.lock:
            self.requests.append(request)
        user = request["messages"][1]["content"]
        time.sleep(0.01 * (5 - int(user[-1])))
        if user.startswith("fail"):
            raise TimeoutError(f"no reply to {user}")
        if user == "empty 0":
            return SimpleNamespace(choices=[])
        content = None if user == "none 0" else user[::-1]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def stub_client() -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=StubCompletions()))


def test_openai_compatible_llm_sends_greedy_requests_and_keeps_the_order():
    client = stub_client()
    llm = OpenAICompatibleLLM("some-model", client=client, max_workers=4)
    users = [f"user {i}" for i in range(5)]
    assert llm.generate("system prompt", users) == [user[::-1] for user in users]
    requests = client.chat.completions.requests
    assert len(requests) == 5
    for request in requests:
        assert request["model"] == "some-model"
        assert request["temperature"] == 0.0 and request["max_tokens"] == 400
        assert request["messages"][0] == {"role": "system", "content": "system prompt"}
        assert request["messages"][1]["role"] == "user"
    assert llm.generate("s", ["none 0"]) == [""]
    assert llm.generate("s", ["empty 0"]) == [""]  # no choices: a parse error later, never a crash
    assert llm.generate("s", []) == []


def test_openai_compatible_llm_keeps_the_replies_of_a_partly_failed_batch():
    llm = OpenAICompatibleLLM("some-model", client=stub_client(), max_workers=4)
    users = ["user 1", "fail 2", "user 3"]
    with pytest.raises(PartialReplies) as caught:
        llm.generate("s", users)
    assert caught.value.replies == ["1 resu", None, "3 resu"] and list(caught.value.errors) == [1]
    requests = llm.client.chat.completions.requests
    before = len(requests)
    outcomes = generate_isolated(llm, "s", users)
    assert outcomes[0] == "1 resu" and outcomes[2] == "3 resu" and isinstance(outcomes[1], TimeoutError)
    sent = [request["messages"][1]["content"] for request in requests[before:]]
    assert sorted(sent) == ["fail 2", "fail 2", "user 1", "user 3"]  # only the failed turn is sent again


def test_openai_compatible_llm_request_options():
    client = stub_client()
    llm = OpenAICompatibleLLM(
        "reasoning-model", client=client, temperature=None, max_tokens=None,
        request_kwargs={"max_completion_tokens": 400, "reasoning_effort": "low"},
    )  # fmt: skip
    llm.generate("s", ["user 1"])
    request = client.chat.completions.requests[0]
    assert "temperature" not in request and "max_tokens" not in request
    assert request["max_completion_tokens"] == 400 and request["reasoning_effort"] == "low"


def test_openai_compatible_llm_reads_the_environment(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm = OpenAICompatibleLLM("served-model")
    assert str(llm.client.base_url).rstrip("/") == "http://localhost:8000/v1"
    assert llm.client.api_key == "EMPTY"
    llm = OpenAICompatibleLLM("served-model", base_url="http://127.0.0.1:11434/v1", api_key="local-key")
    assert str(llm.client.base_url).rstrip("/") == "http://127.0.0.1:11434/v1"
    assert llm.client.api_key == "local-key"


def test_openai_compatible_llm_never_falls_back_to_a_public_endpoint(monkeypatch, caplog):
    pytest.importorskip("openai")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="refusing to fall back"):
        OpenAICompatibleLLM("served-model")
    with caplog.at_level("WARNING", logger="medsae.interpret.llm"):
        OpenAICompatibleLLM("gpt-model", base_url="https://api.openai.com/v1", api_key="test-key")
    assert "sent to api.openai.com" in caplog.text


class StubBatch(dict):
    def to(self, device):
        return self


class StubTokenizer:
    """Chat template stand in: optionally refuses system turns, like the Gemma 2 template."""

    pad_token_id = 0

    def __init__(self, system_turn: bool):
        self.system_turn = system_turn
        self.conversations = []

    def apply_chat_template(self, conversations, **kwargs):
        import jinja2
        import torch

        if not self.system_turn and any(m["role"] == "system" for c in conversations for m in c):
            raise jinja2.TemplateError("System role not supported")
        assert kwargs == {
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
            "padding": True,
        }
        self.conversations = conversations
        return StubBatch(input_ids=torch.zeros((len(conversations), 3), dtype=torch.long))

    def decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens
        return f"{len(tokens)} new tokens"


class StubModel:
    device = "cpu"

    def generate(self, input_ids, max_new_tokens, do_sample, pad_token_id):
        import torch

        assert do_sample is False and max_new_tokens == 400 and pad_token_id == 0
        return torch.zeros((input_ids.shape[0], input_ids.shape[1] + 2), dtype=torch.long)


@pytest.mark.parametrize("system_turn", [True, False])
def test_transformers_llm_batches_greedy_generation(system_turn):
    pytest.importorskip("jinja2")
    llm = TransformersLLM.__new__(TransformersLLM)  # skip loading weights: stub tokenizer and model
    llm.tokenizer, llm.model, llm.max_new_tokens, llm.model_name, llm.deterministic = (
        StubTokenizer(system_turn),
        StubModel(),
        400,
        "s",
        False,
    )
    assert llm.generate("sys", ["u1", "u2"]) == ["2 new tokens", "2 new tokens"]
    if system_turn:
        assert llm.tokenizer.conversations[0] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
        ]
    else:
        assert llm.tokenizer.conversations == [
            [{"role": "user", "content": "sys\n\nu1"}],
            [{"role": "user", "content": "sys\n\nu2"}],
        ]
    assert llm.generate("sys", []) == []


def test_transformers_llm_pads_left_and_scopes_deterministic_kernels(monkeypatch):
    """Offline: stub loaders stand in for the Hub, so a regression of the padding side fails here."""
    transformers = pytest.importorskip("transformers")
    import torch

    class Tokenizer(StubTokenizer):
        pad_token, eos_token, padding_side = None, "<eos>", "right"

    class Model(StubModel):
        def eval(self):
            return self

        def generate(self, input_ids, **kwargs):
            seen.append(torch.are_deterministic_algorithms_enabled())
            return super().generate(input_ids, **kwargs)

    seen = []
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer(True))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: Model())
    before = torch.are_deterministic_algorithms_enabled()
    llm = TransformersLLM("some/chat-model")
    assert llm.tokenizer.padding_side == "left" and llm.tokenizer.pad_token == "<eos>"
    assert torch.are_deterministic_algorithms_enabled() == before  # building the model changes nothing
    llm.generate("sys", ["u1"])
    assert seen == [True] and torch.are_deterministic_algorithms_enabled() == before


def test_dtype_keyword_follows_the_transformers_version():
    assert _dtype_keyword("4.50.1") == "torch_dtype"
    assert _dtype_keyword("4.50.3") == "torch_dtype"
    assert _dtype_keyword("4.56.0") == "dtype"
    assert _dtype_keyword("5.10.1") == "dtype"
    assert _dtype_keyword("4.57.0.dev0") == "dtype"


@pytest.mark.network
def test_transformers_llm_generates_a_batch_with_a_small_model():
    pytest.importorskip("transformers")
    llm = TransformersLLM(
        "HuggingFaceTB/SmolLM2-135M-Instruct", device="cpu", dtype="float32", max_new_tokens=24
    )
    assert llm.tokenizer.padding_side == "left"  # this tokenizer pads right by default
    card = make_card()
    prompts = [
        judge_user_prompt(3, feature_description(card, "tfidf"), text)
        for text in (
            "Kidney.",
            "Kidney stones and renal colic in adults.",
            "Asthma in children who wheeze at night.",
        )
    ]
    # Left padding: a batch of prompts of different lengths gives the replies of the prompts sent alone.
    batch = llm.generate(judge_system_prompt("tfidf"), prompts)
    assert batch == [llm.generate(judge_system_prompt("tfidf"), [prompt])[0] for prompt in prompts]
    replies = llm.generate(
        judge_system_prompt("tfidf"),
        [
            judge_user_prompt(3, feature_description(card, "tfidf"), "Kidney stones and renal colic."),
            judge_user_prompt(3, feature_description(card, "tfidf"), "Asthma in children."),
        ],
    )
    assert len(replies) == 2 and all(isinstance(reply, str) and reply for reply in replies)
    # Greedy decoding: the same batch gives the same replies.
    assert llm.generate(
        judge_system_prompt("tfidf"), [judge_user_prompt(3, "  kidney", "Kidney.")]
    ) == llm.generate(judge_system_prompt("tfidf"), [judge_user_prompt(3, "  kidney", "Kidney.")])
