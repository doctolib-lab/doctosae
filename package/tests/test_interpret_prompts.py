"""Prompts of the interpretation pipeline: verbatim strings pinned by digest and exact renderings."""

from __future__ import annotations

import hashlib

import pytest

from doctosae.interpret import prompts
from doctosae.interpret.cards import PARSE_ERROR, FeatureCard

# SHA256 of the rendered prompts of the paper code (label_neurons.py, predict_neurons_binary.py).
INTERPRETER_SYSTEM_DIGEST = "831f3206a46701a275412aadd0744593fe7c8bcc3078f5b386fe906bfca0f3ad"
LABEL_INSTRUCTION_DIGEST = "03c022527c24fc8ca963d3ae8879618c578dbd63e5a5dcf50b72f781b95da7bc"
JUDGE_SYSTEM_DIGESTS = {
    "snippets": "5fa04355e7e0b135956022cded0b95898834d8fadca7e203c8c5acbf0e729aeb",
    "tfidf": "9a8b7c2e47d9760a23c2d742e060583b446dab33045865ec727bcc5fab1c5fb5",
    "llm": "f9c542b513b632953dc905765330d50f38bf8dbf0a5fcee2e38dd33798f946ba",
}
SCORE_INSTRUCTION_DIGEST = "2e8d8728b3a20ae2e538b5027242732a47b2d63ae3c195287d8cd45819f2a79d"
USER_PROMPT_DIGEST = "c8c594b33596a1fb91f4d43c32013d62d4be254df50d1f6ee01d3f9e9bf4eddb"
OOD_NOTE_DIGEST = "f111d1db5fafdc95ccfa09acfa40e34d718099479c9b481289f805e4688da8ae"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_card(**overrides) -> FeatureCard:
    values = {
        "feature": 2177,
        "n_documents": 10,
        "n_queries": 6,
        "firing_rate": 0.01,
        "top_documents": [
            {
                "id": "d0",
                "text": "Bone tissue engineering A common view on bone tissue engineering portrays four "
                "vital aspects.",
                "title": "Bone tissue engineering",
                "activation": 0.2912,
            },
            {"id": "d1", "text": "untitled passage text", "title": None, "activation": 0.25},
        ],
        "top_queries": [
            {
                "id": "q0",
                "text": "What is the role of cells in these bone grafts?",
                "title": None,
                "activation": 0.2701,
            },
        ],
        "negatives": [
            {"id": "d9", "text": "Sleep apnea Obstructive sleep apnea is common.", "title": "Sleep apnea"},
        ],
        "tfidf_terms": ["bone", "osteomyelitis", "cells", "bone grafts", "recognised"],
    }
    values.update(overrides)
    return FeatureCard(**values)


def test_interpreter_prompts_match_the_paper_digests():
    assert sha256(prompts.interpreter_system_prompt(10)) == INTERPRETER_SYSTEM_DIGEST
    assert sha256(prompts.label_instruction(10)) == LABEL_INSTRUCTION_DIGEST


@pytest.mark.parametrize("method", sorted(JUDGE_SYSTEM_DIGESTS))
def test_judge_system_prompts_match_the_paper_digests(method):
    assert sha256(prompts.judge_system_prompt(method)) == JUDGE_SYSTEM_DIGESTS[method]


def test_judge_constants_match_the_paper_digests():
    assert sha256(prompts.SCORE_INSTRUCTION) == SCORE_INSTRUCTION_DIGEST
    assert sha256(prompts.USER_PROMPT) == USER_PROMPT_DIGEST
    assert sha256(prompts.OOD_NOTE) == OOD_NOTE_DIGEST


def test_judge_system_prompt_rejects_unknown_methods():
    with pytest.raises(ValueError, match="method"):
        prompts.judge_system_prompt("keywords")


def test_interpreter_user_turn_renders_byte_for_byte():
    expected = (
        "Highest activating corpus documents (2):\n"
        "  [0.29] TITLE: Bone tissue engineering | TEXT: A common view on bone tissue engineering portrays "
        "four vital aspects.\n"
        "  [0.25] untitled passage text\n"
        "\n"
        "Highest activating queries (1):\n"
        "  [0.27] What is the role of cells in these bone grafts?\n"
        "\n"
        "A few example documents that this neuron does not fire on at all:\n"
        "  TITLE: Sleep apnea | TEXT: Obstructive sleep apnea is common.\n"
        "\n"
        "Work through the steps above, briefly, then close with a JSON object and nothing after it:\n"
        '{"topic": "...", "description": "...", "keywords": ["...", "..."]}  // at most 10 keywords'
    )
    assert prompts.interpreter_user_prompt(make_card(), n_keywords=10, max_chars=1000) == expected


def test_evidence_without_documents_or_queries_says_so():
    card = make_card(top_documents=[], top_queries=[], negatives=[])
    assert prompts.render_evidence(card.top_documents, card.top_queries, card.negatives) == (
        "Highest activating corpus documents (0):\n"
        "  (none — no corpus passage was pulled for this neuron)\n"
        "\n"
        "Highest activating queries (0):\n"
        "  (none — this neuron fires on no query in the neuron profile)"
    )


def test_render_example_caps_the_body_and_spares_the_title():
    snippet = {"text": "Title body text here", "title": "Title", "activation": 0.5}
    assert prompts.render_example(snippet, max_chars=4) == "  [0.50] TITLE: Title | TEXT: body"
    assert prompts.render_example(snippet, max_chars=0) == "  [0.50] TITLE: Title"
    untitled = {"text": "abcdefgh", "title": None}
    assert prompts.render_example(untitled, max_chars=3) == "  abc"
    assert prompts.render_example(untitled, max_chars=0) == "  abcdefgh"
    # A title that does not start the text is not shown separately.
    assert prompts.render_example({"text": "other text", "title": "Title"}) == "  other text"


def test_llm_description_is_the_interpretation_string():
    card = make_card(topic="bone", description="This neuron reacts to bone.", keywords=["bone", "cells"])
    assert (
        prompts.feature_description(card, "llm")
        == "  bone: This neuron reacts to bone. (keywords: bone, cells)"
    )
    assert card.interpretation == "bone: This neuron reacts to bone. (keywords: bone, cells)"


def test_tfidf_description_joins_the_terms():
    assert prompts.feature_description(make_card(), "tfidf") == (
        "  bone, osteomyelitis, cells, bone grafts, recognised"
    )


def test_snippets_description_indents_every_line_of_the_evidence():
    card = make_card()
    block = prompts.render_evidence(card.top_documents, card.top_queries, card.negatives, max_chars=1000)
    description = prompts.feature_description(card, "snippets", max_chars=1000)
    assert description.split("\n") == ["  " + line for line in block.split("\n")]
    assert "\n  \n" in description  # blank lines become two spaces


@pytest.mark.parametrize("topic", [None, PARSE_ERROR, "LABEL_ERROR"])
def test_llm_description_needs_a_usable_label(topic):
    with pytest.raises(ValueError, match="label"):
        prompts.feature_description(make_card(topic=topic, description="raw reply"), "llm")


def test_judge_user_turn_in_domain():
    user = prompts.judge_user_prompt(2177, "  bone, osteomyelitis, cells, bone grafts, recognised", "<text>")
    assert user == (
        "Neuron 2177:\n"
        "  bone, osteomyelitis, cells, bone grafts, recognised\n"
        "\n"
        "Text:\n"
        "<text>\n"
        "\n"
        "Respond with ONLY a JSON object, no other text:\n"
        '{"score": <number between 0 and 1>}'
    )


def test_judge_user_turn_out_of_domain_adds_the_note_only():
    user = prompts.judge_user_prompt(2177, "  <description>", "<text>", out_of_domain=True)
    assert user == (
        "Neuron 2177:\n"
        "  <description>\n"
        "\n"
        "The text below is from a different corpus than the documents this description was built from, so "
        "its subject matter, vocabulary and writing style may differ. Judge whether the neuron's own concept "
        "is present, not whether the text resembles the corpus the description came from.\n"
        "\n"
        "Text:\n"
        "<text>\n"
        "\n"
        "Respond with ONLY a JSON object, no other text:\n"
        '{"score": <number between 0 and 1>}'
    )
    in_domain = prompts.judge_user_prompt(2177, "  <description>", "<text>")
    assert user.replace(prompts.OOD_NOTE, "") == in_domain


def test_cap_text_counts_the_title_and_treats_zero_as_no_cap():
    assert prompts.cap_text("abcdef", 3) == "abc"
    assert prompts.cap_text("abcdef", None) == "abcdef"
    assert prompts.cap_text("abcdef", 0) == "abcdef"
