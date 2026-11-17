"""Prompts of the Interpreter LLM and of the Activation Prediction LLM, verbatim from the paper code.

The strings are copied character for character from ``paper/src/interpretability/label_neurons.py`` and
``paper/src/interpretability/predict_neurons_binary.py`` (Appendix D.2, Figures 9 and 10). The tests pin
the SHA256 digest of every rendered prompt, so any accidental edit fails loudly.

The paper ran both LLMs through its local text generation backend: the user turn ends with an instruction to
close the reply with a JSON object, and the reply is parsed as text. The package sends the same turns to every
backend, so a hosted model reads exactly what the local model of the paper read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from doctosae.interpret.cards import FeatureCard

__all__ = [
    "ACTIVATION_GUIDELINE",
    "ACTIVATION_NOTE",
    "INTERPRETER_SYSTEM_PROMPT",
    "METHODS",
    "METHOD_DEFINITIONS",
    "OOD_NOTE",
    "SCORE_INSTRUCTION",
    "SCORE_PROMPT",
    "USER_PROMPT",
    "cap_text",
    "feature_description",
    "interpreter_system_prompt",
    "interpreter_user_prompt",
    "judge_system_prompt",
    "judge_user_prompt",
    "label_instruction",
    "render_evidence",
    "render_example",
    "split_title",
]

# Interpreter LLM (label_neurons.SYSTEM_PROMPT, verbatim, rendered with n_keywords = 10 in the paper).
INTERPRETER_SYSTEM_PROMPT = """Context: you are an investigator looking at one neuron from a medical AI \
model. Each neuron fires on certain documents and not on others. Your job is to read the \
documents it fires on, compare them with documents it does not fire on at all, and say what the \
first group has in common.

Input: 1 line = 1 document or query, with its activation value in brackets at the start of the \
line, like "[0.28]". The higher the number, the more strongly the neuron reacted to that one. \
Where a document has a title, the line reads "TITLE: ... | TEXT: ...", and the neuron saw that \
whole string, title included. Weight these by their activation value in your prediction. A few \
example documents that this neuron does not fire on at all are listed at the end, with no \
activation value shown since it did not fire on them.

Before you answer, work through these steps:
1. List what the documents and queries it reacts to have in common — topics, themes, methods, \
wordings — at more than one level of generality, from the broad subject down to the specific.
2. Use the example documents it does not fire on at all to sharpen your description of what the \
neuron means: whatever they have in common with the reacting examples is not what makes it react.
3. Settle on the least complicated account that still fits every reacting example — one theme if \
one covers them, two if the examples genuinely split, and say so plainly if nothing covers them \
all.

Output: fill in these fields.
- "topic": the broad medical topic, in 1-3 words (e.g. "cardiology").
- "description": 1-2 plain sentences on what makes this neuron react, at the level of generality \
you settled on in step 3 — one theme, or the two it splits into. Don't just repeat the topic, and \
don't summarise the documents one by one. If the documents share no single topic, say so here.
- "keywords": up to {n_keywords} words or short phrases from the documents that best describe \
this topic, most useful first.

Write everything in plain, simple language. Do not use any abbreviations or acronyms, anywhere, \
even ones the documents themselves use — spell them out. For example, write "heart attack" \
instead of "MI", and write out what a program or test does instead of naming it by its letters.

Stay accurate: simplify the words, never the medical facts, and don't add anything that isn't in \
the documents."""

DOCUMENTS_HEADER = "Highest activating corpus documents"
QUERIES_HEADER = "Highest activating queries"
NEGATIVES_HEADER = "A few example documents that this neuron does not fire on at all:"
NO_DOCUMENTS_LINE = "  (none — no corpus passage was pulled for this neuron)"
NO_QUERIES_LINE = "  (none — this neuron fires on no query in the neuron profile)"
NO_TERMS = "(no docs)"  # TF IDF description of a neuron without evidence (top_terms_for_mask)

# Activation Prediction LLM (predict_neurons_binary, verbatim).
ACTIVATION_NOTE = """
One line is one corpus document or one query, opening with that neuron's activation value on it in \
brackets, e.g. "[0.28] ...". Higher means the neuron fired more strongly on it. A corpus document \
line reads "TITLE: <title> | TEXT: <body>" where the corpus has titles. The documents listed last, \
with no activation value, are examples this neuron does not fire on at all. Your answer is a \
probability, not an activation value: do not copy the scale of the bracketed numbers above.
"""

ACTIVATION_GUIDELINE = """
- Weight the examples by their activation values: the strongest ones define what the neuron \
detects. A low value means it barely fired on that example, so treat it as marginal evidence rather \
than as a defining case."""

# Method name to (what the description is, what to match on, where it comes from): CLASS_DEFN of the paper.
METHOD_DEFINITIONS: dict[str, tuple[str, str, str]] = {
    "snippets": (
        "the examples that activate it most",
        "a neuron's examples share",
        "These are the corpus passages and queries it activates most strongly on, "
        "followed by a few example documents it does not fire on at all.",
    ),
    "tfidf": (
        "its most distinctive terms",
        "a neuron's terms point to",
        "These terms are the words that most distinguish the corpus documents it activates "
        "most strongly on from the rest of the corpus.",
    ),
    "llm": (
        "a sentence naming what it detects",
        "a neuron's description names",
        "Another model wrote this description after reading the corpus documents it activates "
        "most strongly on.",
    ),
}
METHODS = ("tfidf", "llm", "snippets")

SCORE_PROMPT = """You are a medical expert judging one neuron of a language model trained on \
medical text. The neuron is described by {class_defn}. {provenance} Only a small set of the model's \
neurons is active on any one text: estimate the probability that this neuron is one of them for \
the text below. Give a number from 0 to 1, where 1 means certain it is active, 0 means certain it \
is not, and 0.5 means you have no information either way.
{activation_note}
Guidelines:
- Match on the specific concept {class_evidence}, not just the same broad subject. The text may \
carry topics close to the neuron's without being exactly it; use your judgement.
- The neuron's concept can be a secondary theme of the text rather than its main subject.
- Do not speculate about topics that are not present in the text. Reason only from what it \
says.{activation_guideline}"""

USER_PROMPT = """Neuron {neuron}:
{description}
{domain_note}
Text:
{text}"""

OOD_NOTE = """
The text below is from a different corpus than the documents this description was built from, so its \
subject matter, vocabulary and writing style may differ. Judge whether the neuron's own concept is \
present, not whether the text resembles the corpus the description came from.
"""

SCORE_INSTRUCTION = '\n\nRespond with ONLY a JSON object, no other text:\n{"score": <number between 0 and 1>}'


def interpreter_system_prompt(n_keywords: int = 10) -> str:
    """System prompt of the Interpreter LLM for a keyword budget (10 in the paper)."""
    return INTERPRETER_SYSTEM_PROMPT.format(n_keywords=n_keywords)


def label_instruction(n_keywords: int = 10) -> str:
    """Closing instruction appended to the Interpreter user turn (``LocalLabeler._with_json_instruction``)."""
    return (
        "\n\nWork through the steps above, briefly, then close with a JSON "
        "object and nothing after it:\n"
        '{"topic": "...", "description": "...", '
        f'"keywords": ["...", "..."]}}  // at most {n_keywords} keywords'
    )


def split_title(snippet: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """``(title, body)`` when the snippet text is its title, one space, then the body; else ``(title, None)``.

    The embedded string of a titled passage is ``title + " " + body``; prompts show the two parts separately.
    """
    title = snippet.get("title")
    text = snippet["text"]
    if title and text.startswith(title + " "):
        return title, text[len(title) + 1 :]
    return title, None


def render_example(snippet: Mapping[str, Any], max_chars: int | None = None) -> str:
    """One example on one line (``label_neurons._example_line``).

    The activation is printed with two decimals when the snippet carries one. ``max_chars`` caps the body of a
    titled document (the title is spared) or the whole text otherwise, and ``0`` shows the title alone.
    """
    title, body = split_title(snippet)
    if max_chars == 0:
        shown = f"TITLE: {title}" if body is not None else snippet["text"]
    elif body is not None:
        shown = f"TITLE: {title} | TEXT: {body if max_chars is None else body[:max_chars]}"
    else:
        shown = snippet["text"] if max_chars is None else snippet["text"][:max_chars]
    activation = snippet.get("activation")
    prefix = f"[{activation:.2f}] " if activation is not None else ""
    return f"  {prefix}{shown}"


def render_evidence(
    documents: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    negatives: Sequence[Mapping[str, Any]] = (),
    max_chars: int | None = None,
    max_documents: int | None = None,
    max_queries: int | None = None,
    show_negatives: bool = True,
) -> str:
    """The evidence block of a feature, strongest first (``label_neurons.build_prompt``).

    This is the Interpreter user turn (before its closing instruction) and, indented, the ``snippets``
    description shown to the Activation Prediction LLM.
    """
    documents = list(documents)[:max_documents]
    queries = list(queries)[:max_queries]
    lines = [f"{DOCUMENTS_HEADER} ({len(documents)}):"]
    if documents:
        lines += [render_example(snippet, max_chars) for snippet in documents]
    else:
        lines.append(NO_DOCUMENTS_LINE)
    lines.append("")
    lines.append(f"{QUERIES_HEADER} ({len(queries)}):")
    if queries:
        lines += [render_example(snippet, max_chars) for snippet in queries]
    else:
        lines.append(NO_QUERIES_LINE)
    if show_negatives and negatives:
        lines += ["", NEGATIVES_HEADER]
        lines += [render_example(snippet, max_chars) for snippet in negatives]
    return "\n".join(lines)


def interpreter_user_prompt(card: FeatureCard, n_keywords: int = 10, max_chars: int | None = 1000) -> str:
    """Full Interpreter user turn of a feature: its evidence block, then the JSON instruction."""
    evidence = render_evidence(card.top_documents, card.top_queries, card.negatives, max_chars=max_chars)
    return evidence + label_instruction(n_keywords)


def judge_system_prompt(method: str) -> str:
    """System prompt of the Activation Prediction LLM for one interpretation method."""
    if method not in METHOD_DEFINITIONS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    definition, evidence, provenance = METHOD_DEFINITIONS[method]
    shows_activations = method == "snippets"
    return SCORE_PROMPT.format(
        class_defn=definition,
        class_evidence=evidence,
        provenance=provenance,
        activation_note=ACTIVATION_NOTE if shows_activations else "",
        activation_guideline=ACTIVATION_GUIDELINE.rstrip("\n") if shows_activations else "",
    )


def judge_user_prompt(feature: int, description: str, text: str, out_of_domain: bool = False) -> str:
    """User turn for one (feature, text) pair, closed by the JSON score instruction.

    Out of domain texts get :data:`OOD_NOTE`; it is the only difference between the two settings.
    """
    user = USER_PROMPT.format(
        neuron=feature, description=description, text=text, domain_note=OOD_NOTE if out_of_domain else ""
    )
    return user + SCORE_INSTRUCTION


def cap_text(text: str, max_chars: int | None) -> str:
    """The judged text cut to ``max_chars`` characters, title included; ``None`` or ``0`` keeps it whole."""
    return text if not max_chars else text[:max_chars]


def feature_description(card: FeatureCard, method: str, max_chars: int | None = 1000) -> str:
    """How the Activation Prediction LLM sees a feature (``build_class_description``).

    * ``tfidf``: its TF IDF terms, joined by commas; ``(no docs)`` for a feature without evidence, as the
      paper's ``top_terms_for_mask`` writes it.
    * ``llm``: ``topic: description (keywords: ...)``, the Interpreter label.
    * ``snippets``: its evidence block, each line indented by two more spaces.

    Raises ``ValueError`` for ``llm`` when the card has no usable label; the caller then skips the feature.
    """
    if method == "snippets":
        block = render_evidence(card.top_documents, card.top_queries, card.negatives, max_chars=max_chars)
        return "\n".join("  " + line for line in block.split("\n"))
    if method == "tfidf":
        return "  " + (", ".join(card.tfidf_terms) or NO_TERMS)
    if method == "llm":
        if not card.has_label:
            raise ValueError(f"feature {card.feature} has no usable Interpreter label (topic {card.topic!r})")
        return "  " + card.interpretation
    raise ValueError(f"method must be one of {METHODS}, got {method!r}")
