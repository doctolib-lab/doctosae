"""Interpretation of SAE features: evidence, TF IDF terms, LLM labels, Activation Prediction and explanations.

The pipeline of Section 6 of the paper, with its defaults:

1. :meth:`Interpreter.select_features`: features firing on at least 5 evidence documents and 5 queries.
2. :meth:`Interpreter.describe`: a :class:`FeatureCard` per feature with its top documents (at most two per
   article), top queries, negatives, TF IDF terms and, with an LLM, the Interpreter label.
3. :meth:`Interpreter.score`: Activation Prediction in domain or out of domain, Pearson r and AUROC per
   method.
4. :meth:`Interpreter.explain`: which features make one passage outrank another (Figure 2).

Language models plug in through :class:`LLM`: :class:`TransformersLLM` (local, the paper's backend),
:class:`OpenAICompatibleLLM` (any OpenAI compatible server) or :class:`FakeLLM` (tests and documentation).
"""

from medsae.interpret.activations import ActivationIndex, CodedTexts, encode_sparse
from medsae.interpret.cards import FeatureCard, load_cards, save_cards
from medsae.interpret.explain import RetrievalExplanation, explain_codes
from medsae.interpret.interpreter import Interpreter
from medsae.interpret.labeling import Label, label_cards, parse_label
from medsae.interpret.llm import LLM, FakeLLM, OpenAICompatibleLLM, TransformersLLM
from medsae.interpret.scoring import EvalPool, ScoreReport, parse_score
from medsae.interpret.selection import select_features, theme_entropy
from medsae.interpret.tfidf import TfidfTerms

__all__ = [
    "LLM",
    "ActivationIndex",
    "CodedTexts",
    "EvalPool",
    "FakeLLM",
    "FeatureCard",
    "Interpreter",
    "Label",
    "OpenAICompatibleLLM",
    "RetrievalExplanation",
    "ScoreReport",
    "TfidfTerms",
    "TransformersLLM",
    "encode_sparse",
    "explain_codes",
    "label_cards",
    "load_cards",
    "parse_label",
    "parse_score",
    "save_cards",
    "select_features",
    "theme_entropy",
]
