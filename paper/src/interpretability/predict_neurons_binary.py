"""Predict whether ONE SAE neuron fires on a text, one call per (text, neuron) pair.

Each call shows one neuron's description and one text, and asks for the probability in [0, 1] that
the neuron is active on it.

Usage:
    uv run python -m src.interpretability.predict_neurons_binary \
        --neuron-profiles cure_k64_neuron_profiles.json --description-type tfidf \
        --eval-rows cure_k64_eval_rows.json \
        --llm-backend local --device cuda:0 --output preds_binary_tfidf.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from src.interpretability.label_neurons import DEFAULT_MODEL, build_prompt
from src.interpretability.llm_backend import LocalLLM, build_llm

ACTIVATION_NOTE = """
One line is one corpus document or one query, opening with that neuron's activation value on it in \
brackets, e.g. "[0.28] ...". Higher means the neuron fired more strongly on it. A corpus document \
line reads "TITLE: <title> | TEXT: <body>" where the corpus has titles. The documents listed last, \
with no activation value, are examples this neuron does not fire on at all. Your answer is a \
probability, not an activation value: do not copy the scale of the bracketed numbers above.
"""

# Shown with ACTIVATION_NOTE. Comparable only within one neuron.
ACTIVATION_GUIDELINE = """
- Weight the examples by their activation values: the strongest ones define what the neuron \
detects. A low value means it barely fired on that example, so treat it as marginal evidence rather \
than as a defining case."""

# What each description is made of and where it came from, per --description-type.
CLASS_DEFN = {
    "snippets": ("the examples that activate it most", "a neuron's examples share",
                 "These are the corpus passages and queries it activates most strongly on, "
                 "followed by a few example documents it does not fire on at all."),
    "tfidf": ("its most distinctive terms", "a neuron's terms point to",
              "These terms are the words that most distinguish the corpus documents it activates "
              "most strongly on from the rest of the corpus."),
    "llm": ("a sentence naming what it detects", "a neuron's description names",
            "Another model wrote this description after reading the corpus documents it activates "
            "most strongly on."),
}

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

# Shown for a row from another corpus.
OOD_NOTE = """
The text below is from a different corpus than the documents this description was built from, so its \
subject matter, vocabulary and writing style may differ. Judge whether the neuron's own concept is \
present, not whether the text resembles the corpus the description came from.
"""


class NeuronScore(BaseModel):
    """One (text, neuron) probability, as returned by the LLM. The decision threshold sits at 0.5."""

    score: float = Field(ge=0, le=1, description="Probability the neuron is active on the text, "
                                                 "from 0 (certain it is not) to 1 (certain it is); "
                                                 "0.5 is no information")


def build_class_description(neuron: dict, description_type: str,
                            label_row: dict | None = None, max_chars: int | None = None,
                            max_corpus: int | None = None, max_queries: int | None = None) -> str:
    """One neuron's description. `neuron`: an entry of a profiles file's "neurons" list.
    `label_row`: its entry in a labels file, required for description_type "llm"."""
    if description_type == "snippets":
        # label_neurons.py's formatter keeps activations and the passage/query split.
        block = build_prompt(neuron, max_chars=max_chars, max_corpus=max_corpus,
                            max_queries=max_queries, show_negatives=True)
        return "\n".join("  " + line for line in block.split("\n"))
    if description_type == "tfidf":
        if "top_terms" not in neuron:
            raise ValueError(f"neuron {neuron['neuron']} has no 'top_terms' — re-run "
                             f"select_neurons.py, which records them per neuron")
        return "  " + neuron["top_terms"]
    if description_type == "llm":
        if label_row is None:
            raise ValueError(f"no label for neuron {neuron['neuron']} in --labels")
        if label_row["topic"] in ("LABEL_ERROR", "PARSE_ERROR"):
            raise ValueError(f"neuron {neuron['neuron']}'s label is a {label_row['topic']} "
                             f"({label_row['description']}) -- not a real class description")
        kw = ", ".join(label_row["keywords"])
        return f"  {label_row['topic']}: {label_row['description']} (keywords: {kw})"
    raise ValueError(f"--description-type must be snippets/tfidf/llm, got {description_type!r}")


def build_system_prompt(description_type: str) -> str:
    """The single-neuron system prompt, worded for what description_type shows."""
    defn, evidence, provenance = CLASS_DEFN[description_type]
    shows_activations = description_type == "snippets"
    return SCORE_PROMPT.format(
        class_defn=defn, class_evidence=evidence, provenance=provenance,
        activation_note=ACTIVATION_NOTE if shows_activations else "",
        activation_guideline=ACTIVATION_GUIDELINE.rstrip("\n") if shows_activations else "")


def _parse_json(text: str):
    """(NeuronScore, the score before clamping) from the last brace group, or (None, None) if the
    reply carried no usable JSON."""
    matches = re.findall(r"\{[^{}]*\}", text, re.DOTALL)
    if not matches:
        return None, None
    try:
        raw = float(json.loads(matches[-1])["score"])
        return NeuronScore(score=max(0.0, min(1.0, raw))), raw
    except Exception:
        return None, None


SCORE_INSTRUCTION = ('\n\nRespond with ONLY a JSON object, no other text:\n'
                     '{"score": <number between 0 and 1>}')


def predict_batch(llm, system: str, users: list[str], max_workers: int = 1):
    """One (verdict, pre-clamp score, raw reply) triple per user prompt: verdict is None only for
    LocalLLM, pre-clamp score and raw reply None for GatewayLLM. LocalLLM runs the batch in one
    forward pass; a gateway is one HTTP request per prompt across max_workers threads."""
    if isinstance(llm, LocalLLM):
        replies = llm.generate_batch(system, users)
        return [(*_parse_json(t), t) for t in replies]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return [(v, None, None) for v in ex.map(lambda u: llm.generate(system, u), users)]


def _cap(text: str, max_chars: int | None) -> str:
    """The judged text, truncated to max_chars when one is set."""
    return text if not max_chars else text[:max_chars]


def build_user(local: bool, neuron: int, description: str, text: str, ood: bool = False) -> str:
    """The user turn for one (text, neuron) pair, plus the JSON instruction a local model needs.
    `ood` states in the prompt that the text is from another corpus than the description."""
    user = USER_PROMPT.format(neuron=neuron, description=description, text=text,
                             domain_note=OOD_NOTE if ood else "")
    return user + SCORE_INSTRUCTION if local else user


def sample_pairs(rows: list[dict], neurons: list[int], n_pos: int, n_neg: int, seed: int,
                thresholds: dict | None = None) -> list[tuple[int, int, bool]]:
    """(neuron, row index, is_positive) pairs to score: positive if the row lists the neuron in
    "positive_for", negative if the neuron does not fire on it. thresholds is the fallback."""
    rng = random.Random(seed)
    pairs = []
    for n in neurons:
        def strong(r):
            if "positive_for" in r:
                return n in r["positive_for"]
            a = r["activations"].get(str(n))
            return a is not None and a >= (thresholds or {}).get(str(n), 0.0)
        pos = [i for i, r in enumerate(rows) if strong(r)]
        neg = [i for i, r in enumerate(rows) if str(n) not in r["activations"]]
        if len(pos) < n_pos:
            print(f"  neuron {n}: only {len(pos)} rows in its positive band (< {n_pos})")
        pairs += [(n, i, True) for i in rng.sample(pos, min(n_pos, len(pos)))]
        pairs += [(n, i, False) for i in rng.sample(neg, min(n_neg, len(neg)))]
    return pairs


def derange(neurons: list[int], seed: int) -> dict[int, int]:
    """Map every neuron to another neuron, never to itself."""
    if len(neurons) < 2:
        raise ValueError("--shuffle-descriptions needs at least two scored neurons")
    rng = random.Random(seed)
    perm = list(neurons)
    while any(a == b for a, b in zip(neurons, perm)):
        rng.shuffle(perm)
    return dict(zip(neurons, perm))


def _counts(sub: list[dict]) -> dict:
    """Positive count and scored-pair count over a subset of scored pairs."""
    return {"support": sum(1 for p in sub if p["ground_truth"]), "n_pairs": len(sub)}


def _threshold_free(sub: list[dict]) -> dict:
    """Score correlated with the +1/-1 ground truth."""
    s = np.array([p["score"] for p in sub])
    y = np.array([1.0 if p["ground_truth"] else -1.0 for p in sub])
    return {"pearson": float(np.corrcoef(s, y)[0, 1]) if s.std() else 0.0}


def score_pairs(pairs: list[dict], neurons: list[int]) -> dict:
    """Per-neuron and micro counts over scored (neuron, row) pairs, plus the threshold-free metrics.
    Rows with parse_error are excluded.
    """
    pairs = [p for p in pairs if not p.get("parse_error")]
    out = {str(n): _counts([p for p in pairs if p["neuron"] == n]) for n in neurons}
    out["micro avg"] = _counts(pairs)
    out["micro avg"].update(_threshold_free(pairs))
    for n in neurons:
        sub = [p for p in pairs if p["neuron"] == n]
        if sub:
            out[str(n)].update(_threshold_free(sub))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--neuron-profiles", required=True, type=Path, help="select_neurons.py's output JSON")
    p.add_argument("--labels", type=Path, default=None, help="label_neurons.py's output JSON, "
                                                              "required for --description-type llm")
    p.add_argument("--description-type", required=True, choices=["snippets", "tfidf", "llm"])
    p.add_argument("--eval-rows", type=Path, default=None, help="build_interpret_eval.py's output JSON")
    p.add_argument("--docs", nargs="+", default=None,
                   help="Literal text strings instead of --eval-rows -- prediction only, no scoring")
    p.add_argument("--positives-per-neuron", type=int, default=3,
                   help="--eval-rows only: rows the neuron fires on, per neuron (default 3)")
    p.add_argument("--negatives-per-neuron", type=int, default=3,
                   help="--eval-rows only: rows it does not fire on, per neuron (default 3)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle-descriptions", action="store_true",
                   help="Chance-level control: show each pair another neuron's description, keeping "
                        "the rows and labels of the neuron it was sampled for")
    p.add_argument("--max-example-chars", type=int, default=1000,
                   help="Cap the judged text, and each shown example's body for --description-type "
                        "snippets (default 1000, 0 = titles only)")
    p.add_argument("--max-corpus-examples", type=int, default=None,
                   help="--description-type snippets: show at most this many corpus documents")
    p.add_argument("--max-query-examples", type=int, default=None,
                   help="--description-type snippets: show at most this many queries")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Prompts per chunk. --llm-backend local: prompts per forward pass. gateway: "
                        "max concurrent requests is min(this, --max-workers), so raise this too to "
                        "get concurrency (default 1)")
    p.add_argument("--max-workers", type=int, default=8,
                   help="--llm-backend gateway: concurrent HTTP requests within a --batch-size "
                        "chunk (default 8). No effect on --llm-backend local, which always runs "
                        "its whole chunk in one forward pass")
    p.add_argument("--llm-backend", choices=["gateway", "local"], default=None)
    p.add_argument("--llm-model", default=DEFAULT_MODEL)
    p.add_argument("--llm-base-url", default=None)
    p.add_argument("--llm-api-key", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    if bool(args.eval_rows) == bool(args.docs):
        p.error("exactly one of --eval-rows or --docs is required")
    if args.description_type == "llm" and not args.labels:
        p.error("--description-type llm requires --labels")

    profiles = json.load(open(args.neuron_profiles))
    neurons_meta = profiles["neurons"]
    neurons = [n["neuron"] for n in neurons_meta]
    label_rows = ({r["neuron"]: r for r in json.load(open(args.labels))["neurons"]}
                  if args.labels else None)
    # A neuron whose label failed to parse has no description and is dropped.
    descriptions = {}
    for n in neurons_meta:
        try:
            descriptions[n["neuron"]] = build_class_description(
                n, args.description_type, label_rows.get(n["neuron"]) if label_rows else None,
                args.max_example_chars, args.max_corpus_examples, args.max_query_examples)
        except ValueError as e:
            print(f"  no description for neuron {n['neuron']}: {e}")
    system = build_system_prompt(args.description_type)
    print(f"{len(neurons)} neurons, from {args.neuron_profiles}")

    if args.docs:
        texts = [_cap(t, args.max_example_chars) for t in args.docs]
        truth, row_labels = None, list(range(len(args.docs)))
        act, sources, ranks = [{} for _ in texts], [None] * len(texts), [{} for _ in texts]
        ood_sources = set()
    else:
        eval_data = json.load(open(args.eval_rows))
        # The eval rows decide which neurons are scorable; score the overlap with the profiles.
        # An empty overlap means the two files are from different checkpoints.
        n_profiled = len(neurons)
        neurons = [n for n in eval_data["neurons"] if n in descriptions]
        if not neurons:
            p.error(f"no neuron is in both --eval-rows ({len(eval_data['neurons'])}) and "
                    f"--neuron-profiles ({n_profiled}) — are they from the same checkpoint?")
        skipped = len(eval_data["neurons"]) - len(neurons)
        print(f"scoring {len(neurons)}/{n_profiled} profiled neurons"
              + (f" ({skipped} eval neurons have no description here)" if skipped else ""))
        texts = [_cap(r["text"], args.max_example_chars) for r in eval_data["rows"]]
        act = [r["activations"] for r in eval_data["rows"]]
        sources = [r.get("source") for r in eval_data["rows"]]
        ranks = [r.get("ranks", {}) for r in eval_data["rows"]]
        ood_sources = set(eval_data.get("ood_sources") or [])
        n_ood = sum(1 for s_ in sources if s_ in ood_sources)
        if n_ood:
            print(f"{n_ood} of {len(texts)} rows are from another corpus than the descriptions "
                  f"({', '.join(sorted(ood_sources))}) -- their prompt says so")
        truth = act  # presence of a scored pair's neuron is decided per pair, see sample_pairs
        row_labels = [r["row"] for r in eval_data["rows"]]
        print(f"{len(texts)} eval rows from {args.eval_rows}")

    shown = derange(neurons, args.seed) if args.shuffle_descriptions else {n: n for n in neurons}
    pairs = ([(n, i, None) for n in neurons for i in range(len(texts))] if truth is None else
             sample_pairs(eval_data["rows"], neurons, args.positives_per_neuron,
                          args.negatives_per_neuron, args.seed, eval_data.get("band_thresholds")))
    if truth is not None:
        print(f"{len(pairs)} (neuron, row) pairs -> {len(pairs)} calls")

    if args.dry_run:
        print(f"\n########## SYSTEM ##########\n{system}")
        user = build_user(args.llm_backend == "local", shown[neurons[0]],
                          descriptions[shown[neurons[0]]], texts[0], bool(ood_sources))
        print(f"\n########## row {row_labels[0]} x neuron {neurons[0]} ##########\n{user}")
        if truth is not None:
            print(f"\nground truth for this row: {truth[0]}")
        return

    if not args.llm_backend:
        p.error("--llm-backend is required (choose 'local' or 'gateway'), unless --dry-run")
    try:
        llm = build_llm(args.llm_backend, args.llm_model, NeuronScore, args.device, args.dtype,
                        args.llm_base_url, args.llm_api_key)
    except ValueError as e:
        p.error(str(e))

    scored, n_parse_error = [], 0
    for start in range(0, len(pairs), args.batch_size):
        chunk = pairs[start:start + args.batch_size]
        verdicts = predict_batch(llm, system,
                                 [build_user(isinstance(llm, LocalLLM), shown[n],
                                             descriptions[shown[n]], texts[i],
                                             sources[i] in ood_sources)
                                  for n, i, _ in chunk], args.max_workers)
        for (n, i, is_pos), (v, unclamped, raw) in zip(chunk, verdicts):
            if v is None:
                n_parse_error += 1
            rec = {"neuron": n, "row": row_labels[i], "source": sources[i], "text": texts[i][:300],
                  "ground_truth": is_pos, "activation": act[i].get(str(n)),
                  "rank": ranks[i].get(str(n)), "parse_error": v is None}
            if args.shuffle_descriptions:
                rec["description_neuron"] = shown[n]
            if v is None and raw is not None:
                rec["raw_reply"] = raw
            # Only when the reply fell outside [0, 1] and was clamped.
            if v is not None and unclamped is not None and unclamped != v.score:
                rec["score_raw"] = unclamped
            # fires is the score thresholded at 0.5.
            rec.update(score=float(v.score) if v else 0.5, fires=bool(v and v.score > 0.5))
            scored.append(rec)
        print(f"  {len(scored)}/{len(pairs)} calls")

    result = {"neuron_profiles_file": str(args.neuron_profiles),
             "labels_file": str(args.labels) if args.labels else None,
             "description_type": args.description_type,
             "eval_rows_file": str(args.eval_rows) if args.eval_rows else None,
             "max_example_chars": args.max_example_chars,
             "max_corpus_examples": args.max_corpus_examples,
             "max_query_examples": args.max_query_examples,
             "llm_backend": args.llm_backend, "llm_model": args.llm_model,
             "batch_size": args.batch_size,
             "positives_per_neuron": args.positives_per_neuron,
             "negatives_per_neuron": args.negatives_per_neuron, "seed": args.seed,
             "shuffle_descriptions": args.shuffle_descriptions,
             "n_neurons": len(neurons), "n_rows": len(texts),
             "n_calls": len(scored), "n_parse_error": n_parse_error, "predictions": scored}
    if truth is not None:
        result["metrics"] = score_pairs(scored, neurons)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.output, "w"), indent=2)
    print(f"wrote {args.output} ({len(scored)} calls, {n_parse_error} parse errors)")
    if truth is not None:
        mi = result["metrics"]["micro avg"]
        print(f"{mi['support']} positives of {mi['n_pairs']} scored pairs")
        if "pearson" in mi:
            print(f"  pearson={mi['pearson']:.3f}")


if __name__ == "__main__":
    main()
