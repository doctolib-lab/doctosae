"""Build the cases file of the annotation app from interpretability outputs.

Give exactly one input:

* medsae package outputs: `--cards` and `--scores`, the `cards.json` and `scores.json` that
  `medsae interpret --llm ... --score` writes (the scores must include the Interpreter LLM method, `llm`);
* paper code outputs of `scripts/run_interpretability_pipeline.sh`: `--labels` (`labels.json`) and
  `--predictions` (`pred_binary_llm.json`), both in `RUN/interpretation/`;
* a Markdown review file (`--from-markdown`), such as one written earlier with `--markdown`.

With the first two, features are ranked by the Pearson r between the judge's probability and the ground truth,
the top features get one judged passage each (positive or negative at random), and a seeded draw balanced
between positive and negative passages becomes the cases, so every annotator labels the same cases. Plain JSON
in, plain JSON out: neither streamlit nor medsae is imported.

    python scripts/build_cases.py --cards data/features/cards.json --scores data/features/scores.json
    python scripts/build_cases.py --labels labels.json --predictions pred_binary_llm.json --markdown data/top60.md
    python scripts/build_cases.py --from-markdown data/top60.md
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the app folder holds the case schema
from cases import (
    CASE_KEYS,
    EXTRA_KEYS,
    PASSAGE_MAX_CHARS,
    SAMPLE_SEED,
    CasesError,
    judge_input,
    validate_cases,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "cases.json"
# predict_neurons_binary.py and the medsae ScoreReport both store only the first 300 characters of a text
STORED_TEXT_CHARS = 300
FAILED_LABELS = ("LABEL_ERROR", "PARSE_ERROR")  # topics the paper writes when the Interpreter LLM failed

# Markdown review file: one block per feature, blocks separated by Markdown horizontal rules.
_BLOCK = re.compile(
    r"^### (?P<rank>\d+)\. Feature (?P<feature>\d+)\s*$\n+"
    r"\*\*(?:LLM )?[Dd]escription\.\*\* (?P<description>.+?)\s*\n+"
    r"\*\*(?P<label>Positive|Negative)\*\*\s*\n+"
    r"> (?P<passage>.+?)\s*$",
    re.MULTILINE | re.DOTALL,
)
# "topic: description (keywords: a, b, c)", the exact string the judge LLM is shown
_JUDGE_INPUT = re.compile(
    r"^(?P<topic>[^:]+): (?P<text>.+?)(?: \(keywords: (?P<keywords>[^)]*)\))?$", re.DOTALL)


class BuildError(Exception):
    """Inputs that cannot give a valid cases file; the message says what to change."""


@dataclass
class Inputs:
    """Descriptions, judged pairs and stored per feature r, whichever pipeline wrote them."""

    descriptions: dict[int, dict]  # feature id to {"topic", "description", "keywords"}
    pairs: list[dict]  # {"feature", "row", "text", "ground_truth", "probability", "parse_error", "source"}
    stored_r: dict[int, Any]  # per feature Pearson r saved by the scorer, when the file has it
    dataset: str | None = None


# ── Markdown review files ────────────────────────────────────────────


def split_description(raw: str) -> dict:
    """Topic, description and keywords back from a judge input string, the only form a review file stores.

    The topic is everything before the first colon. A keyword holding ")" breaks the keywords group, which
    then stays inside the description.
    """
    m = _JUDGE_INPUT.match(raw)
    if not m:
        return {"topic": "", "description": raw, "keywords": []}
    keywords = [k.strip() for k in (m["keywords"] or "").split(",") if k.strip()]
    return {"topic": m["topic"].strip(), "description": m["text"].strip(), "keywords": keywords}


def parse_markdown(text: str) -> list[dict]:
    """Features of a review file in file order; rank and ground truth come from the file itself."""
    features = []
    for block in text.split("\n---\n"):
        m = _BLOCK.search(block.strip())
        if not m:
            continue
        features.append({
            "rank": int(m["rank"]),
            "feature_id": int(m["feature"]),
            "judge_input": m["description"].strip(),
            **split_description(m["description"].strip()),
            "passage": m["passage"].strip(),
            "ground_truth": 1 if m["label"] == "Positive" else 0,
        })
    return features


def markdown_text(features: Sequence[dict], dataset: str | None, max_chars: int) -> str:
    """The ranked features as a review file that `--from-markdown` reads back."""
    where = f", {dataset}" if dataset else ""
    header = (
        f"# Top {len(features)} SAE features by r (Interpreter LLM description{where})\n\n"
        "r: Pearson correlation between the judge's predicted probability and the ground truth, per feature. "
        "Description is the exact string the Activation Prediction LLM is shown. One passage per feature, "
        f"drawn at random as a positive or a negative. Passages capped at {max_chars} characters."
    )
    blocks = [
        f"### {f['rank']}. Feature {f['feature_id']}\n\n**Description.** {f['judge_input']}\n\n"
        f"**{'Positive' if f['ground_truth'] else 'Negative'}**\n\n> {f['passage']}"
        for f in sorted(features, key=lambda f: f["rank"])
    ]
    return "\n\n---\n\n".join([header, *blocks]) + "\n"


# ── medsae and paper outputs ─────────────────────────────────────────


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"cannot read {path}: {exc}") from exc


def _convert(records: list, convert: Callable[[dict], Any], path: Path) -> list:
    try:
        return [convert(r) for r in records]
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise BuildError(f"{path}: unexpected record format ({type(exc).__name__}: {exc})") from exc


def _feature_id(record: dict) -> int:
    """medsae names the feature id `feature`, the paper code `neuron`."""
    return int(record["feature"] if "feature" in record else record["neuron"])


def _description(record: dict) -> tuple[int, dict]:
    return _feature_id(record), {
        "topic": record.get("topic"),
        "description": record.get("description"),
        "keywords": record.get("keywords") or [],
    }


def _pair(record: dict) -> dict:
    # medsae records a failed request as an `error` record; like an unparsed reply, it carries no judgment
    parse_error = bool(record.get("parse_error")) or bool(record.get("error"))
    return {
        "feature": _feature_id(record),
        "row": record.get("row"),
        "text": str(record.get("text") or ""),
        "ground_truth": bool(record["ground_truth"]),
        # 0.5 is recorded for a reply that could not be parsed or a failed request; such pairs are dropped anyway
        "probability": math.nan if parse_error else float(record["score"]),
        "parse_error": parse_error,
        "source": record.get("source"),
    }


def _stored_r(per_feature: Any) -> dict[int, Any]:
    """Per feature r saved next to the pairs; "micro avg" and entries without a pearson are skipped."""
    if not isinstance(per_feature, dict):
        return {}
    return {int(key): value["pearson"] for key, value in per_feature.items()
            if str(key).isdigit() and isinstance(value, dict) and value.get("pearson") is not None}


def _read_scores(path: Path) -> tuple[list[dict], dict[int, Any], str | None]:
    """Judged pairs, stored per feature r and dataset name of a scores file, after the step 1 checks.

    Two layouts: the `scores.json` of `medsae interpret --score` (`ScoreReport.to_dict()`, every scored
    method under its name) and the `pred_binary_<type>.json` of the paper code (one description type each).
    """
    data = _read_json(path)
    predictions = data.get("predictions") if isinstance(data, dict) else None
    if isinstance(predictions, dict):  # medsae ScoreReport
        if "llm" not in predictions:
            raise BuildError(f"{path} has no scores of Interpreter LLM descriptions (methods: "
                             f"{', '.join(map(str, predictions))}); score with the 'llm' method")
        records = predictions["llm"]
        per_feature = data.get("per_feature")
        stored = per_feature.get("llm") if isinstance(per_feature, dict) else None
    elif isinstance(predictions, list):  # paper code
        if data.get("description_type") != "llm":
            raise BuildError(f"{path} scores {data.get('description_type')!r} descriptions; the app shows "
                             "Interpreter LLM descriptions, so use pred_binary_llm.json")
        records, stored = predictions, data.get("metrics")
    else:
        raise BuildError(f"{path} has no 'predictions'; expected the scores.json of medsae interpret or the "
                         "pred_binary_llm.json of the paper code")
    if data.get("shuffle_descriptions"):
        raise BuildError(f"{path} is the shuffled control, where each feature was judged with another "
                         "feature's description; use the unshuffled scores")
    if not isinstance(records, list):
        raise BuildError(f"{path}: the Interpreter LLM predictions are not a list of judged pairs")
    return _convert(records, _pair, path), _stored_r(stored), data.get("dataset")


def load_medsae(cards_path: Path, scores_path: Path) -> Inputs:
    """`cards.json` and `scores.json` written by `medsae interpret` (a plain list of cards works too)."""
    cards = _read_json(cards_path)
    if isinstance(cards, dict):
        cards = cards.get("cards")
    if not isinstance(cards, list):
        raise BuildError(f"{cards_path} must hold a list of feature cards, or an object with a 'cards' list")
    pairs, stored_r, dataset = _read_scores(scores_path)
    descriptions = dict(_convert(cards, _description, cards_path))
    return Inputs(descriptions, pairs, stored_r, dataset=dataset)


def load_paper(labels_path: Path, predictions_path: Path) -> Inputs:
    """`labels.json` of label_neurons.py and `pred_binary_llm.json` of predict_neurons_binary.py."""
    labels = _read_json(labels_path)
    if not isinstance(labels, dict) or not isinstance(labels.get("neurons"), list):
        raise BuildError(f"{labels_path} has no 'neurons' list; expected the labels.json of label_neurons.py")
    pairs, stored_r, dataset = _read_scores(predictions_path)
    descriptions = dict(_convert(labels["neurons"], _description, labels_path))
    return Inputs(descriptions, pairs, stored_r, dataset=dataset)


def pearson_r(pairs: Sequence[dict]) -> float:
    """Point biserial r computed like the paper's `_threshold_free`: labels coded +1 and −1, and 0.0 when
    every probability is the same."""
    s = np.array([p["probability"] for p in pairs])
    y = np.array([1.0 if p["ground_truth"] else -1.0 for p in pairs])
    return float(np.corrcoef(s, y)[0, 1]) if s.std() else 0.0


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _usable(description: dict) -> bool:
    """The paper drops features whose label failed; an empty description leaves the app nothing to show."""
    topic, text, keywords = description["topic"], description["description"], description["keywords"]
    return (isinstance(topic, str) and topic not in FAILED_LABELS
            and isinstance(text, str) and bool(text.strip())
            and isinstance(keywords, list) and all(isinstance(k, str) for k in keywords))


def rank_features(inputs: Inputs, top: int) -> list[dict]:
    """Steps 2 to 5: usable descriptions, clean pairs, r per feature, then the `top` features by descending r
    (ties broken by feature id)."""
    described = {f: d for f, d in inputs.descriptions.items() if _usable(d)}
    by_feature: dict[int, list[dict]] = {}
    for pair in inputs.pairs:
        if not pair["parse_error"] and pair["feature"] in described and pair["text"].strip():
            by_feature.setdefault(pair["feature"], []).append(pair)
    ranked = []
    for feature, pairs in by_feature.items():
        if {p["ground_truth"] for p in pairs} != {True, False}:  # one label only: r is undefined
            continue
        r = _as_float(inputs.stored_r[feature]) if feature in inputs.stored_r else pearson_r(pairs)
        if math.isfinite(r):
            ranked.append({"feature": feature, "r": r, "pairs": pairs, **described[feature]})
    ranked.sort(key=lambda f: (-f["r"], f["feature"]))
    return ranked[:top]


def pick_passages(ranked: Sequence[dict], seed: int, max_chars: int, text_cap: int | None) -> list[dict]:
    """Step 6: one judged passage per feature in rank order, positive or negative at random."""
    rng = random.Random(seed)
    features = []
    for rank, feature in enumerate(ranked, start=1):
        positive = rng.random() < 0.5
        # rank_features keeps only features with both labels, so the pool is never empty
        pool = sorted((p for p in feature["pairs"] if p["ground_truth"] == positive),
                      key=lambda p: (str(p["row"]), p["text"]))
        pair = rng.choice(pool)
        text = pair["text"]
        features.append({
            "rank": rank,
            "feature_id": feature["feature"],
            "judge_input": judge_input(feature["topic"], feature["description"], feature["keywords"]),
            "topic": feature["topic"],
            "description": feature["description"],
            "keywords": list(feature["keywords"]),
            # whitespace collapsed: the passage must stay one line in the HTML card and in the review file
            "passage": " ".join(text[:max_chars].split()),
            "ground_truth": int(positive),
            # a text already cut by the scorer may hide a longer original
            "truncated": len(text) > max_chars or (text_cap is not None and len(text) >= text_cap),
            "feature_r": feature["r"],
            "judge_probability": pair["probability"],
            "source": pair["source"],
        })
    return features


# ── Balanced draw ────────────────────────────────────────────────────


def case_id_width(n_cases: int) -> int:
    return max(2, len(str(n_cases)))


def _case(feature: dict, case_id: str) -> dict:
    """Keys in the order of the study file, then the extras that are known."""
    case = {key: feature[key] for key in CASE_KEYS if key != "case_id"}
    case["case_id"] = case_id
    case.update({key: feature[key] for key in EXTRA_KEYS if feature.get(key) is not None})
    return case


def sample_cases(features: Sequence[dict], n_cases: int, seed: int) -> list[dict]:
    """Steps 7 and 8, the balanced draw: `n_cases // 2` positives then the other negatives from a fresh
    `random.Random(seed)`, each list in rank order, and the chosen cases numbered in rank order."""
    by_rank = sorted(features, key=lambda f: f["rank"])
    positives = [f for f in by_rank if f["ground_truth"] == 1]
    negatives = [f for f in by_rank if f["ground_truth"] == 0]
    half = n_cases // 2
    if len(positives) < half or len(negatives) < n_cases - half:
        raise BuildError(
            f"{n_cases} balanced cases need {half} positive and {n_cases - half} negative passages, but the "
            f"features give {len(positives)} and {len(negatives)}")
    rng = random.Random(seed)
    chosen = rng.sample(positives, half) + rng.sample(negatives, n_cases - half)
    chosen.sort(key=lambda f: f["rank"])
    width = case_id_width(n_cases)
    return [_case(f, f"C{i:0{width}d}") for i, f in enumerate(chosen, start=1)]


# ── Command line ─────────────────────────────────────────────────────


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {text}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    medsae = parser.add_argument_group("medsae package outputs")
    medsae.add_argument("--cards", type=Path,
                        help="JSON list of FeatureCard.to_dict(), or an object with a 'cards' list")
    medsae.add_argument("--scores", type=Path,
                        help="Activation Prediction report for the 'llm' descriptions, with per pair records")
    paper = parser.add_argument_group("paper code outputs")
    paper.add_argument("--labels", type=Path, help="labels.json written by label_neurons.py")
    paper.add_argument("--predictions", type=Path,
                       help="pred_binary_llm.json written by predict_neurons_binary.py")
    review = parser.add_argument_group("Markdown review file")
    review.add_argument("--from-markdown", type=Path, metavar="MD",
                        help="review file in the --markdown format; rank and ground truth are read from it, "
                        "so the same file and seed give the same cases")
    parser.add_argument("--top", type=_positive, default=60,
                        help="features kept by r before the draw, medsae and paper inputs only (default 60)")
    parser.add_argument("--n-cases", type=_positive, default=30,
                        help="cases to draw, half of them positive (default 30)")
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED,
                        help=f"seed of the passage pick and of the draw (default {SAMPLE_SEED})")
    parser.add_argument("--max-chars", type=_positive, default=PASSAGE_MAX_CHARS,
                        help=f"characters of each passage shown to clinicians (default {PASSAGE_MAX_CHARS})")
    parser.add_argument("--dataset", help="dataset name for the review file title (default: the 'dataset' "
                        "field of the scores file, if any)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="cases file to write (default data/cases.json, which the app picks up)")
    parser.add_argument("--markdown", type=Path, metavar="PATH",
                        help="also write the ranked features as a Markdown review file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    modes = {
        "medsae": (args.cards, args.scores),
        "paper": (args.labels, args.predictions),
        "markdown": (args.from_markdown,),
    }
    given = [name for name, paths in modes.items() if any(p is not None for p in paths)]
    if len(given) != 1 or any(p is None for p in modes[given[0]]):
        parser.error("give exactly one input: --cards with --scores, --labels with --predictions, "
                     "or --from-markdown")
    try:
        hint = "lower --n-cases"
        if given[0] == "markdown":
            features = parse_markdown(Path(args.from_markdown).read_text(encoding="utf-8"))
            if not features:
                raise BuildError(f"no feature block found in {args.from_markdown}")
            dataset = args.dataset
        else:
            inputs = (load_medsae(args.cards, args.scores) if given[0] == "medsae"
                      else load_paper(args.labels, args.predictions))
            ranked = rank_features(inputs, args.top)
            if not ranked:
                raise BuildError("no feature has a usable Interpreter LLM description together with judged "
                                 "positive and negative passages")
            features = pick_passages(ranked, args.seed, args.max_chars, STORED_TEXT_CHARS)
            dataset = args.dataset or inputs.dataset
            hint = ("lower --n-cases or raise --top" if len(ranked) == args.top else
                    f"only {len(ranked)} features have a usable description and judged passages of both labels "
                    f"(fewer than --top {args.top}): describe more features with medsae interpret (leave out "
                    "--n-features, or give 60 or more), or lower --n-cases")
        try:
            cases = sample_cases(features, args.n_cases, args.seed)
        except BuildError as exc:
            raise BuildError(f"{exc}: {hint}") from exc
        validate_cases(cases)
    except (BuildError, CasesError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_text(features, dataset, args.max_chars), encoding="utf-8")
    n_positive = sum(c["ground_truth"] for c in cases)
    n_negative = len(cases) - n_positive
    print(f"wrote {len(cases)} cases ({n_positive} positive, {n_negative} negative) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
