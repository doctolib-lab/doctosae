"""Label concept-specific SAE neurons with an LLM, from select_neurons.py's neuron-profiles JSON.

For each neuron in --neuron-profiles, prompts an LLM with its top-activating corpus passages and
queries and asks what concept they share. Returns a NeuronLabel: a topic, a 1-2 sentence
description and up to --n-keywords keywords. Passages are shown whole, one activation value each,
with queries alongside them. Prompt after Bills et al. 2023 and O'Neill et al. 2024
(arXiv:2408.00657).

--llm-backend local loads a Hugging Face causal LM on --device; gateway calls an
OpenAI-compatible endpoint and requires --llm-base-url. --dry-run prints each prompt and calls no
LLM. Failed calls become LABEL_ERROR rows and unparseable replies PARSE_ERROR rows, both written
with everything that succeeded. The output JSON records `llm_backend` and `llm_model`; the sibling
`.md` carries the same in a footer.

Usage:
    uv run python -m src.interpretability.label_neurons \
        --neuron-profiles cure_k64_neuron_profiles.json \
        --llm-backend local --device cuda:0 \
        --output cure_k64_labels.json

    uv run python -m src.interpretability.label_neurons \
        --neuron-profiles cure_k64_neuron_profiles.json --llm-backend gateway \
        --llm-base-url https://your-vllm-host \
        --output cure_k64_labels.json

    uv run python -m src.interpretability.label_neurons \
        --neuron-profiles cure_k64_neuron_profiles.json --dry-run --output /dev/null
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from src.interpretability.llm_backend import GatewayLLM, LocalLLM

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Hugging Face id of the Interpreter LLM, also the default Activation Prediction LLM; the
# INTERPRETER_LLM environment variable overrides it. The model is gated: accept its terms on the
# Hub and log in with `hf auth login` first. A gateway call needs its own --llm-model.
DEFAULT_MODEL = os.environ.get("INTERPRETER_LLM", "google/medgemma-4b-it")

DEFAULT_N_KEYWORDS = 10
MAX_KEYWORDS = 10   # ceiling enforced in NeuronLabel; --n-keywords only shapes the prompt


class NeuronLabel(BaseModel):
    """Structured explanation of one SAE neuron, as returned by the LLM."""

    # topic carries no default, unlike the rest.
    topic: str = Field(description="The broad medical topic, in 1-3 words "
                                   "(e.g. 'cardiology', 'wound care')")
    description: str = Field(default="", description="1-2 plain sentences on what makes the "
                                                     "neuron fire, grounded only in the shown "
                                                     "examples, with no abbreviations")
    keywords: list[str] = Field(default_factory=list,
                                description="Distinctive terms shared by the examples, most "
                                            "distinctive first")

    @field_validator("keywords", mode="before")
    @classmethod
    def _coerce_keywords(cls, value):
        """Coerce the shapes a 4B model reaches for, drop duplicates, cap at MAX_KEYWORDS."""
        if value is None:
            return []
        if isinstance(value, str):
            value = [value] if value.strip() else []
        if not isinstance(value, list):
            return value
        out, seen = [], set()
        for v in value:
            text = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
            key = text.strip().casefold()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(text)
            if len(out) == MAX_KEYWORDS:
                break
        return out

    @field_validator("topic", "description", mode="before")
    @classmethod
    def _coerce_str(cls, value):
        """Flatten a list or null into the single string these fields are specified as."""
        if value is None:
            return ""
        if isinstance(value, list):
            return "; ".join(str(v) for v in value)
        return value


SYSTEM_PROMPT = """Context: you are an investigator looking at one neuron from a medical AI \
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


def build_system_prompt(n_keywords: int = DEFAULT_N_KEYWORDS) -> str:
    """Render the system prompt for a given keyword budget."""
    return SYSTEM_PROMPT.format(n_keywords=n_keywords)


def _example_line(s: dict, max_chars: int | None = None) -> str:
    """One example, one line: activation where the example carries one, then text with its title
    marked. max_chars spares the title, and 0 shows the title alone."""
    title = s.get("title")
    body = s["text"][len(title) + 1:] if title and s["text"].startswith(title + " ") else None
    if max_chars == 0:
        # "Title only", except a query has no title.
        shown = f"TITLE: {title}" if body is not None else s["text"]
    elif body is not None:
        shown = f"TITLE: {title} | TEXT: {body if max_chars is None else body[:max_chars]}"
    else:
        shown = s["text"] if max_chars is None else s["text"][:max_chars]
    act = f"[{s['activation']:.2f}] " if s.get("activation") is not None else ""
    return f"  {act}{shown}"


def build_prompt(neuron: dict, corpus_header: str = "Highest activating corpus documents",
                 query_header: str = "Highest activating queries", max_chars: int | None = None,
                 max_corpus: int | None = None, max_queries: int | None = None,
                 show_negatives: bool = False) -> str:
    """Render one neuron's activating examples as the user prompt, strongest first, each prefixed
    with its activation value. `show_negatives` appends the documents it does not react to.
    """
    corpus = neuron["corpus_snippets"][:max_corpus]
    queries = neuron["query_snippets"][:max_queries]
    lines = [f"{corpus_header} ({len(corpus)}):"]
    if corpus:
        lines += [_example_line(s, max_chars) for s in corpus]
    else:
        # Reachable via --n-corpus-snippets 0.
        lines.append("  (none — no corpus passage was pulled for this neuron)")
    lines.append("")
    lines.append(f"{query_header} ({len(queries)}):")
    if queries:
        lines += [_example_line(s, max_chars) for s in queries]
    else:
        lines.append("  (none — this neuron fires on no query in the neuron profile)")
    negatives = neuron.get("negative_snippets") or []
    if show_negatives and negatives:
        lines += ["", "A few example documents that this neuron does not fire on at all:"]
        lines += [_example_line(s, max_chars) for s in negatives]
    return "\n".join(lines)


def _last_comma_outside_string(s: str) -> int | None:
    """Index of the last top-level ',' in `s` that isn't inside a JSON string, or None."""
    in_string, escape, last = False, False, None
    for i, ch in enumerate(s):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == ",":
            last = i
    return last


def _close_truncated_json(text: str) -> dict | None:
    """Recover a dict from a reply cut off mid-JSON: close any open string, then any open
    object/array, backing off past the last top-level comma. None if nothing parses.
    """
    start = text.find("{")
    if start == -1:
        return None
    candidate = text[start:]
    for _ in range(6):
        stack, in_string, escape = [], False, False
        for ch in candidate:
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if stack:
                    stack.pop()
        repaired = (candidate + '"' if in_string else candidate).rstrip()
        if repaired.endswith(","):
            repaired = repaired[:-1]
        repaired += "".join("}" if c == "{" else "]" for c in reversed(stack))
        try:
            obj = json.loads(repaired)
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            pass
        cut = _last_comma_outside_string(candidate)
        if cut is None:
            return None
        candidate = candidate[:cut]
    return None


def _brace_groups(text: str) -> tuple[list[str], int | None]:
    """Every balanced {...} group in `text`, plus the start of an unclosed one. Braces inside
    strings are skipped, so prose before the JSON cannot swallow it."""
    groups, stack, in_string, escape = [], [], False, False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            if not stack:
                groups.append(text[start:i + 1])
    return groups, stack[0] if stack else None


def _parse_json_label(text: str) -> NeuronLabel:
    """Parse a NeuronLabel out of a raw model response, tolerating prose or code fences around the
    JSON. Falls back to closing a truncated reply, then to a PARSE_ERROR label. Never raises.
    """
    groups, unclosed = _brace_groups(text)
    for group in reversed(groups):
        try:
            return NeuronLabel(**json.loads(group))
        except Exception:
            continue
    recovered = _close_truncated_json(text if unclosed is None else text[unclosed:])
    if recovered and recovered.get("topic"):
        try:
            return NeuronLabel(**recovered)
        except Exception:
            pass
    return NeuronLabel(topic="PARSE_ERROR", description=text, keywords=[])


class GatewayLabeler:
    """Labeler backed by an OpenAI-compatible chat endpoint (GatewayLLM), structured output
    enforced against NeuronLabel."""

    supports_concurrency = GatewayLLM.supports_concurrency

    def __init__(self, model: str, base_url: str, api_key: str, n_keywords: int = DEFAULT_N_KEYWORDS):
        self.llm = GatewayLLM(model, base_url, api_key, NeuronLabel)
        self.system = build_system_prompt(n_keywords)

    def label(self, user_prompt: str) -> NeuronLabel:
        """user_prompt: output of build_prompt. Returns the model's structured explanation."""
        return self.llm.generate(self.system, user_prompt)


class LocalLabeler:
    """Labeler backed by a locally-loaded Hugging Face causal LM (LocalLLM). Asks for raw JSON,
    parsed by _parse_json_label."""

    supports_concurrency = LocalLLM.supports_concurrency

    def __init__(self, model: str, device: str | None, dtype: str,
                 n_keywords: int = DEFAULT_N_KEYWORDS):
        self.llm = LocalLLM(model, device, dtype)
        self.system = build_system_prompt(n_keywords)
        self.n_keywords = n_keywords

    def _with_json_instruction(self, user_prompt: str) -> str:
        return user_prompt + ('\n\nWork through the steps above, briefly, then close with a JSON '
                              'object and nothing after it:\n'
                              '{"topic": "...", "description": "...", '
                              f'"keywords": ["...", "..."]}}  // at most {self.n_keywords} keywords')

    def label(self, user_prompt: str) -> NeuronLabel:
        """user_prompt: output of build_prompt. Returns the parsed explanation, or a
        PARSE_ERROR placeholder if the reply was not JSON."""
        return _parse_json_label(self.llm.generate(self.system, self._with_json_instruction(user_prompt)))

    def label_batch(self, user_prompts: list[str]) -> list[NeuronLabel]:
        """One label per prompt, all in one forward pass. Not bit-identical to one-at-a-time
        (left-padding, see LocalLLM.generate_batch)."""
        texts = self.llm.generate_batch(self.system,
                                        [self._with_json_instruction(u) for u in user_prompts])
        return [_parse_json_label(t) for t in texts]


def label_neurons(profiles: dict, labeler, max_workers: int = 1,
                  n_keywords: int = DEFAULT_N_KEYWORDS, max_chars: int | None = None,
                  max_corpus: int | None = None, max_queries: int | None = None,
                  batch_size: int = 1) -> list[dict]:
    """Label every neuron in a profiles dict: its id and profile statistics plus the LLM's topic,
    description and keywords. ``batch_size`` > 1 uses ``label_batch``; a raising call becomes a
    LABEL_ERROR row.
    """
    def _stats(neuron: dict) -> dict:
        return {"neuron": neuron["neuron"],
                "dominant_theme": neuron["dominant_theme"],
                "firing_rate_corpus": neuron["firing_rate_corpus"],
                "entropy_norm": neuron["entropy_norm"]}

    def _prompt(neuron: dict) -> str:
        return build_prompt(neuron, max_chars=max_chars, max_corpus=max_corpus,
                            max_queries=max_queries, show_negatives=True)

    def _row(neuron: dict, result) -> dict:
        return {**_stats(neuron), "topic": result.topic, "description": result.description,
                "keywords": list(result.keywords)[:n_keywords]}

    def _err(neuron: dict, exc: Exception) -> dict:
        print(f"neuron #{neuron['neuron']}: labelling failed ({type(exc).__name__}: {exc})")
        return {**_stats(neuron), "topic": "LABEL_ERROR",
                "description": f"{type(exc).__name__}: {exc}", "keywords": []}

    def _one(neuron: dict) -> dict:
        try:
            return _row(neuron, labeler.label(_prompt(neuron)))
        except Exception as exc:
            return _err(neuron, exc)

    neurons = profiles["neurons"]
    if batch_size > 1 and hasattr(labeler, "label_batch"):
        rows = []
        for start in range(0, len(neurons), batch_size):
            chunk = neurons[start:start + batch_size]
            try:
                rows += [_row(n, r) for n, r in zip(chunk, labeler.label_batch([_prompt(n) for n in chunk]))]
            except Exception as exc:
                # One batch failing costs only its own neurons.
                rows += [_err(n, exc) for n in chunk]
            print(f"  {len(rows)}/{len(neurons)} labelled")
        return rows

    if not getattr(labeler, "supports_concurrency", True) and max_workers > 1:
        print(f"{type(labeler).__name__} is not thread-safe — running 1 neuron at a time "
              f"instead of {max_workers}")
        max_workers = 1
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        rows = []
        for row in ex.map(_one, neurons):
            rows.append(row)
            print(f"  {len(rows)}/{len(neurons)} labelled")
        return rows


def _md_cell(value) -> str:
    """Escape a value for a markdown table cell: a literal | or newline would corrupt the table."""
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()


def to_markdown(rows: list[dict]) -> str:
    """Render :func:`label_neurons`'s rows as a markdown table, one row per neuron."""
    lines = ["| neuron | theme | corpus fr | entropy | topic | description | keywords |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['neuron']} | {_md_cell(r['dominant_theme'])} | "
                     f"{r['firing_rate_corpus']:.2%} | {r['entropy_norm']:.3f} | "
                     f"{_md_cell(r['topic'])} | {_md_cell(r['description'])} | "
                     f"{_md_cell(', '.join(r.get('keywords') or []))} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--neuron-profiles", required=True, type=Path, help="select_neurons.py's output JSON")
    p.add_argument("--llm-backend", choices=["gateway", "local"], default=None,
                   help="'local': a Hugging Face model on this machine (no credentials, no "
                        "network, but a one-off weights download). 'gateway': an "
                        "OpenAI-compatible endpoint (needs a base URL, sends text off-box). "
                        "Required unless --dry-run — deliberately not defaulted, since the two "
                        "differ in where your corpus text goes, which should never be an accident")
    p.add_argument("--llm-model", default=DEFAULT_MODEL,
                   help=f"Model id (default: {DEFAULT_MODEL}). For --llm-backend gateway it must "
                        f"be one the endpoint actually serves; for local, a Hugging Face id")
    p.add_argument("--llm-base-url", default=None,
                   help="Gateway backend only. Default: $LITELLM_PROXY_API_BASE")
    p.add_argument("--llm-api-key", default=None,
                   help="Gateway backend only, for endpoints that need one. Default: "
                        "$LITELLM_PROXY_API_KEY")
    p.add_argument("--max-example-chars", type=int, default=1000,
                   help="Cap each shown example's body in the prompt (default 1000)")
    p.add_argument("--max-corpus-examples", type=int, default=None,
                   help="Show at most this many corpus documents, strongest first (default: all)")
    p.add_argument("--max-query-examples", type=int, default=None,
                   help="Show at most this many queries, strongest first (default: all)")
    p.add_argument("--n-keywords", type=int, default=DEFAULT_N_KEYWORDS,
                   help="Keywords requested per neuron, listed after the description")
    p.add_argument("--device", default=None, help="Local backend only, e.g. cuda:0 (default: device_map='auto')")
    p.add_argument("--dtype", default="bfloat16", help="Local backend only, a torch dtype name")
    p.add_argument("--max-workers", type=int, default=8,
                   help="Concurrent gateway requests. Forced to 1 for --llm-backend local, "
                        "which batches instead (see --batch-size)")
    p.add_argument("--batch-size", type=int, default=1,
                   help="--llm-backend local: neurons per forward pass (default 1). Keep fixed "
                        "across compared runs -- it changes borderline greedy decodes")
    p.add_argument("--dry-run", action="store_true",
                   help="Print each neuron's prompt and exit without building a labeler or calling any LLM")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    profiles = json.load(open(args.neuron_profiles))
    print(f"{len(profiles['neurons'])} neurons to label, from {args.neuron_profiles}")

    if args.dry_run:
        print(f"\n########## SYSTEM ##########\n{build_system_prompt(args.n_keywords)}")
        for neuron in profiles["neurons"]:
            print(f"\n########## neuron #{neuron['neuron']} ##########\n"
                  f"{build_prompt(neuron, max_chars=args.max_example_chars,
                                  max_corpus=args.max_corpus_examples,
                                  max_queries=args.max_query_examples, show_negatives=True)}")
        return

    # Only a real run needs a backend; --dry-run calls nothing.
    if not args.llm_backend:
        p.error("--llm-backend is required (choose 'local' or 'gateway'), unless --dry-run")

    if args.llm_backend == "gateway":
        base_url = args.llm_base_url or os.environ.get("LITELLM_PROXY_API_BASE")
        api_key = args.llm_api_key or os.environ.get("LITELLM_PROXY_API_KEY")
        # Required: left None, the client falls back to api.openai.com.
        if not base_url:
            p.error("--llm-backend gateway needs --llm-base-url or $LITELLM_PROXY_API_BASE "
                    "(refusing to fall back to the public OpenAI endpoint)")
        if not api_key:
            # The client demands a non-empty string.
            print("no API key given — sending a placeholder (fine for a keyless self-hosted "
                  "server; a managed proxy will reject it)")
            api_key = "EMPTY"
        labeler = GatewayLabeler(args.llm_model, base_url, api_key, args.n_keywords)
    else:
        labeler = LocalLabeler(args.llm_model, args.device, args.dtype, args.n_keywords)

    # label_neurons clamps max_workers itself for LocalLabeler.
    rows = label_neurons(profiles, labeler, args.max_workers, args.n_keywords,
                         args.max_example_chars, args.max_corpus_examples, args.max_query_examples,
                         batch_size=args.batch_size)
    n_failed = {lbl: sum(1 for r in rows if r["topic"] == lbl)
                for lbl in ("LABEL_ERROR", "PARSE_ERROR")}
    # Record which model wrote these labels.
    result = {
        "neuron_profiles_file": str(args.neuron_profiles),
        "llm_backend": args.llm_backend,
        "llm_model": args.llm_model,
        "batch_size": args.batch_size,
        "n_keywords": args.n_keywords,
        # Prompt-side caps.
        "max_example_chars": args.max_example_chars,
        "max_corpus_examples": args.max_corpus_examples,
        "max_query_examples": args.max_query_examples,
        "n_labelled": len(rows),
        "n_failed": {k: v for k, v in n_failed.items() if v},
        "neurons": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.output, "w"), indent=2)
    md_path = args.output.with_suffix(".md")
    md_path.write_text(to_markdown(rows) + f"\n\nLabelled by `{args.llm_model}` "
                                           f"(--llm-backend {args.llm_backend}).\n")
    print(f"wrote {args.output} and {md_path} ({len(rows)} labels from {args.llm_model})")
    # The rows are written either way.
    for lbl, n in n_failed.items():
        if n:
            print(f"{n} / {len(rows)} neurons came back as {lbl}")


if __name__ == "__main__":
    main()
