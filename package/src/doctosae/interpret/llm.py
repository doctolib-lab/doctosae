"""Language model backends for the Interpreter LLM and the Activation Prediction LLM.

Every backend implements :class:`LLM`: raw text replies to one system prompt and a batch of user turns. The
prompts end by asking for a JSON object, and the replies are parsed as text by the same functions whatever the
backend (:func:`doctosae.interpret.labeling.parse_label`, :func:`doctosae.interpret.scoring.parse_score`),
which is how the paper produced every number.

* :class:`TransformersLLM` runs a local chat model with greedy decoding, as the paper did with
  ``google/medgemma-4b-it`` (bf16, 400 new tokens, left padded batches).
* :class:`OpenAICompatibleLLM` calls any server that speaks the OpenAI chat completions API (vLLM, Ollama, TGI
  or the OpenAI API) with temperature 0.
* :class:`FakeLLM` returns deterministic replies without any model, for tests and documentation.

A backend whose batch partly fails raises :class:`PartialReplies` with the replies it got, so that callers
only send the failed turns again.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

from doctosae.interpret.prompts import NEGATIVES_HEADER, SCORE_INSTRUCTION

__all__ = ["LLM", "FakeLLM", "OpenAICompatibleLLM", "PartialReplies", "TransformersLLM", "generate_isolated"]

logger = logging.getLogger(__name__)

PAPER_MODEL = "google/medgemma-4b-it"
MAX_NEW_TOKENS = 400

_GATED_HINT = (
    "{model} is a gated model on the Hugging Face Hub: accept its terms on https://huggingface.co/{model} "
    "with your account, then log in with `hf auth login` (or pass token=...) and try again."
)
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "0.0.0.0")


class PartialReplies(RuntimeError):
    """Some turns of a batch failed while the others got a reply.

    Attributes:
        replies: One reply per user turn, ``None`` where the request failed.
        errors: Position of each failed turn to its exception.
    """

    def __init__(self, replies: list[str | None], errors: Mapping[int, BaseException]) -> None:
        self.replies = replies
        self.errors = dict(errors)
        first = next(iter(self.errors.values()))
        super().__init__(
            f"{len(self.errors)} of {len(replies)} requests failed: {type(first).__name__}: {first}"
        )


@runtime_checkable
class LLM(Protocol):
    """A language model as the interpretation pipeline uses it.

    ``generate`` returns one raw text reply per user turn, in order, each user turn sent after the same system
    prompt. The Interpreter sends consecutive chunks of 20 features (labels) or 8 pairs (activation
    prediction), the batch sizes of the paper runs.
    """

    def generate(self, system: str, users: Sequence[str]) -> list[str]: ...


def _dtype_keyword(transformers_version: str) -> str:
    """``from_pretrained`` takes ``dtype`` since Transformers 4.56 and ``torch_dtype`` before."""
    major, minor = (int(part) for part in re.findall(r"\d+", transformers_version)[:2])
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


class TransformersLLM:
    """A local chat model run with Hugging Face Transformers, decoded greedily (the paper's local backend).

    Args:
        model: Hub id or local path of a chat model. The paper used ``google/medgemma-4b-it``, which is gated:
            accept its terms on https://huggingface.co/google/medgemma-4b-it, then log in with
            ``hf auth login`` (or pass ``token``).
        device: ``device_map`` used for loading, for example ``"cuda:0"`` or ``"cpu"``; ``None`` lets
            Accelerate place the model (``"auto"``), as the paper did.
        dtype: Torch dtype name of the weights, ``"bfloat16"`` in the paper.
        max_new_tokens: Generation budget per reply, 400 in the paper (brief reasoning, then the JSON).
        deterministic: Use deterministic kernels, as the paper did. ``torch.use_deterministic_algorithms`` is
            switched on (kernels without a deterministic version only warn) during each :meth:`generate`
            call only, and the previous setting is restored afterwards. ``CUBLAS_WORKSPACE_CONFIG`` cannot be
            scoped that way: it is set for the whole process (unless already set) and inherited by
            subprocesses, and it only takes effect when set before anything initialises CUDA.
        token: Hugging Face token for gated or private models.

    One :meth:`generate` call is one left padded batch. Greedy outputs depend slightly on the batch
    composition, so keep the batch sizes fixed across compared runs.
    """

    def __init__(
        self,
        model: str = PAPER_MODEL,
        device: str | None = None,
        dtype: str = "bfloat16",
        max_new_tokens: int = MAX_NEW_TOKENS,
        deterministic: bool = True,
        token: str | None = None,
    ) -> None:
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            import torch
            import transformers
        except ImportError as error:
            message = "TransformersLLM needs transformers: pip install 'doctosae[local-llm]'"
            raise ImportError(message) from error
        self.deterministic = bool(deterministic)
        self.model_name = model
        self.max_new_tokens = int(max_new_tokens)
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model, token=token)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model,
                token=token,
                device_map=device or "auto",
                **{_dtype_keyword(transformers.__version__): getattr(torch, dtype)},
            )
        except OSError as error:
            if "gated" in str(error).lower():
                raise OSError(_GATED_HINT.format(model=model)) from error
            raise
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def __repr__(self) -> str:
        return f"TransformersLLM(model={self.model_name!r})"

    @contextmanager
    def _kernels(self) -> Iterator[None]:
        """Deterministic kernels for the duration of one call, then the caller's setting again."""
        if not self.deterministic:
            yield
            return
        import torch

        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)

    def _encode(self, conversations: list[list[dict[str, str]]]) -> Any:
        return self.tokenizer.apply_chat_template(
            conversations, add_generation_prompt=True, return_tensors="pt", return_dict=True, padding=True
        )

    def generate(self, system: str, users: Sequence[str]) -> list[str]:
        """Greedy replies for a batch of user turns, all in one forward pass (prompt tokens not echoed)."""
        if not users:
            return []
        from jinja2 import TemplateError

        # The chat template decides where the system prompt goes (Gemma 3 folds it into the user turn).
        try:
            inputs = self._encode(
                [[{"role": "system", "content": system}, {"role": "user", "content": user}] for user in users]
            )
        except TemplateError as error:
            if "system" not in str(error).lower():
                raise
            # Templates without a system turn (Gemma 2 for one): fold it into the user turn as Gemma 3 does.
            inputs = self._encode([[{"role": "user", "content": f"{system}\n\n{user}"}] for user in users])
        inputs = inputs.to(self.model.device)
        with self._kernels():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        n_prompt = inputs["input_ids"].shape[1]
        return [self.tokenizer.decode(row[n_prompt:], skip_special_tokens=True) for row in output]


class OpenAICompatibleLLM:
    """A model served behind the OpenAI chat completions API (vLLM, Ollama, TGI, OpenAI...).

    Args:
        model: Model name as the server knows it.
        base_url: Server URL, for example ``http://localhost:8000/v1``; defaults to ``$OPENAI_BASE_URL``.
            One of them is required: the texts of the evidence corpus and of the evaluation pool are sent to
            this server, so the package never falls back to a public endpoint on its own. To use the OpenAI
            API, pass ``base_url="https://api.openai.com/v1"``. A server that is not on this machine is named
            in a warning.
        api_key: Defaults to ``$OPENAI_API_KEY``; a keyless local server gets the placeholder ``"EMPTY"``.
        temperature: Sampling temperature, 0 as in the paper; ``None`` leaves it out of the request.
        max_tokens: Reply budget, 400 like the local backend; ``None`` leaves it out of the request.
        max_workers: Concurrent requests within one :meth:`generate` call.
        client: A ready ``openai.OpenAI`` client, used as is (custom timeouts, proxies, tests).
        request_kwargs: Extra arguments of every ``chat.completions.create`` call, for example
            ``{"max_completion_tokens": 400}`` with ``max_tokens=None`` and ``temperature=None``, as models
            with hidden reasoning require, or ``{"extra_body": ...}``.
        **client_kwargs: Passed to ``openai.OpenAI`` (``timeout``, ``max_retries``...).

    The prompts expect a chat model that answers directly: a model with hidden reasoning spends much of the
    400 token budget before the JSON object the parsers read.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = 0.0,
        max_tokens: int | None = MAX_NEW_TOKENS,
        max_workers: int = 8,
        client: Any | None = None,
        request_kwargs: Mapping[str, Any] | None = None,
        **client_kwargs: Any,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise ImportError("OpenAICompatibleLLM needs openai: pip install 'doctosae[llm]'") from error
            base_url = base_url or os.environ.get("OPENAI_BASE_URL")
            if not base_url:
                raise ValueError(
                    "OpenAICompatibleLLM needs base_url (or OPENAI_BASE_URL): refusing to fall back to the "
                    "public OpenAI endpoint. Pass base_url='https://api.openai.com/v1' to use the OpenAI API."
                )
            api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
            client = OpenAI(base_url=base_url, api_key=api_key, **client_kwargs)
            host = urlsplit(str(client.base_url)).hostname or ""
            if host not in _LOCAL_HOSTS:
                logger.warning("Evidence and evaluation texts are sent to %s (model %s)", host, model)
            else:
                logger.info("OpenAI compatible model %s at %s", model, client.base_url)
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = None if max_tokens is None else int(max_tokens)
        self.max_workers = max(1, int(max_workers))
        self.request_kwargs = dict(request_kwargs or {})

    def __repr__(self) -> str:
        return f"OpenAICompatibleLLM(model={self.model!r})"

    def _complete(self, system: str, user: str) -> str:
        request = {"temperature": self.temperature, "max_tokens": self.max_tokens}
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **{key: value for key, value in request.items() if value is not None},
            **self.request_kwargs,
        )
        if not response.choices:  # an empty reply: parsed as an error, never a crash
            return ""
        return response.choices[0].message.content or ""

    def generate(self, system: str, users: Sequence[str]) -> list[str]:
        """One request per user turn, ``max_workers`` at a time; replies come back in order.

        Every request runs to its end: when some fail, :class:`PartialReplies` carries the other replies.
        """
        if len(users) <= 1 or self.max_workers == 1:
            outcomes = [self._outcome(system, user) for user in users]
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(users))) as pool:
                outcomes = list(pool.map(lambda user: self._outcome(system, user), users))
        errors = {i: outcome for i, outcome in enumerate(outcomes) if isinstance(outcome, BaseException)}
        if errors:
            replies = [None if isinstance(outcome, BaseException) else outcome for outcome in outcomes]
            raise PartialReplies(replies, errors) from next(iter(errors.values()))
        return outcomes

    def _outcome(self, system: str, user: str) -> str | Exception:
        try:
            return self._complete(system, user)
        except Exception as error:  # kept per request, so one failure does not hide the other replies
            return error


def _cause(error: Exception) -> Exception:
    """The error of the request itself, not the batch wrapper around it."""
    if isinstance(error, PartialReplies) and len(error.errors) == 1:
        cause = next(iter(error.errors.values()))
        return cause if isinstance(cause, Exception) else error
    return error


def generate_isolated(llm: LLM, system: str, users: Sequence[str]) -> list[str | Exception]:
    """Replies to ``users``, with failures isolated per turn: one failed request never costs the others.

    When the batch call fails, the turns without a reply are sent again one at a time (only the failed ones
    after :class:`PartialReplies`); a turn that still fails gets its exception in place of a reply. Greedy
    replies can depend slightly on the batch, so the turns that succeeded in the batch are never sent again.
    """
    users = list(users)
    try:
        replies = llm.generate(system, users)
        if len(replies) != len(users):
            raise RuntimeError(f"{len(replies)} replies for {len(users)} prompts")
        return list(replies)
    except PartialReplies as partial:
        outcomes: list[str | Exception | None] = list(partial.replies)
        error: Exception = partial
    except Exception as failure:  # any backend error: retried turn by turn below
        outcomes, error = [None] * len(users), failure
    if len(users) == 1:
        return [_cause(error)]
    logger.warning("an LLM call failed (%s): its failed prompts are sent one at a time", error)
    for position, outcome in enumerate(outcomes):
        if outcome is not None:
            continue
        try:
            [outcomes[position]] = llm.generate(system, [users[position]])
        except Exception as failure:
            outcomes[position] = _cause(failure)
    return outcomes


_WORD = re.compile(r"[^\W\d_]{4,}")
_MARKUP = re.compile(r"\[\d+\.\d+\]|TITLE:|\| TEXT:")


def _words(text: str) -> list[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return [word for word in _WORD.findall(text.lower()) if word not in ENGLISH_STOP_WORDS]


def _fake_reply(system: str, user: str) -> str:
    """Score replies from word overlap between description and text; labels from the most frequent words."""
    if user.endswith(SCORE_INSTRUCTION):
        head, _, text = user[: -len(SCORE_INSTRUCTION)].rpartition("\nText:\n")
        description = set(_words(head.partition("\n")[2]))
        words = set(_words(text))
        overlap = len(description & words) / max(1, min(len(description), len(words)))
        return json.dumps({"score": round(overlap, 3)})
    # Words of the example lines that fire, without the activation values and title markers.
    firing = user.split(NEGATIVES_HEADER, 1)[0].split("\n\nWork through the steps above", 1)[0]
    lines = [_MARKUP.sub(" ", line) for line in firing.split("\n") if line.startswith("  ")]
    counts = Counter(_words(" ".join(lines)))
    keywords = [word for word, _ in counts.most_common(5)]
    topic = keywords[0] if keywords else "unknown"
    label = {"topic": topic, "description": f"Texts about {topic}.", "keywords": keywords}
    return "The examples share one theme.\n" + json.dumps(label)


class FakeLLM:
    """Deterministic stand in for a language model, for tests and documentation.

    Without ``reply``, an Activation Prediction turn gets ``{"score": s}`` where ``s`` is the word overlap
    between the feature description and the text, and an Interpreter turn gets a label made of the most
    frequent words of the evidence. Every ``(system, user)`` pair received is kept in :attr:`calls`.

    Args:
        reply: Function ``(system, user) -> reply text`` replacing the default behaviour.
    """

    def __init__(self, reply: Callable[[str, str], str] | None = None) -> None:
        self.reply = reply or _fake_reply
        self.calls: list[tuple[str, str]] = []

    def __repr__(self) -> str:
        # Recorded in ScoreReport.llm and scores.json: no memory address, so reports stay comparable.
        return "FakeLLM()" if self.reply is _fake_reply else "FakeLLM(reply=custom)"

    def generate(self, system: str, users: Sequence[str]) -> list[str]:
        replies = []
        for user in users:
            self.calls.append((system, user))
            replies.append(self.reply(system, user))
        return replies
