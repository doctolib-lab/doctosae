"""Case schema shared by the app and `scripts/build_cases.py`.

A case pairs one SAE feature description with one passage. This module imports neither streamlit nor medsae,
so the build script and the tests only need the standard library.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

PASSAGE_MAX_CHARS = 300  # annotators see the first 300 characters of each passage
SAMPLE_SEED = 42  # seeds the per annotator order
MINUTES_PER_CASE = 0.5  # 30 cases took about 15 minutes

# Case ids land inside a <script> comment and in widget keys, so they stay plain.
CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_]{1,32}$")
# Key order of the cases file, kept by the build script so diffs stay readable.
CASE_KEYS = ("rank", "feature_id", "judge_input", "topic", "description", "keywords", "passage",
             "ground_truth", "case_id")
EXTRA_KEYS = ("truncated", "feature_r", "judge_probability", "source")


class CasesError(ValueError):
    """A cases file the app cannot show. `problems` lists every issue, so one pass fixes the file."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("invalid cases:\n" + "\n".join(f"* {p}" for p in self.problems))


def judge_input(topic: str, description: str, keywords: Sequence[str]) -> str:
    """The text the Activation Prediction LLM read for an Interpreter LLM description.

    This is the paper's `build_class_description(..., "llm")` without its two leading spaces. An empty keyword
    list still gives "(keywords: )", as in the paper.
    """
    return f"{topic}: {description} (keywords: {', '.join(keywords)})"


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return _is_int(value) or (isinstance(value, float) and math.isfinite(value))


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


_Check = tuple[Callable[[Any], bool], str]
_REQUIRED: dict[str, _Check] = {
    "feature_id": (_is_int, "an integer"),
    "rank": (_is_int, "an integer"),
    "description": (_is_text, "a non empty string"),
    "passage": (_is_text, "a non empty string"),
}
# Optional keys; null counts as absent. Unknown keys are ignored.
_OPTIONAL: dict[str, _Check] = {
    "topic": (lambda v: isinstance(v, str), "a string"),
    "keywords": (lambda v: isinstance(v, list) and all(isinstance(k, str) for k in v), "a list of strings"),
    "judge_input": (lambda v: isinstance(v, str), "a string"),
    "ground_truth": (lambda v: _is_int(v) and v in (0, 1), "0 or 1"),
    "truncated": (lambda v: isinstance(v, bool), "true or false"),
    "feature_r": (_is_number, "a finite number"),
    "judge_probability": (lambda v: _is_number(v) and 0 <= v <= 1, "a number between 0 and 1"),
    "source": (lambda v: isinstance(v, str), "a string"),
}


def _problems(case: dict, where: str) -> list[str]:
    found = []
    for key, (ok, expected) in _REQUIRED.items():
        if key not in case:
            found.append(f"{where}: missing {key}")
        elif not ok(case[key]):
            found.append(f"{where}: {key} must be {expected}, got {case[key]!r:.60}")
    for key, (ok, expected) in _OPTIONAL.items():
        if case.get(key) is not None and not ok(case[key]):
            found.append(f"{where}: {key} must be {expected}, got {case[key]!r:.60}")
    return found


def validate_cases(cases: Any) -> list[dict]:
    """Check a list of cases and return copies with the optional display fields filled in.

    Raises CasesError listing every problem found, not only the first one.
    """
    if not isinstance(cases, list) or not cases:
        raise CasesError(["the cases file must hold a non empty JSON list of objects"])
    problems: list[str] = []
    seen: set[str] = set()
    valid = []
    for i, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            problems.append(f"case {i}: expected an object, got {type(case).__name__}")
            continue
        case_id = case.get("case_id")
        where = f"case {i}"
        if not isinstance(case_id, str) or not CASE_ID_PATTERN.match(case_id):
            problems.append(f"{where}: case_id must be 1 to 32 letters, digits or underscores, "
                            f"got {case_id!r:.60}")
        elif case_id in seen:
            problems.append(f"{where}: duplicate case_id {case_id}")
        else:
            seen.add(case_id)
            where = f"case {case_id}"
        case_problems = _problems(case, where)
        problems += case_problems
        if not case_problems:
            keywords = list(case.get("keywords") or [])
            valid.append({**case, "topic": case.get("topic") or "", "keywords": keywords})
    if problems:
        raise CasesError(problems)
    return valid


def load_cases(path: str | Path) -> list[dict]:
    """Read and validate a cases JSON file; every failure is a CasesError the app can show as is."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CasesError([f"cases file not found: {path}"]) from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CasesError([f"cannot read {path.name}: {exc}"]) from exc
    return validate_cases(data)
