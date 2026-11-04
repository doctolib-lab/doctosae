"""LaTeX figures of the prompts used in this package: the Interpreter (label_neurons.py) and the
classifier (predict_neurons_binary.py, one box per neuron-description type).

Each box is a tcolorbox holding one prompt, with its markup derived from the prompt constants.
Requires \\usepackage[most]{tcolorbox} and \\usepackage{enumitem}.

Usage:
    uv run python -m src.interpretability.latex_prompt_figures --output-dir figures
"""
from __future__ import annotations

import argparse
import inspect
import re
from types import SimpleNamespace

from src.interpretability.chart_style import save_text
from src.interpretability.llm_backend import LocalLLM
from src.interpretability.label_neurons import (DEFAULT_N_KEYWORDS, LocalLabeler,
                                                build_system_prompt as interp_system_prompt)
from src.interpretability.predict_neurons_binary import (CLASS_DEFN, OOD_NOTE, SCORE_INSTRUCTION,
                                                          USER_PROMPT,
                                                          build_system_prompt as cls_system_prompt)
from src.interpretability.select_neurons import (DEFAULT_N_CORPUS_SNIPPETS, DEFAULT_N_NEGATIVE_SNIPPETS,
                                                 DEFAULT_N_QUERY_SNIPPETS)

INTERPRETER_MAX_EXAMPLE_CHARS = 1000    # label_neurons.py's --max-example-chars default
PINK = ("magenta!60!black", "pink!8!white")     # interpreter: frame, background
PURPLE = ("violet!65!black", "violet!8!white")   # classifier

_ESCAPES = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
            "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}

INTERPRETER_EXAMPLES = f"""Highest activating corpus documents ({DEFAULT_N_CORPUS_SNIPPETS}):
  [0.31] TITLE: <document title> | TEXT: <document text>
  ... {DEFAULT_N_CORPUS_SNIPPETS - 1} more, by decreasing activation

Highest activating queries ({DEFAULT_N_QUERY_SNIPPETS}):
  [0.24] <query text>
  ... {DEFAULT_N_QUERY_SNIPPETS - 1} more, by decreasing activation

A few example documents that this neuron does not fire on at all:
""" + "\n".join(["  TITLE: <document title> | TEXT: <document text>"] * DEFAULT_N_NEGATIVE_SNIPPETS)


def _escape(s: str) -> str:
    return "".join(_ESCAPES.get(c, c) for c in s)


def _inline(s: str) -> str:
    """Escape, then set quoted field names, bracketed activations and <placeholders> in \\texttt."""
    out = _escape(s)
    out = re.sub(r'"([a-z]+)"', r'\\texttt{"\1"}', out)
    out = re.sub(r"\[([0-9.]+)\]", r"\\texttt{[\1]}", out)
    out = re.sub(r"<([^<>]+)>", r"\\texttt{<\1>}", out)
    out = out.replace("TITLE:", r"\texttt{TITLE:}").replace("TEXT:", r"\texttt{TEXT:}")
    return out.replace("|", r"\texttt{|}")


def _list(items: list[str], env: str) -> str:
    """`items` as a LaTeX itemize or enumerate."""
    return (f"\\begin{{{env}}}[leftmargin=*, itemsep=1pt, topsep=2pt]\n"
            + "\n".join(f"  \\item {_inline(i)}" for i in items)
            + f"\n\\end{{{env}}}")


def _paragraph(lines: list[str]) -> str:
    """One prompt paragraph as LaTeX: '- ' lines become an itemize and '1. ' lines an enumerate,
    while a leading 'Word:' becomes a bold run-in heading."""
    bullets = [l[2:] for l in lines if l.startswith("- ")]
    numbered = [re.sub(r"^\d+\.\s*", "", l) for l in lines if re.match(r"^\d+\.\s", l)]
    head = " ".join(l for l in lines
                    if not l.startswith("- ") and not re.match(r"^\d+\.\s", l)).strip()
    out = []
    if head:
        m = re.match(r"^([A-Z][A-Za-z ]{1,30}):\s*(.*)$", head, re.DOTALL)
        if m:
            rest = _inline(m.group(2))
            out.append(f"\\textbf{{{_escape(m.group(1))}:}} {rest}".rstrip())
        else:
            out.append(_inline(head))
    if bullets:
        out.append(_list(bullets, "itemize"))
    if numbered:
        out.append(_list(numbered, "enumerate"))
    return "\n\n".join(out)


def prompt_box(title: str, prompt: str, colors: tuple[str, str],
               verbatim_tail: str | None = None, labels: tuple[str, str] | None = None) -> str:
    """One tcolorbox in (frame, background) `colors`: `prompt` as flowing text with bold run-ins
    and itemize, then `verbatim_tail` (the example block) as monospace lines. `labels` heads the
    prompt and the tail, e.g. as the system and the user message."""
    body = "\n\n".join(_paragraph(p.split("\n")) for p in prompt.split("\n\n") if p.strip())
    if labels:
        body = f"\\textbf{{\\underline{{{labels[0]}}}}}\n\n" + body
    if verbatim_tail:
        gap = "\n\n\\medskip\n"
        if labels:
            body += f"{gap}\\textbf{{\\underline{{{labels[1]}}}}}"
            gap = "\n"
        # \\ inside a \texttt{} group escapes it; a flushleft block takes them.
        rows = []
        for line in verbatim_tail.split("\n"):
            if line.strip():
                rows.append(_escape(line).replace(" ", "~"))
            elif rows:
                rows[-1] += "\\\\[4pt]"
        tail = "\\\\\n".join(rows).replace("\\\\[4pt]\\\\\n", "\\\\[4pt]\n")
        body += (gap + "\\begin{flushleft}\\ttfamily\\footnotesize\n"
                 + tail + "\n\\end{flushleft}")
    frame, back = colors
    return (f"\\begin{{tcolorbox}}[colback={back}, colframe={frame}, coltitle=white,\n"
            f"    title={{{_escape(title)}}}, fonttitle=\\bfseries\\small, boxrule=1pt,\n"
            f"    arc=1mm, left=3mm, right=3mm, top=2mm, bottom=2mm]\n"
            f"{body}\n\\end{{tcolorbox}}")


def figure(boxes: list[str], caption: str, label: str) -> str:
    """One two-column figure per box, so a box that does not fit moves to the next page. Only the
    last carries the caption, which therefore sits under the last box and takes one number."""
    out = []
    for box in boxes[:-1]:
        out += ["\\begin{figure*}[t]", "\\centering", "\\small", box, "\\end{figure*}", ""]
    out += ["\\begin{figure*}[t]", "\\centering", "\\small",
            "\\floatconts", f"  {{{label}}}", f"  {{\\caption{{{caption}}}}}",
            "  {%", "  " + boxes[-1].replace("\n", "\n  "), "  }", "\\end{figure*}"]
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    # The local backend appends this instruction to the Interpreter's user turn.
    json_instruction = LocalLabeler._with_json_instruction(
        SimpleNamespace(n_keywords=DEFAULT_N_KEYWORDS), "").strip()
    interp = prompt_box("Interpreter Model Prompt", interp_system_prompt(DEFAULT_N_KEYWORDS),
                        PINK, INTERPRETER_EXAMPLES + "\n\n" + json_instruction,
                        labels=("System message", "User message"))
    max_new_tokens = inspect.signature(LocalLLM.generate_batch).parameters["max_new_tokens"].default
    interp_caption = ("The Interpreter prompt. Each example body is cut to its first "
                      f"{INTERPRETER_MAX_EXAMPLE_CHARS} characters; decoding is greedy, at most "
                      f"{max_new_tokens} new tokens.")
    print(save_text(figure([interp], interp_caption, "fig:interpreter-prompt"),
                    "prompt_interpreter", args.output_dir, dated=False))

    labels = {"tfidf": "TF-IDF terms", "llm": "LLM-written description",
              "snippets": "activating snippets"}
    # SCORE_INSTRUCTION is what the local backend appends to every user turn.
    user_tail = USER_PROMPT.format(neuron="<neuron id>", description="<description>",
                                   domain_note="<domain note>", text="<text>") + SCORE_INSTRUCTION
    ood = ("\\texttt{<domain note>} is empty for an in-domain text, and the following for an "
           "out-of-domain one: ``" + _inline(OOD_NOTE.strip()) + "''")
    letters = "abc"
    boxes = [prompt_box(f"({letters[i]}) Activation Prediction LLM Prompt -- {labels[t]}",
                        cls_system_prompt(t), PURPLE, user_tail)
             for i, t in enumerate(CLASS_DEFN)]
    caption = ("Activation Prediction LLM prompt, one per neuron interpretation: (a) "
               + ", (b) ".join([labels[t] for t in CLASS_DEFN][:2])
               + ", (c) " + labels[list(CLASS_DEFN)[2]] + ". " + ood)
    print(save_text(figure(boxes, caption, "fig:classifier-prompt"),
                    "prompt_classifier", args.output_dir, dated=False))


if __name__ == "__main__":
    main()
