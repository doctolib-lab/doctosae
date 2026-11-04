"""Annotation app: clinicians read one SAE feature description and one passage, then say whether the feature
would react to the passage. It is the question the Activation Prediction LLM answers, put to medical experts.

Run it with `streamlit run streamlit_app.py`. Settings come from config.py, answers go through storage.py and
the cases are validated by cases.py.
"""

from __future__ import annotations

import hmac
import html
import random
import re
import secrets

import streamlit as st

from cases import MINUTES_PER_CASE, PASSAGE_MAX_CHARS, SAMPLE_SEED, CasesError, load_cases
from config import Settings, SettingsError, StorageSettings, load_settings
from storage import AnswerStore, StorageConfigError, get_store

SCALE = ["1 · Definitely not", "2 · Probably not", "3 · Probably yes", "4 · Definitely yes"]

CSS = """
<style>
.block-container { max-width: 860px; padding-top: 4.5rem; }
.desc-card {
    background: #eef5fc; border-left: 4px solid #107aca; border-radius: 8px;
    padding: 0.9rem 1.1rem; margin: 0.4rem 0 1rem; line-height: 1.55;
}
.passage-card {
    background: #fafafa; border: 1px solid #e2e2e2; border-radius: 8px;
    padding: 0.9rem 1.1rem; margin: 0.4rem 0 1rem; line-height: 1.6; font-size: 0.95rem;
}
.desc-topic { font-weight: 700; font-size: 1.05rem; color: #0b4f86; margin-bottom: 0.3rem; }
.kw-row { margin-top: 0.6rem; }
.kw {
    display: inline-block; background: #ffffff; border: 1px solid #b7d3ec; color: #0b4f86;
    border-radius: 999px; padding: 0.05rem 0.6rem; margin: 0.15rem 0.25rem 0 0; font-size: 0.8rem;
}
.card-label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.8px;
    text-transform: uppercase; color: #5b6b7b;
}
#MainMenu, footer { visibility: hidden; }
</style>
"""

_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$])")


def md_escape(text: str) -> str:
    """Names typed by annotators are shown through Markdown; escaped, they render exactly as typed."""
    return _MARKDOWN_SPECIAL.sub(r"\\\1", text)


def one_line(text: str) -> str:
    """Collapse whitespace: a blank line inside an HTML card would end the card early in Markdown."""
    return " ".join(text.split())


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def run_in_frame(html_doc: str) -> None:
    """Run a small script in a same origin frame. Recent Streamlit replaces components.html, which it
    deprecates, with st.iframe; older releases in the supported range only have components.html."""
    if hasattr(st, "iframe"):
        st.iframe(html_doc, height=1)  # st.iframe refuses a zero height
    else:
        import streamlit.components.v1 as components

        components.html(html_doc, height=0)


@st.cache_data(show_spinner=False)
def cached_cases(path: str, mtime_ns: int) -> list[dict]:
    """Cached per file version (`mtime_ns` is in the key), so a rebuilt file applies without a restart."""
    return load_cases(path)


@st.cache_resource(show_spinner=False)
def answer_store(storage: StorageSettings) -> AnswerStore:
    """One store per storage configuration, shared by all sessions: opening a Google Sheet is slow."""
    return get_store(storage)


def stored_spelling(store: AnswerStore, name: str) -> str:
    """A typed name as the store already knows it: the same letters in another case, or with other spaces,
    are the same annotator, who keeps their answers and their order of cases."""
    name = " ".join(name.split())
    known = store.annotators()
    if name in known:
        return name
    return next((stored for stored in known if " ".join(stored.split()).casefold() == name.casefold()), name)


def case_order(annotator: str, n_cases: int) -> list[int]:
    """Same cases for everyone, in an order that is stable per annotator string."""
    order = list(range(n_cases))
    random.Random(f"{SAMPLE_SEED}:{annotator}").shuffle(order)
    return order


def next_unanswered(cases: list[dict], start: int) -> int | None:
    """First position from `start` (wrapping around) whose case has no answer yet; None when all are done."""
    order, answers = st.session_state["order"], st.session_state["answers"]
    for step in range(len(order)):
        p = (start + step) % len(order)
        if cases[order[p]]["case_id"] not in answers:
            return p
    return None


def intro(settings: Settings, n_cases: int) -> str:
    thanks = (f"Thank you for helping with {settings.study_name}." if settings.study_name
              else "Thank you for helping with this annotation study.")
    minutes = max(1, round(n_cases * MINUTES_PER_CASE))
    return (
        f"{thanks}\n\n"
        "**The idea.** Inside a medical AI search engine, we found thousands of *neurons*: small units "
        "that each react to a specific medical topic (not biological neurons). An AI read what each neuron "
        "reacts to and wrote a short description: a topic, one or two sentences, and keywords.\n\n"
        f"**Your task.** You will see **{plural(n_cases, 'case')}**. Each shows one neuron's description and "
        "one medical passage. From the description alone, would this neuron react to this passage? "
        "Answer on a 4 point scale, from *Definitely not* to *Definitely yes*.\n\n"
        "Use your medical knowledge: a neuron can react to a passage even when it does not use the "
        "description's words (synonyms, related procedures, underlying conditions).\n\n"
        f"It takes about {plural(minutes, 'minute')}. Every answer is saved as soon as you click, so you can "
        "stop at any time and come back later: you will resume where you left off."
    )


# ── Settings, cases and storage ──────────────────────────────────────

try:
    settings, settings_error = load_settings(), ""
except SettingsError as exc:
    settings, settings_error = Settings(), str(exc)

st.set_page_config(page_title=settings.app_title, layout="centered")
st.markdown(CSS, unsafe_allow_html=True)

if settings_error:
    st.error(settings_error)
    st.stop()

cases_file = settings.cases_file
try:
    CASES = cached_cases(str(cases_file), cases_file.stat().st_mtime_ns if cases_file.is_file() else 0)
except CasesError as exc:
    st.error("The cases file cannot be used:\n\n" + "\n".join(f"* {p}" for p in exc.problems))
    st.stop()
N_CASES = len(CASES)

try:
    store = answer_store(settings.storage)
except StorageConfigError as exc:
    st.error(f"Answer storage is not configured correctly: {exc}")
    st.stop()

for key, default in {
    "authenticated": not settings.app_password,
    "annotator": None,
    "order": [],
    "answers": {},
    "pos": 0,
    "session_id": secrets.token_hex(4),
}.items():
    st.session_state.setdefault(key, default)

# ── Access code ──────────────────────────────────────────────────────
# One code shared by all annotators: it keeps casual visitors out, it does not authenticate people.

if not st.session_state["authenticated"]:
    st.title(settings.app_title)
    code = st.text_input("Access code", type="password")
    if st.button("Enter", type="primary"):
        if hmac.compare_digest(code.encode(), settings.app_password.encode()):
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("Wrong access code.")
    st.stop()

# ── Identity ─────────────────────────────────────────────────────────

if st.session_state["annotator"] is None:
    st.title(settings.app_title)
    if settings.demo:
        st.info("Demo mode: these are synthetic example cases. Build real ones with scripts/build_cases.py.")
    st.markdown(intro(settings, N_CASES))
    if settings.annotators:
        name = st.selectbox("Who are you?", settings.annotators, index=None,
                            placeholder="Select your name or annotator code")
    else:
        name = st.text_input("Your name or annotator code", max_chars=64,
                             help="Use the same name or code each time you come back, so that you resume "
                                  "where you left off.").strip() or None
    if st.button("Start", type="primary", disabled=name is None):
        feature_ids = {c["case_id"]: str(c["feature_id"]) for c in CASES}
        try:
            if not settings.annotators:  # free text: "a1" or "A1 " is the annotator stored as "A1"
                name = stored_spelling(store, name)
            answers = store.load_answers(name, feature_ids)
        except Exception as exc:  # noqa: BLE001  any storage failure: say so and stay on this screen
            retry = "please retry"
            if settings.contact_email:
                retry += f" or contact {settings.contact_email}"
            st.error(f"Could not load your saved answers, {retry}.\n\n{exc}")
            st.stop()
        st.session_state["annotator"] = name
        st.session_state["answers"] = answers
        st.session_state["order"] = case_order(name, N_CASES)
        nxt = next_unanswered(CASES, 0)
        st.session_state["pos"] = 0 if nxt is None else nxt
        st.rerun()
    st.stop()

annotator = st.session_state["annotator"]
answers = st.session_state["answers"]
if len(st.session_state["order"]) != N_CASES:  # the cases file changed under a running session
    st.session_state["order"] = case_order(annotator, N_CASES)
    st.session_state["pos"] = next_unanswered(CASES, 0)
n_done = sum(1 for c in CASES if c["case_id"] in answers)

# ── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"**{md_escape(annotator)}**")
    st.progress(n_done / N_CASES, text=f"{n_done} / {N_CASES} cases labeled")
    if st.button("Change annotator"):
        for k in [k for k in st.session_state if k.startswith("comment_")]:
            del st.session_state[k]
        for k in ("annotator", "order", "answers", "pos"):
            del st.session_state[k]
        st.rerun()
    st.divider()
    st.caption(
        "Would the neuron **react** to the passage, given its description?  \n"
        "**1** Definitely not · **2** Probably not  \n"
        "**3** Probably yes · **4** Definitely yes  \n"
        f"Passages are cut at {PASSAGE_MAX_CHARS} characters, judge on what you see."
    )
    if settings.contact_email:
        st.caption(f"Questions: [{settings.contact_email}](mailto:{settings.contact_email})")
    st.caption(f"Answers are saved to: {store.label}")

# ── Done ─────────────────────────────────────────────────────────────

if st.session_state["pos"] is None:
    st.balloons()
    st.title("Thank you!")
    first = annotator.split()[0] if annotator.split() else annotator
    st.markdown(
        f"All **{plural(N_CASES, 'case')}** are labeled and saved, {md_escape(first)}. "
        "Your annotations go straight into the medical expert evaluation."
    )
    if st.button("Review my answers"):
        st.session_state["pos"] = 0
        st.rerun()
    st.stop()

# ── Annotation ───────────────────────────────────────────────────────

pos = st.session_state["pos"]
case = CASES[st.session_state["order"][pos]]
previous = answers.get(case["case_id"])

# Back to the top on every new case; the case id in the script makes it rerun. Safe to inline because
# cases.validate_cases restricts case ids to letters, digits and underscores.
run_in_frame(
    f"""<script>/* {case['case_id']} */
    const d = window.parent.document;
    for (const s of ['[data-testid="stMain"]', '[data-testid="stAppViewContainer"]', 'section.main'])
        d.querySelector(s)?.scrollTo({{top: 0}});
    window.parent.scrollTo(0, 0);
    </script>"""
)

st.progress(n_done / N_CASES, text=f"Case {pos + 1} of {N_CASES}  ·  {n_done} labeled")

kw_html = "".join(f'<span class="kw">{html.escape(one_line(k))}</span>' for k in case["keywords"])
topic_html = f'<div class="desc-topic">{html.escape(one_line(case["topic"]))}</div>' if case["topic"] else ""
st.markdown('<div class="card-label">Neuron description</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="desc-card">{topic_html}{html.escape(one_line(case["description"]))}'
    f'{f"<div class=kw-row>{kw_html}</div>" if kw_html else ""}</div>',
    unsafe_allow_html=True,
)
truncated = case.get("truncated")
if truncated is None:  # cases read from a Markdown review file carry no flag: a full length passage was cut
    truncated = len(case["passage"]) >= PASSAGE_MAX_CHARS
st.markdown('<div class="card-label">Passage</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="passage-card">{html.escape(one_line(case["passage"]))}{"…" if truncated else ""}</div>',
    unsafe_allow_html=True,
)

comment_key = f"comment_{case['case_id']}"
if comment_key not in st.session_state:
    st.session_state[comment_key] = previous["comment"] if previous else ""


def save(shown: dict, index: int, score: int) -> None:
    """Button callback: append the answer, and move on only once the store has accepted it."""
    comment = st.session_state[f"comment_{shown['case_id']}"].strip()
    try:
        store.append_answer(annotator=annotator, case=shown, score=score, comment=comment, position=index + 1,
                            session_id=st.session_state["session_id"])
    except Exception as exc:  # noqa: BLE001  any storage failure: keep the annotator on this case
        st.session_state["save_error"] = str(exc)
        return
    answers[shown["case_id"]] = {"score": score, "comment": comment}
    st.session_state["pos"] = next_unanswered(CASES, index + 1)


st.markdown("**Based on this description, would this neuron react to this passage?**")
st.text_area("Comment (optional, before answering)", key=comment_key, height=68,
             placeholder="Anything worth flagging?")

# Keys carry the case id so each case gets fresh buttons (no focus outline carried over)
for value, (col, label) in enumerate(zip(st.columns(4), SCALE, strict=True), start=1):
    with col:
        st.button(
            label,
            key=f"score{value}_{case['case_id']}",
            type="primary" if previous and previous["score"] == value else "secondary",
            width="stretch",
            on_click=save,
            kwargs={"shown": case, "index": pos, "score": value},
        )

if err := st.session_state.pop("save_error", None):
    help_line = f" If it keeps failing, contact {settings.contact_email}." if settings.contact_email else ""
    st.error(f"Your answer was NOT saved, please click again.{help_line}\n\n{err}")

if previous:
    st.caption(f"Saved answer: **{SCALE[previous['score'] - 1]}**. Click another option to change it.")

nav_prev, _, nav_next = st.columns([1, 2, 1])
with nav_prev:
    if st.button("← Previous", key=f"prev_{case['case_id']}", disabled=pos == 0, width="stretch"):
        st.session_state["pos"] = pos - 1
        st.rerun()
with nav_next:
    if st.button("Skip →", key=f"skip_{case['case_id']}", disabled=pos == N_CASES - 1, width="stretch"):
        st.session_state["pos"] = pos + 1
        st.rerun()
