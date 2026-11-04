"""Streamlit AppTest runs of the app screens: no network, answers in a temporary CSV file, synthetic cases."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pytest

import config
from cases import SAMPLE_SEED
from storage import HEADERS

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"
EXAMPLES = json.loads(config.EXAMPLE_CASES.read_text(encoding="utf-8"))


def run_app():
    return AppTest.from_file(str(APP), default_timeout=30).run()


def button(at, label: str):
    return next(b for b in at.button if b.label == label)


def start(at, name: str = "A1"):
    at.text_input[0].input(name).run()
    return button(at, "Start").click().run()


def order_of(annotator: str) -> list[str]:
    """Case ids in the order the app shows them to this annotator."""
    order = list(range(len(EXAMPLES)))
    random.Random(f"{SAMPLE_SEED}:{annotator}").shuffle(order)
    return [EXAMPLES[i]["case_id"] for i in order]


def shown_case(at) -> str:
    """Case on screen, read from the keys of its score buttons."""
    return next(b.key.split("_", 1)[1] for b in at.button if b.key and b.key.startswith("score1_"))


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == HEADERS
    return [dict(zip(HEADERS, row, strict=True)) for row in rows[1:]]


def test_app_starts_without_configuration():
    at = run_app()
    assert not at.exception
    assert at.title[0].value == "MedSAE feature annotation"
    assert at.info[0].value.startswith("Demo mode: these are synthetic example cases.")
    assert any("Thank you for helping with this annotation study." in m.value for m in at.markdown)
    assert any("**6 cases**" in m.value and "about 3 minutes" in m.value for m in at.markdown)
    assert at.text_input[0].label == "Your name or annotator code"
    assert button(at, "Start").disabled
    assert not at.error


def test_answer_saved_on_click_then_resumed(tmp_path, monkeypatch):
    answers_csv = tmp_path / "answers.csv"
    monkeypatch.setenv("ANSWERS_CSV", str(answers_csv))
    at = start(run_app(), "A1")
    assert not at.exception
    first, second = order_of("A1")[:2]
    assert shown_case(at) == first
    assert any("passage-card" in m.value and "Synthetic example." in m.value for m in at.markdown)
    assert not any("Questions" in c.value for c in at.sidebar.caption)  # no contact configured
    assert "Answers are saved to: local CSV file" in [c.value for c in at.sidebar.caption]

    at.text_area[0].input("=looks right").run()
    button(at, "3 · Probably yes").click().run()
    rows = csv_rows(answers_csv)
    assert len(rows) == 1
    row = rows[0]
    assert (row["annotator"], row["case_id"], row["score"], row["label"]) == ("A1", first, "3", "1")
    assert row["comment"] == "'=looks right" and row["position"] == "1"
    assert shown_case(at) == second

    at.sidebar.button[0].click().run()  # Change annotator
    at = start(at, "A1")
    assert shown_case(at) == second  # resumes at the first unanswered case


def test_all_cases_answered_shows_the_done_screen(tmp_path, monkeypatch):
    monkeypatch.setenv("ANSWERS_CSV", str(tmp_path / "answers.csv"))
    at = start(run_app(), "Dr Who")
    for _ in EXAMPLES:
        button(at, "1 · Definitely not").click().run()
    assert not at.exception
    assert at.title[0].value == "Thank you!"
    assert any("labeled and saved, Dr." in m.value for m in at.markdown)
    assert len(csv_rows(tmp_path / "answers.csv")) == len(EXAMPLES)
    button(at, "Review my answers").click().run()
    assert shown_case(at) == order_of("Dr Who")[0]
    assert button(at, "1 · Definitely not").proto.type == "primary"  # the saved answer is highlighted
    assert button(at, "2 · Probably not").proto.type == "secondary"


def test_access_code_gate(fake_secrets):
    fake_secrets["APP_PASSWORD"] = "open sesame"
    at = run_app()
    assert [t.label for t in at.text_input] == ["Access code"]
    at.text_input[0].input("wrong").run()
    button(at, "Enter").click().run()
    assert at.error[0].value == "Wrong access code."
    at.text_input[0].input("open sesame").run()
    button(at, "Enter").click().run()
    assert not at.exception
    assert [t.label for t in at.text_input] == ["Your name or annotator code"]


def test_annotator_picker_study_name_and_contact(monkeypatch):
    monkeypatch.setenv("ANNOTATORS", "A1, A2")
    monkeypatch.setenv("STUDY_NAME", "the pilot study")
    monkeypatch.setenv("CONTACT_EMAIL", "study@example.org")
    at = run_app()
    assert any("Thank you for helping with the pilot study." in m.value for m in at.markdown)
    assert at.selectbox[0].options == ["A1", "A2"]
    assert button(at, "Start").disabled
    at.selectbox[0].select("A2").run()
    button(at, "Start").click().run()
    assert not at.exception
    assert shown_case(at) == order_of("A2")[0]
    assert any("study@example.org" in c.value for c in at.sidebar.caption)


def test_unreadable_secrets_stop_the_app_instead_of_dropping_the_access_code(monkeypatch):
    def broken_secrets():
        raise config.SettingsError("cannot read the Streamlit secrets: Error parsing secrets file")

    monkeypatch.setattr(config, "_secrets", broken_secrets)
    at = run_app()
    assert not at.exception
    assert at.error[0].value.startswith("cannot read the Streamlit secrets")
    assert not at.text_input and not at.button


def test_storage_misconfiguration_stops_the_app(monkeypatch):
    monkeypatch.setenv("STORAGE", "sheets")
    at = run_app()
    assert not at.exception
    assert "STORAGE=sheets needs SPREADSHEET_ID" in at.error[0].value
    assert not at.text_input


def test_invalid_cases_file_lists_every_problem(monkeypatch, tmp_path):
    bad = tmp_path / "cases.json"
    bad.write_text(json.dumps([
        {"case_id": "bad id!", "feature_id": 1, "rank": 1, "description": "d", "passage": "p"},
        {"case_id": "C02", "rank": "x", "description": "", "passage": "p"},
    ]), encoding="utf-8")
    monkeypatch.setenv("CASES_FILE", str(bad))
    at = run_app()
    assert not at.exception
    message = at.error[0].value
    for problem in ("case_id must be", "missing feature_id", "rank must be an integer",
                    "description must be"):
        assert problem in message


def test_unreadable_answers_keep_the_annotator_on_the_screen(monkeypatch, tmp_path):
    answers_csv = tmp_path / "answers.csv"
    answers_csv.write_text("a,b\n", encoding="utf-8")
    monkeypatch.setenv("ANSWERS_CSV", str(answers_csv))
    monkeypatch.setenv("CONTACT_EMAIL", "study@example.org")
    at = start(run_app())
    assert not at.exception
    assert at.error[0].value.startswith("Could not load your saved answers, please retry or contact "
                                        "study@example.org.")


def test_failed_save_stays_on_the_case(monkeypatch, tmp_path):
    monkeypatch.setenv("ANSWERS_CSV", str(tmp_path))  # a folder: every append fails
    at = start(run_app())
    case = shown_case(at)
    button(at, "4 · Definitely yes").click().run()
    assert not at.exception
    assert at.error[0].value.startswith("Your answer was NOT saved, please click again.\n\n")
    assert shown_case(at) == case
