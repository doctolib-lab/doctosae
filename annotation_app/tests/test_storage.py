"""Answer stores: CSV round trips, the latest row rule, formula protection and store selection."""

from __future__ import annotations

import csv
import sys
import threading
import types
from pathlib import Path

import pytest

from config import StorageSettings
from storage import HEADERS, CsvStore, SheetsStore, StorageConfigError, get_store, latest_answers

CASE = {"case_id": "C01", "rank": 3, "feature_id": 101}


def answer(store, score: int, comment: str = "", annotator: str = "A1", case: dict = CASE,
           position: int = 1) -> None:
    store.append_answer(annotator=annotator, case=case, score=score, comment=comment, position=position,
                        session_id="s1")


def read_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def test_header_written_once_and_row_layout(tmp_path):
    store = CsvStore(tmp_path / "answers.csv")
    answer(store, 3, "first")
    answer(store, 1, "second", position=2)
    rows = read_rows(store.path)
    assert rows[0] == HEADERS
    assert len(rows) == 3
    first = dict(zip(HEADERS, rows[1], strict=True))
    assert first["annotator"] == "A1" and first["case_id"] == "C01"
    assert (first["feature_rank"], first["feature_id"]) == ("3", "101")
    assert (first["label"], first["score"], first["position"], first["session_id"]) == ("1", "3", "1", "s1")
    assert dict(zip(HEADERS, rows[2], strict=True))["label"] == "0"


def test_round_trip_keeps_the_latest_row_per_annotator_and_case(tmp_path):
    store = CsvStore(tmp_path / "nested" / "answers.csv")  # missing folders are created
    assert store.load_answers("A1") == {}
    answer(store, 2, "unsure")
    answer(store, 4, "changed my mind")
    answer(store, 1, annotator="A2")
    answer(store, 3, case={"case_id": "C02", "rank": 5, "feature_id": 7})
    assert store.load_answers("A1") == {
        "C01": {"score": 4, "comment": "changed my mind"},
        "C02": {"score": 3, "comment": ""},
    }
    assert store.load_answers("A2") == {"C01": {"score": 1, "comment": ""}}
    assert store.load_answers("nobody") == {}


def test_rows_without_a_score_from_1_to_4_are_ignored():
    rows = [
        ["t", "A1", "C01", "1", "9", "1", "kept", "1", "s", "3"],
        ["t", "A1", "C02", "1", "9", "1", "short row"],
        ["t", "A1", "C03", "1", "9", "1", "out of range", "1", "s", "5"],
        ["t", "A1", "C01", "1", "9", "0", "later, but no score", "1", "s", ""],
    ]
    assert latest_answers(rows, "A1") == {"C01": {"score": 3, "comment": "kept"}}


@pytest.mark.parametrize("comment",
                         ["=1+1", "+33 6", "-2", "@SUM(A1)", "\tindent", "'quoted", "'=both", "plain"])
def test_formula_like_cells_are_protected_and_round_trip(tmp_path, comment):
    store = CsvStore(tmp_path / "answers.csv")
    answer(store, 3, comment)
    stored = read_rows(store.path)[1][HEADERS.index("comment")]
    assert stored == (comment if comment == "plain" else "'" + comment)
    assert store.load_answers("A1")["C01"]["comment"] == comment


def test_annotator_names_are_protected_too(tmp_path):
    store = CsvStore(tmp_path / "answers.csv")
    answer(store, 2, annotator="=A3")
    assert read_rows(store.path)[1][HEADERS.index("annotator")] == "'=A3"
    assert store.load_answers("=A3") == {"C01": {"score": 2, "comment": ""}}


def test_a_file_with_other_columns_is_refused(tmp_path):
    path = tmp_path / "answers.csv"
    path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected columns"):
        CsvStore(path).load_answers("A1")


def test_concurrent_appends_keep_every_row_whole(tmp_path):
    store = CsvStore(tmp_path / "answers.csv")

    def annotate(i: int) -> None:
        for j in range(20):
            case = {"case_id": f"C{j:02d}", "rank": j, "feature_id": j}
            answer(store, 1 + j % 4, annotator=f"A{i}", case=case)

    threads = [threading.Thread(target=annotate, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    rows = read_rows(store.path)
    assert rows.count(HEADERS) == 1
    assert len(rows) == 1 + 8 * 20
    assert all(len(row) == len(HEADERS) for row in rows)
    assert len(store.load_answers("A3")) == 20


# ── Store selection ──────────────────────────────────────────────────


def test_csv_is_the_default(tmp_path):
    store = get_store(StorageSettings(answers_csv=tmp_path / "a.csv"))
    assert isinstance(store, CsvStore) and store.label == "local CSV file"


def test_csv_mode_ignores_sheet_settings(tmp_path):
    settings = StorageSettings(mode="csv", answers_csv=tmp_path / "a.csv", spreadsheet_id="1AbC",
                               service_account_secret=True)
    assert isinstance(get_store(settings), CsvStore)


def test_machine_wide_credentials_alone_keep_csv(tmp_path):
    settings = StorageSettings(answers_csv=tmp_path / "a.csv", credentials_file="/elsewhere/key.json")
    assert isinstance(get_store(settings), CsvStore)


@pytest.mark.parametrize(("settings", "message"), [
    (StorageSettings(mode="sheets"), "needs SPREADSHEET_ID"),
    (StorageSettings(mode="sheets", spreadsheet_id="1AbC"), "credentials"),
    (StorageSettings(mode="auto", spreadsheet_id="1AbC"), "credentials"),
    (StorageSettings(mode="cloud"), "auto, csv or sheets"),
])
def test_misconfigured_storage_fails_closed(settings, message):
    with pytest.raises(StorageConfigError, match=message):
        get_store(settings)


# ── Google Sheets, with stand ins for gspread and google-auth ───────


class WorksheetNotFound(Exception):
    pass


class FakeWorksheet:
    def __init__(self) -> None:
        self.rows: list[list[str]] = []
        self.options: list[str] = []

    def row_values(self, index: int) -> list[str]:
        return self.rows[index - 1] if len(self.rows) >= index else []

    def update(self, range_name: str, values: list[list[str]]) -> None:
        assert range_name == "A1"
        self.rows[:1] = values

    def get_all_values(self) -> list[list[str]]:
        return [list(row) for row in self.rows]

    def append_row(self, row: list, value_input_option: str) -> None:
        self.options.append(value_input_option)
        self.rows.append([str(cell) for cell in row])


class FakeClient:
    def __init__(self) -> None:
        self.tabs: dict[str, FakeWorksheet] = {}
        self.created: list[tuple[str, int, int]] = []
        self.opened: list[str] = []
        self.credentials = None

    def authorize(self, credentials):
        self.credentials = credentials
        return self

    def open_by_key(self, key: str):
        self.opened.append(key)
        return self

    def worksheet(self, name: str) -> FakeWorksheet:
        if name not in self.tabs:
            raise WorksheetNotFound(name)
        return self.tabs[name]

    def add_worksheet(self, title: str, rows: int, cols: int) -> FakeWorksheet:
        self.created.append((title, rows, cols))
        self.tabs[title] = FakeWorksheet()
        return self.tabs[title]


class FakeCredentials:
    @classmethod
    def from_service_account_info(cls, info, scopes):
        return ("info", info, tuple(scopes))

    @classmethod
    def from_service_account_file(cls, path, scopes):
        return ("file", path, tuple(scopes))


@pytest.fixture
def google(monkeypatch) -> FakeClient:
    """Fake gspread and google-auth modules: no network, and the tests run without those packages."""
    client = FakeClient()
    gspread = types.ModuleType("gspread")
    gspread.WorksheetNotFound = WorksheetNotFound
    gspread.authorize = client.authorize
    service_account = types.ModuleType("google.oauth2.service_account")
    service_account.Credentials = FakeCredentials
    monkeypatch.setitem(sys.modules, "gspread", gspread)
    monkeypatch.setitem(sys.modules, "google.oauth2.service_account", service_account)
    return client


def test_sheets_selected_when_configured(google):
    key_file = "/outside/key.json"
    for settings in (StorageSettings(spreadsheet_id="1AbC", service_account_secret=True),
                     StorageSettings(mode="sheets", spreadsheet_id="1AbC", credentials_file=key_file)):
        store = get_store(settings)
        assert isinstance(store, SheetsStore) and store.label == "Google Sheet"
        assert "1AbC" not in repr(store)


def test_sheets_store_creates_the_tab_and_appends_raw_rows(google, fake_secrets):
    fake_secrets["gcp_service_account"] = {"project_id": "example"}
    store = SheetsStore("1AbC", "annotations")
    assert store.load_answers("A1") == {}
    assert google.created == [("annotations", 2000, len(HEADERS))]
    assert google.tabs["annotations"].rows == [HEADERS]
    assert google.credentials[:2] == ("info", {"project_id": "example"})
    answer(store, 2, "=1+1")
    answer(store, 4, "sure now")
    tab = google.tabs["annotations"]
    assert tab.options == ["RAW", "RAW"]
    assert tab.rows[1][HEADERS.index("comment")] == "=1+1"  # raw values need no apostrophe
    assert store.load_answers("A1") == {"C01": {"score": 4, "comment": "sure now"}}
    assert google.opened == ["1AbC"]  # the worksheet is opened once per store


def test_sheets_store_uses_the_key_file_without_a_secrets_table(google):
    store = SheetsStore("1AbC", "annotations", credentials_file="/outside/key.json")
    store.load_answers("A1")
    assert google.credentials[:2] == ("file", "/outside/key.json")


def test_sheets_store_rewrites_a_wrong_header(google, fake_secrets):
    fake_secrets["gcp_service_account"] = {"project_id": "example"}
    google.tabs["annotations"] = FakeWorksheet()
    google.tabs["annotations"].rows = [["old", "header"]]
    SheetsStore("1AbC", "annotations").load_answers("A1")
    assert google.tabs["annotations"].rows[0] == HEADERS
    assert google.created == []


def test_missing_google_packages_give_an_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "gspread", None)  # makes "import gspread" fail
    with pytest.raises(StorageConfigError, match="pip install gspread google-auth"):
        SheetsStore("1AbC", "annotations")
