"""Where answers go: a local CSV file by default, or a Google Sheet when one is configured.

Both stores write the same 10 columns and append one row per click; the answer to a case is the latest row of
that (annotator, case_id) whose feature_id is the feature of that case today, so rows of another cases file with
the same case ids (the demo, an earlier build) never count. Stores raise on failure, so the app never moves on
after losing an answer.
"""

from __future__ import annotations

import csv
import os
import threading
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from config import SERVICE_ACCOUNT_TABLE, STORAGE_MODES, StorageSettings, secret_table

HEADERS = ["timestamp", "annotator", "case_id", "feature_rank", "feature_id",
           "label", "comment", "position", "session_id", "score"]
_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
_SCORES = ("1", "2", "3", "4")


class StorageConfigError(RuntimeError):
    """The storage settings cannot work; raised before any answer is collected."""


class AnswerStore(Protocol):
    label: str  # shown to annotators, never the spreadsheet id

    def load_answers(self, annotator: str, feature_ids: Mapping[str, str] | None = None) -> dict[str, dict]: ...

    def annotators(self) -> list[str]: ...

    def append_answer(self, *, annotator: str, case: dict, score: int, comment: str,
                      position: int, session_id: str) -> None: ...


def answer_row(*, annotator: str, case: dict, score: int, comment: str, position: int,
               session_id: str) -> list[Any]:
    """One stored answer. `label` collapses the 4 point score to binary: 3 or 4 means the feature reacts."""
    return [datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"), annotator, case["case_id"],
            case["rank"], case["feature_id"], int(score >= 3), comment, position, session_id, score]


def latest_answers(
    rows: Iterable[Sequence[str]], annotator: str, feature_ids: Mapping[str, str] | None = None
) -> dict[str, dict]:
    """Latest answer per case_id for this annotator.

    Short rows are padded, rows without a score from 1 to 4 are ignored (older rows predate the scale), and a
    later row replaces an earlier one, so changing an answer only ever appends. With `feature_ids` (case_id to
    the feature_id of the current cases, as strings), a row is kept only when it answered that very feature.
    """
    idx = {h: i for i, h in enumerate(HEADERS)}
    answers: dict[str, dict] = {}
    for row in rows:
        row = list(row) + [""] * (len(HEADERS) - len(row))
        if row[idx["annotator"]] != annotator or row[idx["score"]] not in _SCORES:
            continue
        case_id = row[idx["case_id"]]
        if feature_ids is not None and feature_ids.get(case_id) != row[idx["feature_id"]]:
            continue
        answers[case_id] = {"score": int(row[idx["score"]]), "comment": row[idx["comment"]]}
    return answers


def _distinct(names: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(name for name in names if name))


# Spreadsheet programs evaluate cells starting with these characters, so a comment typed as "=..." could run
# as a formula when answers.csv is opened. Such cells get a leading apostrophe; cells already starting with an
# apostrophe get one too, so reading strips exactly one and every string round trips.
_FORMULA = ("=", "+", "-", "@", "\t", "\r")
_QUOTE = "'"


def _protect(cell: Any) -> Any:
    if isinstance(cell, str) and cell.startswith((*_FORMULA, _QUOTE)):
        return _QUOTE + cell
    return cell


def _unprotect(cell: str) -> str:
    return cell[1:] if cell.startswith(_QUOTE) else cell


# Streamlit serves every session from threads of one process: one lock keeps rows and the header whole.
_LOCK = threading.Lock()


class CsvStore:
    """Answers in a local CSV file. Suits one machine and one Streamlit process."""

    label = "local CSV file"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _rows(self) -> list[list[str]]:
        """Data rows of the file, without the header; none when the file does not exist yet."""
        with _LOCK:
            if not self.path.is_file():
                return []
            with self.path.open(newline="", encoding="utf-8") as f:
                rows = [[_unprotect(c) for c in row] for row in csv.reader(f)]
        if rows and rows[0] != HEADERS:
            raise ValueError(f"{self.path.name} does not have the expected columns {HEADERS}")
        return rows[1:]

    def load_answers(self, annotator: str, feature_ids: Mapping[str, str] | None = None) -> dict[str, dict]:
        return latest_answers(self._rows(), annotator, feature_ids)

    def annotators(self) -> list[str]:
        """Annotator names as stored, in order of first answer."""
        return _distinct(row[HEADERS.index("annotator")] for row in self._rows() if len(row) > 1)

    def append_answer(self, **answer: Any) -> None:
        row = [_protect(c) for c in answer_row(**answer)]
        with _LOCK:
            new = not self.path.is_file() or self.path.stat().st_size == 0
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # A last row cut short (a killed process, an editor that dropped the final newline) must stay a row
            # of its own: appended to it, this answer would be lost and another one credited to its annotator.
            cut = not new and _last_byte(self.path) not in (b"\n", b"\r")
            with self.path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if new:
                    writer.writerow(HEADERS)
                elif cut:
                    f.write("\r\n")
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())  # the click is acknowledged only once the row is on disk


def _last_byte(path: Path) -> bytes:
    with path.open("rb") as f:
        f.seek(-1, os.SEEK_END)
        return f.read(1)


class SheetsStore:
    """Answers appended to one worksheet of a Google Sheet, as raw values (never parsed as formulas).

    Credentials come from the `[gcp_service_account]` secrets table, else from the key file that
    GOOGLE_APPLICATION_CREDENTIALS points to. There is no key file default: key files never live in the repo.
    """

    label = "Google Sheet"

    def __init__(self, spreadsheet_id: str, worksheet_name: str, credentials_file: str = "") -> None:
        try:  # imported here so the CSV mode runs without the Google packages
            import gspread
            from google.oauth2.service_account import Credentials
        except ImportError as exc:
            raise StorageConfigError(
                "Google Sheets storage needs two more packages: pip install gspread google-auth") from exc
        self._gspread = gspread
        self._credentials_cls = Credentials
        self._spreadsheet_id = spreadsheet_id
        self._worksheet_name = worksheet_name
        self._credentials_file = credentials_file
        self._worksheet: Any = None
        self._lock = threading.Lock()

    def __repr__(self) -> str:  # keeps the spreadsheet id out of logs and error pages
        return f"SheetsStore(worksheet={self._worksheet_name!r})"

    def _credentials(self) -> Any:
        info = secret_table(SERVICE_ACCOUNT_TABLE)
        if info is not None:
            return self._credentials_cls.from_service_account_info(info, scopes=_SCOPES)
        return self._credentials_cls.from_service_account_file(self._credentials_file, scopes=_SCOPES)

    def _sheet(self) -> Any:
        """The worksheet, opened once; created with the header row when missing.

        An empty first row gets the header, and so does a header of an earlier version of the app (a prefix of
        the current one, before the score column). A tab holding anything else is refused rather than
        overwritten: WORKSHEET_NAME then names another tab of the spreadsheet.
        """
        with self._lock:
            if self._worksheet is None:
                book = self._gspread.authorize(self._credentials()).open_by_key(self._spreadsheet_id)
                try:
                    ws = book.worksheet(self._worksheet_name)
                except self._gspread.WorksheetNotFound:
                    ws = book.add_worksheet(title=self._worksheet_name, rows=2000, cols=len(HEADERS))
                first = ws.row_values(1)
                if first and HEADERS[: len(first)] != first:
                    raise StorageConfigError(
                        f"the tab {self._worksheet_name!r} holds other data (its first row is not the answers "
                        "header): set WORKSHEET_NAME to another tab")
                if first != HEADERS:
                    ws.update(range_name="A1", values=[HEADERS])
                self._worksheet = ws
            return self._worksheet

    def load_answers(self, annotator: str, feature_ids: Mapping[str, str] | None = None) -> dict[str, dict]:
        return latest_answers(self._sheet().get_all_values()[1:], annotator, feature_ids)

    def annotators(self) -> list[str]:
        """Annotator names as stored, in order of first answer."""
        return _distinct(self._sheet().col_values(HEADERS.index("annotator") + 1)[1:])

    def append_answer(self, **answer: Any) -> None:
        self._sheet().append_row(answer_row(**answer), value_input_option="RAW")


def get_store(settings: StorageSettings) -> AnswerStore:
    """The store the settings ask for. Fails closed when a Google Sheet was asked for but cannot be used."""
    if settings.mode not in STORAGE_MODES:
        raise StorageConfigError(f"STORAGE must be auto, csv or sheets, got {settings.mode!r}")
    wants_sheets = settings.mode == "sheets" or (settings.mode == "auto" and bool(settings.spreadsheet_id))
    if wants_sheets:
        if not settings.spreadsheet_id:
            raise StorageConfigError("STORAGE=sheets needs SPREADSHEET_ID")
        if not settings.has_credentials:
            raise StorageConfigError(
                "Google Sheets storage needs service account credentials: a [gcp_service_account] table "
                "in .streamlit/secrets.toml, or GOOGLE_APPLICATION_CREDENTIALS pointing to a key file "
                "outside the repository")
        return SheetsStore(settings.spreadsheet_id, settings.worksheet_name, settings.credentials_file)
    return CsvStore(settings.answers_csv)
