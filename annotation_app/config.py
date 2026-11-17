"""Settings of the annotation app.

Each key is read from Streamlit secrets (`.streamlit/secrets.toml`, or the secrets of a hosted deployment),
then from an environment variable of the same name, then falls back to a default, so the app starts with no
configuration at all. Streamlit itself copies top level text and number secrets into the environment when it
starts, so secrets win either way; reading them first keeps lists such as ANNOTATORS on the same rule.

No default names a person, a spreadsheet, a cloud project or a contact.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
EXAMPLE_CASES = ROOT / "data" / "example_cases.json"
BUILT_CASES = ROOT / "data" / "cases.json"  # default output of scripts/build_cases.py, ignored by git

DEFAULT_TITLE = "DoctoSAE feature annotation"
STUDY_ANSWERS = "answers.csv"
DEMO_ANSWERS = "answers_demo.csv"
DEFAULT_WORKSHEET = "annotations"
STORAGE_MODES = ("auto", "csv", "sheets")
SERVICE_ACCOUNT_TABLE = "gcp_service_account"
# Streamlit marks a plain "no secrets file anywhere" with this id; any other secrets error is a broken file.
_NO_SECRETS_FILE = "no-secrets-found"


class SettingsError(RuntimeError):
    """The secrets exist but cannot be read; failing here keeps a broken file from dropping APP_PASSWORD."""


def _available(secrets: Mapping[str, Any]) -> Mapping[str, Any]:
    """`secrets` when a secrets file was found, an empty mapping when there is none.

    Streamlit raises StreamlitSecretNotFoundError, a FileNotFoundError, both when no file exists and when a
    file cannot be parsed. Only the first case means "not configured". Recent versions tell them apart by an
    error id; versions before 1.63 have none, but a file that fails to parse leaves its parsing error as the
    cause of the exception, while a missing file leaves no cause.
    """
    try:
        secrets.keys()
    except FileNotFoundError as exc:
        error_id = getattr(exc, "error_id", None)
        if exc.__cause__ is not None or error_id not in (None, _NO_SECRETS_FILE):
            raise SettingsError(f"cannot read the Streamlit secrets: {exc}") from exc
        return {}
    return secrets


def _secrets() -> Mapping[str, Any]:
    try:
        import streamlit as st
    except ImportError:  # scripts and some tests run without streamlit
        return {}
    return _available(st.secrets)


def setting(key: str, default: Any = "") -> Any:
    """Streamlit secrets first, then the environment, then `default`."""
    secrets = _secrets()
    if key in secrets:
        return secrets[key]
    return os.environ.get(key, default)


def has_secret_table(key: str) -> bool:
    """True when the secrets hold a TOML table under `key` (environment variables cannot carry tables)."""
    return isinstance(_secrets().get(key), Mapping)


def secret_table(key: str) -> dict[str, Any] | None:
    """A plain copy of the secrets table `key`, or None when there is no such table."""
    value = _secrets().get(key)
    return dict(value) if isinstance(value, Mapping) else None


def as_list(value: Any) -> list[str]:
    """A TOML list, or a comma separated string as environment variables carry it; blanks, repeats dropped."""
    items = value.split(",") if isinstance(value, str) else list(value or [])
    return list(dict.fromkeys(str(v).strip() for v in items if str(v).strip()))


def resolve(path: str | Path) -> Path:
    """Relative paths are taken from this folder, not from the working directory."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def default_cases_file() -> Path:
    """Cases built locally when they exist, else the synthetic examples (demo mode)."""
    return BUILT_CASES if BUILT_CASES.is_file() else EXAMPLE_CASES


def default_answers_file(cases_file: Path) -> str:
    """The demo writes to its own file, so that clicks on the synthetic cases never mix with a real study."""
    return DEMO_ANSWERS if cases_file.resolve() == EXAMPLE_CASES.resolve() else STUDY_ANSWERS


def _raw(value: Any) -> str:
    return "" if value is None else str(value)


def _text(value: Any) -> str:
    return _raw(value).strip()


@dataclass(frozen=True)
class StorageSettings:
    """Everything storage.get_store needs, and nothing else, so it can key a resource cache."""

    mode: str = "auto"
    answers_csv: Path = ROOT / STUDY_ANSWERS
    spreadsheet_id: str = ""
    worksheet_name: str = DEFAULT_WORKSHEET
    service_account_secret: bool = False  # a [gcp_service_account] table exists in the secrets
    credentials_file: str = ""  # GOOGLE_APPLICATION_CREDENTIALS, a key file kept outside the repo

    @property
    def has_credentials(self) -> bool:
        return self.service_account_secret or bool(self.credentials_file)


@dataclass(frozen=True)
class Settings:
    app_title: str = DEFAULT_TITLE
    study_name: str = ""
    app_password: str = field(default="", repr=False)
    annotators: tuple[str, ...] = ()
    contact_email: str = ""
    cases_file: Path = EXAMPLE_CASES
    storage: StorageSettings = field(default_factory=StorageSettings)

    @property
    def demo(self) -> bool:
        """True when the app shows the synthetic example cases."""
        return self.cases_file.resolve() == EXAMPLE_CASES.resolve()


def load_settings() -> Settings:
    """Read every setting now. The app calls this on each rerun, so edited secrets apply without a restart."""
    cases_text = _text(setting("CASES_FILE"))
    cases_file = resolve(cases_text) if cases_text else default_cases_file()
    storage = StorageSettings(
        mode=_text(setting("STORAGE")).lower() or "auto",
        answers_csv=resolve(_text(setting("ANSWERS_CSV")) or default_answers_file(cases_file)),
        spreadsheet_id=_text(setting("SPREADSHEET_ID")),
        worksheet_name=_text(setting("WORKSHEET_NAME")) or DEFAULT_WORKSHEET,
        service_account_secret=has_secret_table(SERVICE_ACCOUNT_TABLE),
        credentials_file=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip(),
    )
    return Settings(
        app_title=_text(setting("APP_TITLE")) or DEFAULT_TITLE,
        study_name=_text(setting("STUDY_NAME")),
        app_password=_raw(setting("APP_PASSWORD")),  # not stripped: compared byte for byte
        annotators=tuple(as_list(setting("ANNOTATORS", []))),
        contact_email=_text(setting("CONTACT_EMAIL")),
        cases_file=cases_file,
        storage=storage,
    )
