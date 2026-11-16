"""Test setup: annotation_app/ goes on sys.path, and no setting leaks in from the machine running tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

SETTING_KEYS = ("APP_TITLE", "STUDY_NAME", "APP_PASSWORD", "ANNOTATORS", "CONTACT_EMAIL", "CASES_FILE",
                "STORAGE", "ANSWERS_CSV", "SPREADSHEET_ID", "WORKSHEET_NAME",
                "GOOGLE_APPLICATION_CREDENTIALS")


@pytest.fixture(autouse=True)
def fake_secrets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    """Stands in for st.secrets (tests add keys to it), so real secrets files are never read.

    Answers go to a temporary CSV file and a locally built data/cases.json is ignored, so every test starts
    from the demo configuration.
    """
    for key in SETTING_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ANSWERS_CSV", str(tmp_path / "answers.csv"))
    monkeypatch.setattr(config, "BUILT_CASES", tmp_path / "no_built_cases.json")
    secrets: dict = {}
    monkeypatch.setattr(config, "_secrets", lambda: secrets)
    return secrets
