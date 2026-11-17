"""Settings: empty defaults, environment and secrets values, lists, paths and secrets file errors."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

import config
from config import ROOT, SettingsError, as_list, load_settings, resolve


def test_defaults_without_any_configuration(monkeypatch):
    monkeypatch.delenv("ANSWERS_CSV")
    settings = load_settings()
    assert settings.app_title == "DoctoSAE feature annotation"
    assert (settings.study_name, settings.app_password, settings.contact_email) == ("", "", "")
    assert settings.annotators == ()
    assert settings.cases_file == config.EXAMPLE_CASES and settings.demo
    storage = settings.storage
    assert (storage.mode, storage.answers_csv) == ("auto", ROOT / "answers_demo.csv")  # demo answers apart
    assert (storage.spreadsheet_id, storage.worksheet_name) == ("", "annotations")
    assert not storage.has_credentials


def test_environment_values_are_cleaned(monkeypatch):
    monkeypatch.setenv("APP_TITLE", "  Feature study  ")
    monkeypatch.setenv("ANNOTATORS", "A1, A2,,A1 , A3")
    monkeypatch.setenv("STORAGE", " CSV ")
    monkeypatch.setenv("CONTACT_EMAIL", "study@example.org")
    monkeypatch.setenv("WORKSHEET_NAME", "pilot")
    settings = load_settings()
    assert settings.app_title == "Feature study"
    assert settings.annotators == ("A1", "A2", "A3")
    assert settings.storage.mode == "csv"
    assert settings.contact_email == "study@example.org"
    assert settings.storage.worksheet_name == "pilot"


def test_secrets_come_first_and_carry_lists(monkeypatch, fake_secrets):
    monkeypatch.setenv("APP_PASSWORD", "from the environment")
    monkeypatch.setenv("ANNOTATORS", "E1,E2")
    fake_secrets.update({"APP_PASSWORD": 1234, "ANNOTATORS": ["S1", " S2 "], "STUDY_NAME": "the pilot study"})
    settings = load_settings()
    assert settings.app_password == "1234"
    assert settings.annotators == ("S1", "S2")
    assert settings.study_name == "the pilot study"


def test_blank_values_fall_back_to_defaults(monkeypatch):
    for key in ("APP_TITLE", "STORAGE", "WORKSHEET_NAME", "CASES_FILE", "ANSWERS_CSV"):
        monkeypatch.setenv(key, "  ")
    settings = load_settings()
    assert settings.app_title == config.DEFAULT_TITLE
    assert settings.storage.mode == "auto" and settings.storage.worksheet_name == "annotations"
    assert settings.cases_file == config.EXAMPLE_CASES
    assert settings.storage.answers_csv == ROOT / "answers_demo.csv"


def test_password_is_kept_as_typed_and_out_of_repr(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", " s3cret ")
    settings = load_settings()
    assert settings.app_password == " s3cret "
    assert "s3cret" not in repr(settings)


def test_relative_paths_start_from_the_app_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("CASES_FILE", "data/cases_v2.json")
    monkeypatch.setenv("ANSWERS_CSV", str(tmp_path / "out.csv"))
    settings = load_settings()
    assert settings.cases_file == ROOT / "data" / "cases_v2.json" and not settings.demo
    assert settings.storage.answers_csv == tmp_path / "out.csv"
    assert resolve("~/cases.json") == Path.home() / "cases.json"


def test_built_cases_replace_the_examples(monkeypatch, tmp_path):
    built = tmp_path / "cases.json"
    built.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(config, "BUILT_CASES", built)
    monkeypatch.delenv("ANSWERS_CSV")
    settings = load_settings()
    assert settings.cases_file == built and not settings.demo
    assert settings.storage.answers_csv == ROOT / "answers.csv"  # a real study keeps the study file


def test_service_account_table(fake_secrets):
    fake_secrets.update({"gcp_service_account": {"project_id": "example"}, "SPREADSHEET_ID": "1AbC"})
    storage = load_settings().storage
    assert storage.service_account_secret and storage.has_credentials and storage.spreadsheet_id == "1AbC"
    assert config.secret_table("gcp_service_account") == {"project_id": "example"}
    fake_secrets["gcp_service_account"] = "not a table"
    assert not config.has_secret_table("gcp_service_account")
    assert config.secret_table("gcp_service_account") is None


def test_key_file_path_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", " /outside/key.json ")
    storage = load_settings().storage
    assert storage.credentials_file == "/outside/key.json" and storage.has_credentials


def test_as_list():
    assert as_list("a, b,,a") == ["a", "b"]
    assert as_list(["x", " y ", "", 3]) == ["x", "y", "3"]
    assert as_list(None) == [] and as_list("") == []


# ── Secrets files, against the real Streamlit implementation ─────────


class _SecretsError:
    """A st.secrets stand in whose first access raises like Streamlit does."""

    def __init__(self, error_id: str | None, cause: Exception | None = None) -> None:
        self.error_id = error_id
        self.cause = cause

    def keys(self):
        error = FileNotFoundError("secrets")
        if self.error_id is not None:
            error.error_id = self.error_id
        raise error from self.cause


def test_only_a_missing_secrets_file_means_no_secrets():
    assert config._available(_SecretsError("no-secrets-found")) == {}
    assert config._available(_SecretsError(None)) == {}  # versions without error ids: no cause, no file
    assert config._available({"A": 1}) == {"A": 1}
    with pytest.raises(SettingsError, match="cannot read the Streamlit secrets"):
        config._available(_SecretsError("failed-parsing-secrets-file"))
    with pytest.raises(SettingsError, match="cannot read the Streamlit secrets"):  # before Streamlit 1.63
        config._available(_SecretsError(None, cause=ValueError("Unterminated string")))


def test_streamlit_secrets_files(monkeypatch, tmp_path):
    st_config = pytest.importorskip("streamlit.config")
    from streamlit.runtime.secrets import Secrets

    # Streamlit copies text secrets into the environment; registering the key restores it after the test.
    monkeypatch.setenv("DOCTOSAE_TEST_VALUE", "")
    path = tmp_path / "secrets.toml"
    previous = st_config.get_option("secrets.files")
    st_config.set_option("secrets.files", [str(path)])
    try:
        assert config._available(Secrets()) == {}
        path.write_text('DOCTOSAE_TEST_VALUE = "x"\n[table]\nkey = "value"\n', encoding="utf-8")
        secrets = config._available(Secrets())
        assert secrets["DOCTOSAE_TEST_VALUE"] == "x"
        assert isinstance(secrets["table"], Mapping) and dict(secrets["table"]) == {"key": "value"}
        path.write_text("this is = = not toml\n", encoding="utf-8")
        with pytest.raises(SettingsError):
            config._available(Secrets())
    finally:
        st_config.set_option("secrets.files", previous)
