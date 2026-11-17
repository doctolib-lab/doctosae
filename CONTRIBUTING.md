# Contributing to DoctoSAE

Thanks for your interest! Issues and pull requests are welcome.

## Where things live

| folder | what to change there |
| --- | --- |
| `package/` | the `doctosae` Python package: new features, bug fixes, docs, tests |
| `paper/` | the exact code behind the paper numbers; only fixes that keep results reproducible |
| `annotation_app/` | the Streamlit app to collect clinician judgments of feature descriptions |

## Development setup

```bash
git clone https://github.com/doctolib-lab/doctosae.git
cd doctosae/package
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,llm]"
```

## Before opening a pull request

```bash
cd package
ruff check src tests examples
ruff format --check src tests examples
pytest -m "not network"
```

Tests that download datasets or models from the Hugging Face Hub are marked `network`: the command above deselects
them, and a plain `pytest` runs them. Run them alone with `pytest -m network`; the CURE and MIRIAD tests also need
`DOCTOSAE_LARGE_DOWNLOADS=1` (about 300 MB to download each).

For changes to the annotation app, run its own checks from the repository root:

```bash
pip install -r annotation_app/requirements.txt pytest ruff
ruff check annotation_app
pytest annotation_app/tests
```

Please keep pull requests focused, add a test for every bug fix or new feature, and describe the motivation in the
pull request description.

## Never commit

* credentials of any kind (`.env` files, API keys, service account files, `.streamlit/secrets.toml`)
* datasets, embeddings or model checkpoints (publish checkpoints on the Hugging Face Hub instead)
* private or patient data

## Releasing (maintainers)

Preconditions of the first release, all of them:

* The ML4H 2026 decision allows the authors to be named. A GitHub release triggers the `Publish` workflow, and a
  manual run of it uploads to TestPyPI: both indexes are public, an uploaded file can never be replaced, and the
  package README names the authors next to the paper title. Until then, create no release and run no `Publish`
  workflow, TestPyPI included; the safest gate is to register the trusted publishers on PyPI and TestPyPI only
  then, and to add required reviewers to the `pypi` and `testpypi` GitHub environments.
* The repository is public. The PyPI page loads its figure from raw.githubusercontent.com and links to GitHub,
  which serve nothing of a private repository.
* The first upload happens when the repository becomes public, before any announcement: a pending trusted
  publisher does not reserve the name `doctosae`. Then drop the "not on PyPI yet" note of the README and show the
  PyPI badges again.
* Only the `Publish` workflow uploads, from a clean checkout. Never upload a distribution built locally: a local
  build packs whatever sits in `package/examples` or `package/tests`. The dry run stays local: `python -m build`,
  then `twine check --strict dist/*`.
* Checkpoints trained on CURE stay private (`doctosae push` without `--public`) until Legal confirms that they may
  be published: the CURE license excludes training models that may be used commercially.

Then:

1. Bump `__version__` in `package/src/doctosae/__init__.py` and `version` in `CITATION.cff`.
2. Merge to `main`, then create a GitHub release whose tag is `v<version>` (for example `v0.1.1`).
3. The `Publish` workflow builds the package and uploads it to PyPI through Trusted Publishing.
