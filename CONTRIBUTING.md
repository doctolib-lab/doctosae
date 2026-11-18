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

[`doctosae`](https://pypi.org/project/doctosae/) is built and uploaded to PyPI from a maintainer's machine. The
`Publish` workflow is kept for the day Trusted Publishing can be enabled for the organisation, but it does not run
today. Every step needs [uv](https://docs.astral.sh/uv/).

1. Bump `__version__` in `package/src/doctosae/__init__.py` and `version` in `CITATION.cff`, and merge to `main`.
   PyPI takes each version once: a file can never be replaced, even after it is deleted.
2. Build from a fresh clone of `main`, never from a working copy: a local build packs whatever sits in
   `package/examples` or `package/tests`, run outputs included.

   ```bash
   RELEASE=$(mktemp -d) && git clone --depth 1 https://github.com/doctolib-lab/doctosae.git "$RELEASE/doctosae"
   cd "$RELEASE/doctosae/package"
   uv build --out-dir dist
   uvx twine check --strict dist/*
   ```

3. Check the wheel in an empty environment, then run the offline tests against it:

   ```bash
   uv run --isolated --no-project --with dist/*.whl doctosae --version
   uv run --isolated --no-project --with dist/*.whl --with pytest pytest -m "not network" -p no:cacheprovider
   ```

4. Upload with a PyPI API token scoped to the `doctosae` project (pypi.org, Account settings, API tokens). twine
   asks for it, or reads it from a `[pypi]` section of `~/.pypirc` with `username = __token__`. Never commit the
   token or paste it anywhere else.

   ```bash
   uvx twine upload dist/*
   ```

   To rehearse first, upload to TestPyPI with a token of a test.pypi.org account, a separate account:
   `uvx twine upload --repository-url https://test.pypi.org/legacy/ dist/*`.

5. Check the release from PyPI in an empty environment, then tag the commit it was built from:

   ```bash
   uv run --isolated --no-project --refresh-package doctosae --with "doctosae==<version>" doctosae --version
   git tag v<version> && git push origin v<version>
   ```

   Push the tag only, without a GitHub release: a release starts the `Publish` workflow, which would try to upload
   the same version again.

Checkpoints trained on CURE are pushed private first (`doctosae push` without `--public`) and made public only
once Legal confirms they may be: the CURE license excludes training models that may be used commercially.
