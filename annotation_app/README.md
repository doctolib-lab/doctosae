# DoctoSAE annotation app

A Streamlit app to collect clinician judgments of the feature descriptions of *Explanations that Transfer:
Interpreting Medical Retrieval with Sparse AutoEncoders*. See the [repository README](../README.md) for the paper and
the `doctosae` package.

## What the study measures

Each case shows one SAE feature through its Interpreter LLM description (a topic, one or two sentences and
keywords) and one passage. The app calls features *neurons*, as the descriptions do, and explains the word in
plain language, so annotators need no background on sparse autoencoders. The annotator answers the question the
Activation Prediction LLM answers: *based on this description, would this neuron react to this passage?*

| score | answer | label |
| --- | --- | --- |
| 1 | Definitely not | 0 |
| 2 | Probably not | 0 |
| 3 | Probably yes | 1 |
| 4 | Definitely yes | 1 |

The scale has no midpoint. The hidden ground truth says whether the passage was a positive for that feature (the
feature is among the 20 strongest features active on the passage) or a negative (the feature does not fire on
it). Comparing answers with the ground truth measures how well a description lets a medical expert predict the
feature; comparing them with the judge's probability shows whether clinicians and the LLM judge agree. `label`
gives accuracy and Fleiss kappa, `score` gives AUROC and a graded comparison with `judge_probability`.

## Quickstart

Python 3.11 or later.

```bash
cd annotation_app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

With no configuration the app runs in demo mode on the six synthetic cases of `data/example_cases.json` (ids
`X01` to `X06`) and writes answers to `answers_demo.csv` next to `streamlit_app.py`, so that demo clicks never mix
with the answers of a real study.

## Configuration

Every setting is optional. Put it in `.streamlit/secrets.toml` (next to `streamlit_app.py`, or in the secrets of
a hosted app) or in an environment variable of the same name. Set each key in one place only: when both exist
the secrets value wins, because Streamlit copies top level secrets into the environment at startup.

| key | default | effect |
| --- | --- | --- |
| `APP_TITLE` | `DoctoSAE feature annotation` | page title and headings |
| `STUDY_NAME` | empty | adds "Thank you for helping with {STUDY_NAME}." to the introduction |
| `APP_PASSWORD` | empty | when set, a shared access code screen comes first |
| `ANNOTATORS` | empty | names or codes offered in a picker (a TOML list, or comma separated in the environment); empty means a free text field, recommended only for trials |
| `CONTACT_EMAIL` | empty | contact shown in the sidebar and in error messages; hidden when empty |
| `CASES_FILE` | `data/cases.json` if it exists, else `data/example_cases.json` | cases to annotate; relative paths start from this folder |
| `STORAGE` | `auto` | `auto`, `csv` or `sheets`, see below |
| `ANSWERS_CSV` | `answers.csv` (`answers_demo.csv` in demo mode) | CSV file used in csv mode; relative paths start from this folder |
| `SPREADSHEET_ID` | empty | Google Sheet that receives the answers in sheets mode |
| `WORKSHEET_NAME` | `annotations` | tab of that sheet, created when missing |
| `[gcp_service_account]` | absent | service account key fields, secrets only |
| `GOOGLE_APPLICATION_CREDENTIALS` | absent | environment only: path to a key file kept outside the repository |

```toml
# .streamlit/secrets.toml: ignored by git, never commit it
STUDY_NAME = "the feature annotation study"
APP_PASSWORD = "<shared access code>"
ANNOTATORS = ["A1", "A2", "A3", "A4", "A5"]
CONTACT_EMAIL = "<contact address>"
SPREADSHEET_ID = "<id from the sheet URL>"

[gcp_service_account]
type = "service_account"
project_id = "<your project id>"
private_key_id = "<key id>"
private_key = "<private key>"
client_email = "<service account address>"
token_uri = "https://oauth2.googleapis.com/token"
```

## Storage

* `csv` mode appends one row per click to `ANSWERS_CSV`. It suits one machine and one Streamlit process.
* `sheets` mode appends the same row to a Google Sheet, as raw values that are never parsed as formulas. The tab
  named by `WORKSHEET_NAME` is created when missing; an existing tab whose first row is not the answers header is
  refused, never overwritten.
* `auto` (the default) uses the Google Sheet when `SPREADSHEET_ID` is set and the local CSV file otherwise. A
  `SPREADSHEET_ID` without credentials, or `STORAGE = "sheets"` without both, stops the app with an error
  instead of silently writing somewhere else.

Hosted platforms such as Streamlit Community Cloud wipe the local disk on restart, so hosted studies should use
a Google Sheet:

1. Create a service account in your own Google Cloud project and create a JSON key for it. Keep the key file
   outside this repository.
2. Create a spreadsheet and share it with the service account address as Editor.
3. Copy the fields of the key into a `[gcp_service_account]` table of `.streamlit/secrets.toml` (or of the
   hosted app's secrets), or set `GOOGLE_APPLICATION_CREDENTIALS` to the path of the key file.
4. Set `SPREADSHEET_ID` to the long id in the sheet URL, and `WORKSHEET_NAME` if `annotations` does not suit.

The sidebar tells annotators where answers go ("local CSV file" or "Google Sheet"), never the spreadsheet id.

## Build real cases

`scripts/build_cases.py` turns interpretability outputs into `data/cases.json`, which the app then picks up on
its own. It imports neither streamlit nor `doctosae`. Keep its inputs and outputs under `data/` as below: that folder
is ignored by git, and so are `features/` and Markdown files at the root of this folder, because they hold corpus
passages and LLM outputs that this code only release must not carry. Give exactly one input:

```bash
# outputs of the doctosae package: doctosae interpret --llm ... --score --output data/features
python scripts/build_cases.py --cards data/features/cards.json --scores data/features/scores.json
# outputs of the paper code (paper/scripts/run_interpretability_pipeline.sh)
python scripts/build_cases.py --labels RUN/interpretation/labels.json \
    --predictions RUN/interpretation/pred_binary_llm.json
# a Markdown review file, for example one written earlier with --markdown data/top60.md
python scripts/build_cases.py --from-markdown data/top60.md
```

Each feature gives one case, with a passage whose label is drawn at random, so the default 30 balanced cases need
well over 30 features that have a description and judged passages of both labels. Run `doctosae interpret` without
`--n-features`, or with 60 or more.

Shared options: `--top 60` features kept by r, `--n-cases 30` cases drawn, `--seed 42`, `--max-chars 300`
characters shown per passage, `--out data/cases.json`, `--dataset NAME` for the review file title, and
`--markdown PATH` to also write the ranked features as a review file that `--from-markdown` reads back.

With the doctosae or paper inputs the script

1. keeps only the scores of the Interpreter LLM descriptions (`llm`) and refuses the shuffled description
   control;
2. drops features whose label failed (`LABEL_ERROR`, `PARSE_ERROR`) or whose description is empty, pairs the
   judge could not parse or whose request failed, and features judged on a single label;
3. ranks features by the Pearson r between the judge's probability and the ground truth (the stored value when
   the scores file has one, else computed as the paper does), ties broken by feature id, and keeps `--top`;
4. picks one judged passage per feature, positive or negative at random, and cuts it at `--max-chars`;
5. draws `--n-cases` features with a fresh `random.Random(seed)`, half with a positive passage and half with a
   negative one, and numbers them `C01`, `C02`, ... in rank order.

Input contracts:

* `--cards`: the `cards.json` of `doctosae interpret` (`save_cards`), or a plain JSON list of
  `FeatureCard.to_dict()`; the fields used are `feature`, `topic`, `description` and `keywords`.
* `--scores`: the `scores.json` of `doctosae interpret --score` (`ScoreReport.to_dict()`). It must include the
  `llm` method: the records of `predictions["llm"]` (`feature`, `row`, `source`, `text`, `ground_truth`,
  `score`, `parse_error`, and `error` for a failed request) are the judged pairs, and `per_feature["llm"]` gives
  the stored r.
* `--labels` and `--predictions`: the `labels.json` of `label_neurons.py` and the `pred_binary_llm.json` of
  `predict_neurons_binary.py`, unchanged.
* `--from-markdown`: rank, feature, description and label are read from the file; the draw is the same as step 5,
  so the same file and seed always give the same cases. Review files and cases are not part of this release,
  which ships code only.

Each case of the output holds `rank`, `feature_id`, `judge_input` (the exact text the judge read), `topic`,
`description`, `keywords`, `passage`, `ground_truth` (never shown) and `case_id`, plus `truncated`, `feature_r`,
`judge_probability` and `source` when known. The app checks every field when it loads the file and lists all
problems at once.

## Answers

Each click appends one row with these columns:

| column | content |
| --- | --- |
| `timestamp` | UTC time of the click |
| `annotator` | name or code entered on the first screen |
| `case_id` | case id from the cases file |
| `feature_rank`, `feature_id` | rank of the feature by r, and its SAE feature index |
| `label` | 0 or 1, the score collapsed to binary (3 or 4 gives 1) |
| `comment` | optional free text |
| `position` | position of the case in that annotator's order |
| `session_id` | random id of the browser session |
| `score` | 1 to 4 |

When an annotator comes back, the app reads their rows, restores the counter and jumps to the first unanswered
case. With the free text field, a name typed in another case or with other spaces is the annotator already
stored. Only rows whose `feature_id` is the feature of the case today count, so the rows of another cases file
with the same case ids never do; still, start a new answers file (or worksheet) whenever `data/cases.json` is
rebuilt. Changing an answer appends a new row, so for analysis keep the latest row per (annotator, case_id). The
ground truth is not stored with the answers: join on `case_id` and `feature_id` with the cases file. In the CSV
file a cell that
starts with `=`, `+`, `-`, `@`, a tab, a carriage return or an apostrophe gets one more leading apostrophe, so
that spreadsheet programs never run it as a formula; remove one leading apostrophe when analysing.

```python
import json

import pandas as pd

answers = pd.read_csv("answers.csv", dtype=str, keep_default_na=False)
answers = answers.apply(lambda column: column.str.replace(r"^'", "", regex=True))
answers = answers[answers["score"].isin(["1", "2", "3", "4"])]
latest = answers.groupby(["annotator", "case_id"]).tail(1)
cases = pd.DataFrame(json.load(open("data/cases.json", encoding="utf-8"))).astype({"feature_id": str})
columns = ["case_id", "feature_id", "ground_truth", "judge_probability"]
table = latest.merge(cases.filter(columns), on=["case_id", "feature_id"])
```

## Protocol notes

* Every annotator labels the same cases. Only the order changes: it is shuffled with a seed made of
  `SAMPLE_SEED` (42) and the annotator string, so it stays the same across visits.
* Every answer is saved as soon as the button is clicked, and the app only moves to the next case once the store
  has accepted the row.
* Clinicians see the first 300 characters of each passage, marked with "…" when cut. The Activation Prediction
  LLM saw up to 1000 characters of the same text (`--max-example-chars` of the paper code).

## Privacy

* Prefer annotator codes (A1, A2, ...) to names, set in `ANNOTATORS` for a real study: what is typed or picked is
  stored in every row.
* `answers.csv` holds personal data (annotator identifiers and free text comments). Every CSV file inside this
  folder is ignored by git, but a real study should rather set `ANSWERS_CSV` to an absolute path outside the
  repository, and keep the file out of shared places.
* `APP_PASSWORD` is one code shared by all annotators. It keeps casual visitors out; it does not authenticate
  people.
* Never commit `.streamlit/secrets.toml` or a service account key. JSON files at the root of this folder are
  ignored by git as a safety net, but key files belong outside the repository.

## Tests

```bash
pip install -r requirements.txt pytest ruff
pytest tests
ruff check .
```

The tests need no network and no credentials: answers go to temporary files and Google Sheets is replaced by
stand ins. Without streamlit installed, the Streamlit tests are skipped and the others still run.
