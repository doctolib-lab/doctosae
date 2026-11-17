# Paper code: Explanations that Transfer

This folder holds the exact code behind the numbers of the ML4H 2026 paper ["Explanations that
Transfer: Interpreting Medical Retrieval with Sparse AutoEncoders"](https://arxiv.org/abs/2611.18179). A TopK sparse autoencoder (SAE)
is trained on the frozen dense embeddings of the CURE medical retrieval benchmark, with a
reconstruction loss plus an InfoNCE loss on the sparse codes. Retrieval then ranks documents by the
cosine similarity of their sparse codes, and each latent is interpreted from the texts that activate
it most, in domain and on five out of domain corpora.

These are the scripts that ran for the paper. The release only changed where data and runs live
(`DATA_ROOT`, relative run folders, data paths left out of run matching), defaults that pointed to
local copies of public models and datasets (now their Hub ids), and the documentation.

## Relation to the `doctosae` package

* [`../package`](../package) is the reusable library: a clean reimplementation of the same method with
  the paper defaults, a Python API and a command line tool. Start there for new work.
* This folder reruns the paper pipeline as it ran: hyperparameter grids, model selection, out of domain
  evaluation, interpretability pipeline and LaTeX tables.
* A run folder trained here becomes a package checkpoint with
  `doctosae convert <run folder> <output folder> --embedder bge` (see `doctosae convert --help`).

## Layout

| Path | Content |
| --- | --- |
| `scripts/sweep_cure_*.sh` | The training grids of the paper, one script per embedder or ablation |
| `scripts/eval_ood_cross_domain_sweep.sh` | Out of domain evaluation of the configuration selected for each k |
| `scripts/run_interpretability_pipeline.sh` | Interpretability pipeline on one run |
| `src/data_prep/` | Dataset download and BEIR conversion, query and corpus splits, embedding |
| `src/training/` | `train_sae.py` (one run) and `convert_embeddings.py` (sparse codes of any embedding pickle) |
| `src/evaluation/eval_ir.py` | NDCG@10 of one embedding column against a qrels file |
| `src/utils/` | Dataset registry (`datasets_constants.py`), model selection (`sweep_runs.py`), baselines, metrics, SAE classes |
| `src/analysis/` | LaTeX retrieval tables |
| `src/interpretability/` | Feature selection, Interpreter and Activation Prediction LLMs, tables and plots |

## Setup

Python 3.12 or newer. The setup below starts at the repository root and leaves you in this folder, where every
later command runs.

With [uv](https://docs.astral.sh/uv/), which installs the locked versions of the paper runs:

```bash
cd paper
uv sync
source .venv/bin/activate        # or prefix each python command with `uv run`
```

With pip:

```bash
cd paper
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
```

* The shell scripts in `scripts/` call `uv run`, so they need uv installed.
* `uv.lock` pins the library versions of the paper runs (for example `torch` 2.12.0, `numpy` 2.4.6,
  `scikit-learn` 1.9.0, `sentence-transformers` 5.5.1). The query and corpus splits come from
  `numpy.random.default_rng(42)`: keep the locked `numpy` to rebuild the same splits.
* Training, out of domain evaluation and the interpretability pipeline expect NVIDIA GPUs. The shell
  scripts expect Linux and bash 4.3 or newer (`wait -n`). The sweep scripts spread runs over GPUs 0 to 7,
  round robin: edit `GPUS` and `MAX` at their top to match your machine.
* `plot_2d_projection.py --method umap` also needs `pip install umap-learn`.

## Data location: `DATA_ROOT`

Every dataset lives under one folder, `DATA_ROOT`, which defaults to `~/data`. The dataset registry
`src/utils/datasets_constants.py` (baselines, out of domain evaluation, tables) and the sweep scripts
read it; the data preparation commands below write to it explicitly.

```bash
export DATA_ROOT=~/data
mkdir -p $DATA_ROOT
```

Layout once steps 1 and 2 are done (`<embedder>` is `bge-large-en-v1.5`, `MedEmbed-large-v0.1` or
`gemini-embedding-001`):

```text
$DATA_ROOT/
├── cure-beir/                corpus.jsonl, qrels/all.tsv, queries_{en,es,fr}.jsonl, corpus_{train,val}.jsonl
│   ├── queries_split/        queries_{train,val,test}_{en,es,fr}.jsonl, qrels_{train,val,test}.tsv
│   └── <embedder>/           corpus.pkl, corpus_{train,val}.pkl, queries_{train,val,test}_en.pkl
├── miriad-beir/              corpus_test.jsonl, queries_test.jsonl, qrels/test.tsv
│   └── <embedder>/           corpus_test.pkl, queries_test.pkl
├── NFCorpus/                 corpus.jsonl, queries.jsonl, qrels/test.tsv, queries_split/queries_test.jsonl
│   └── <embedder>/           corpus.pkl, queries_test.pkl
└── R2MED-beir/<subset>/      corpus.jsonl, query.jsonl, qrels.jsonl, qrels.tsv   (Biology, Bioinformatics, Medical-Sciences)
    └── <embedder>/           corpus.pkl, query.pkl
```

## Reproducing the paper

### 1. Prepare the data

CURE, the training set and in domain test set (`clinia/CUREv1`):

```bash
python -m src.data_prep.cure.prepare_data --output-dir $DATA_ROOT/cure-beir --languages en es fr

python -m src.data_prep.cure.split_multilang_queries \
  --queries-dir $DATA_ROOT/cure-beir --languages en es fr \
  --qrels $DATA_ROOT/cure-beir/qrels/all.tsv \
  --output-dir $DATA_ROOT/cure-beir/queries_split

python -m src.data_prep.split_corpus \
  --corpus-file $DATA_ROOT/cure-beir/corpus.jsonl --corpus-id-col _id \
  --output-dir $DATA_ROOT/cure-beir \
  --force-train-positives \
  --train-qrels $DATA_ROOT/cure-beir/queries_split/qrels_train.tsv \
  --train-queries $DATA_ROOT/cure-beir/queries_split/queries_train_en.jsonl \
  --query-id-col _id
```

This gives 244,600 passages, 1,400 training, 200 validation and 400 test queries (query ids are shared
across languages, the paper uses English only), and a corpus split of 223,425 training and 21,175
validation passages that puts every positive of a training query in the training split. Do not add
`--force-val-positives`: test positives never get that treatment.

MIRIAD (out of domain), from `tomaarsen/miriad-4.4M-split`:

```bash
python -m src.data_prep.miriad.prepare_data --output-dir $DATA_ROOT/miriad-beir
```

The paper evaluates the test files: 1,000 questions, each with one relevant passage, against a corpus
of those 1,000 passages plus 30,000 training passages. The first run downloads the whole dataset (about
12 GB, almost all of it the training split) although only the first rows of each split are used;
`--data-path` (or `MIRIAD_PATH`) points to a local copy instead.

NFCorpus (out of domain), the BEIR release
([`nfcorpus.zip`](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip), md5
`a89dba18a62ef92f7d323ec890a0d38d`; the Hub also hosts it as `BeIR/nfcorpus` and `BeIR/nfcorpus-qrels`):

```bash
curl -LO https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip
unzip nfcorpus.zip && mv nfcorpus $DATA_ROOT/NFCorpus
python -m src.data_prep.nfc.filter_queries_by_qrels \
  --queries $DATA_ROOT/NFCorpus/queries.jsonl --qrels-dir $DATA_ROOT/NFCorpus/qrels \
  --output-dir $DATA_ROOT/NFCorpus/queries_split
```

R2MED Biology, Bioinformatics and Medical Sciences (out of domain):

```bash
for S in Biology Bioinformatics Medical-Sciences; do
  hf download R2MED/$S corpus.jsonl query.jsonl qrels.jsonl --repo-type dataset \
    --local-dir $DATA_ROOT/R2MED-beir/$S
  python -m src.data_prep.r2med.convert_qrels --input $DATA_ROOT/R2MED-beir/$S/qrels.jsonl
done
```

### 2. Embed

Each embedder gets its own folder under every dataset. BGE (`BAAI/bge-large-en-v1.5`) encodes queries
with its retrieval instruction and documents without one; titles are prepended to the text when the
corpus has them.

```bash
E=bge-large-en-v1.5
M=BAAI/bge-large-en-v1.5
Q="Represent this sentence for searching relevant passages: "
doc()   { python -m src.data_prep.embed_texts_hf_models --model $M --mode document --input $1 --output $2; }
query() { python -m src.data_prep.embed_texts_hf_models --model $M --mode query --query-instruction "$Q" --input $1 --output $2; }

# CURE, then split the embedded corpus with the ids of step 1
doc $DATA_ROOT/cure-beir/corpus.jsonl $DATA_ROOT/cure-beir/$E/corpus.pkl
for s in train val test; do
  query $DATA_ROOT/cure-beir/queries_split/queries_${s}_en.jsonl $DATA_ROOT/cure-beir/$E/queries_${s}_en.pkl
done
python -m src.data_prep.split_corpus \
  --corpus-file $DATA_ROOT/cure-beir/$E/corpus.pkl --corpus-id-col _id \
  --ids-from $DATA_ROOT/cure-beir/corpus_train.jsonl $DATA_ROOT/cure-beir/corpus_val.jsonl \
  --output-dir $DATA_ROOT/cure-beir/$E

# out of domain datasets
doc   $DATA_ROOT/miriad-beir/corpus_test.jsonl  $DATA_ROOT/miriad-beir/$E/corpus_test.pkl
query $DATA_ROOT/miriad-beir/queries_test.jsonl $DATA_ROOT/miriad-beir/$E/queries_test.pkl
doc   $DATA_ROOT/NFCorpus/corpus.jsonl                    $DATA_ROOT/NFCorpus/$E/corpus.pkl
query $DATA_ROOT/NFCorpus/queries_split/queries_test.jsonl $DATA_ROOT/NFCorpus/$E/queries_test.pkl
for S in Biology Bioinformatics Medical-Sciences; do
  doc   $DATA_ROOT/R2MED-beir/$S/corpus.jsonl $DATA_ROOT/R2MED-beir/$S/$E/corpus.pkl
  query $DATA_ROOT/R2MED-beir/$S/query.jsonl  $DATA_ROOT/R2MED-beir/$S/$E/query.pkl
done
```

* MedEmbed: rerun the block with `E=MedEmbed-large-v0.1` and `M=abhinand/MedEmbed-large-v0.1`. The code
  does not record which query instruction the paper used for this model; without `--query-instruction`
  (`Q=""`) the script encodes the raw query.
* Gemini (`E=gemini-embedding-001`, d = 3072): same files and corpus split, embedded with
  `python -m src.data_prep.embed_texts --column text --input <file> --output <pkl>`, no instruction. It
  calls Vertex AI (billed) with Application Default Credentials and reads `GOOGLE_CLOUD_PROJECT` and
  `GOOGLE_CLOUD_LOCATION`; lower `--batch-size` for long passages such as MIRIAD to stay under the
  request token limit.
* Always pass `--model` to `embed_texts_hf_models.py`: its default is not a paper embedder.

### 3. Train

One run, with the flags shared by every paper run (here k = 64, m = 3072, λ = `1e-5`, τ = 0.05, seed 42).
The argparse defaults of `train_sae.py` are not the paper values, so keep every flag.

```bash
E=bge-large-en-v1.5
DATA=$DATA_ROOT/cure-beir/$E
QRELS=$DATA_ROOT/cure-beir/queries_split
python -m src.training.train_sae \
  --train-corpus  $DATA/corpus_train.pkl     --val-corpus  $DATA/corpus_val.pkl \
  --train-queries $DATA/queries_train_en.pkl --val-queries $DATA/queries_val_en.pkl \
  --test-queries  $DATA/queries_test_en.pkl \
  --train-qrels $QRELS/qrels_train.tsv --val-qrels $QRELS/qrels_val.tsv --test-qrels $QRELS/qrels_test.tsv \
  --corpus-id-col _id --query-id-col _id --score-col score \
  --sparsity-type topknorelu --activation-fn relu --init default \
  --query-recon-weight 0.1 --lr 5e-4 --batch 512 --epochs 100 --val-every-steps 100 \
  --k 64 --hidden 3072 --lambda 1e-5 --temperature 0.05 --seed 42 \
  --experiment-folder runs/repo_cure/$E/cure_k64_l1e-5_a0.1_lr5e-4_temp0.05_s42_h3072
```

`--k` is the number of active latents, `--hidden` the SAE width m, `--lambda` and `--temperature` the
InfoNCE weight and temperature, `--query-recon-weight` the weight of the query reconstruction term.
Validation scores the full corpus every 100 steps. The run folder holds:

| File | Content |
| --- | --- |
| `best_sae.pth` | Weights at the highest validation NDCG@10, the checkpoint behind every paper number |
| `best_sae_loss.pth` | Weights at the lowest validation loss, for reference only |
| `sae_config.json` | Architecture and loss hyperparameters |
| `summary.json` | `best_val_ndcg`, `test_ndcg` (the CURE cells of the tables) and the full run configuration |
| `training_log.txt`, `training_curves.png` | Training log and curves |

`DataParallel` turns on when several GPUs are visible; every paper run pinned one GPU with
`CUDA_VISIBLE_DEVICES`.

The paper grids, one script per embedder:

```bash
bash scripts/sweep_cure_topk_bge.sh
bash scripts/sweep_cure_topk_medembed.sh
bash scripts/sweep_cure_topk_gemini.sh
```

| Script | Embedder | k | m | Runs |
| --- | --- | --- | --- | --- |
| `sweep_cure_topk_bge.sh` | BGE | 64, 128, 256, 1024 | 3072, 9216, 21504 | 252 |
| `sweep_cure_topk_medembed.sh` | MedEmbed | 64, 128, 256, 1024 | 3072, 9216, 21504 | 252 |
| `sweep_cure_topk_gemini.sh` | Gemini | 64, 128, 256, 3072 | 9216, 27648 | 168 |
| `sweep_cure_topk_gemini_auxk.sh` | Gemini with the auxiliary loss (`--k-aux` = 2k, Appendix E) | 64, 128 | 9216, 27648 | 120 |
| `sweep_cure_nonsparse_unexpanded_bge.sh` | BGE, k = m = d | 1024 | 1024 | 21 |
| `sweep_cure_nonsparse_unexpanded_gemini.sh` | Gemini, k = m = d | 3072 | 3072 | 21 |
| `sweep_cure_kdm_medembed.sh` | MedEmbed, k = m = d | 1024 | 1024 | 21 |

Every grid crosses λ in {0, `5e-6`, `1e-5`} with τ in {0.01, 0.05, 0.1} (λ = 0 runs once, at τ = 0.1;
the auxiliary loss grid uses τ in {0.01, 0.05}, λ = 0 at τ = 0.01, and also crosses the auxiliary loss weight
`--aux-coef` in {0.25, 0.03125}, hence its 120 runs), with seeds 41, 42 and 43, query
weight 0.1, learning rate `5e-4`, batch size 512, 100 epochs and validation every 100 steps. A script
starts each run in the background under `nohup`,
at most `MAX` at once, skips runs whose `summary.json` exists (so it can be relaunched), writes logs and
job scripts to `logs/sae_cure/<embedder>/` and runs to `runs/repo_cure/<embedder>/cure_<tag>`, where the
tag encodes k, λ, the query weight, the learning rate, τ, the seed and m.
`src/utils/sweep_runs.py` reads each grid back from the script text, so edit a sweep script only through
the variables at its top (the comment there lists the constraints).

### 4. Select the configuration for each k

```bash
python -m src.utils.sweep_runs --sweep scripts/sweep_cure_topk_bge.sh --metric test_ndcg --per k
```

Runs are matched through their own `summary.json`. For each configuration (all hyperparameters but the
seed), the rule averages the validation NDCG@10 over the seeds, keeps every configuration whose 95%
confidence interval (t quantile) overlaps that of the best mean, and selects the smallest m among them
(Appendix A). The command prints one tab separated line per k: hyperparameters, validation and test
NDCG@10 (mean ± SEM), seeds and run folders. `--filter KEY VALUE` pins a hyperparameter, for example
`--filter lambda_ 0`.

### 5. Evaluate retrieval

* In domain: the CURE cells are `test_ndcg` of each `summary.json`, computed at the end of training
  with `best_sae.pth` for the 400 test queries against all 244,600 passages.
* Out of domain, for the configuration selected on CURE validation for each k:

  ```bash
  bash scripts/eval_ood_cross_domain_sweep.sh scripts/sweep_cure_topk_bge.sh test_ndcg
  ```

  For every seed of those configurations, the script encodes the corpus and queries of NFCorpus, MIRIAD
  and the three R2MED subsets with `best_sae.pth`, then writes `<run>/ood_eval/<dataset>/eval.json`. It
  reads the embeddings from the `<embedder>` folders of the layout above, and `<embedder>` defaults to
  the embedder folder of the sweep. Optional positional arguments, in this order (`""` keeps a default):
  a pinned m, another embedder folder, another runs folder, datasets to skip (comma separated) and
  pinned hyperparameters (`lambda_=1e-5,temperature=0.05`). Datasets run in parallel, one GPU each,
  with logs in `/tmp/ood_eval_<dataset>.log`.
* Original embeddings (the "original" rows):

  ```bash
  python -m src.utils.generate_baselines --embedder bge-large-en-v1.5 \
    --name CURE NFCorpus MIRIAD Bioinformatics Biology Medical-Sciences
  ```

  Scores are cached in `baselines/<embedder>.json` inside this folder; the table scripts compute
  missing ones on the fly.
* One run on one dataset, by hand. `convert_embeddings.py` writes `<name>_with_sae.pkl` (sparse codes in
  the `sae embeddings` column, reconstructions in `reconstructed embeddings`) under `<run>/converted/`:

  ```bash
  RUN=runs/repo_cure/bge-large-en-v1.5/cure_k64_l1e-5_a0.1_lr5e-4_temp0.05_s42_h3072
  for f in corpus_train corpus_val queries_train_en queries_test_en; do
    python -m src.training.convert_embeddings --experiment-folder $RUN \
      --emb-file $DATA_ROOT/cure-beir/bge-large-en-v1.5/$f.pkl
  done
  python -m src.evaluation.eval_ir --metric ndcg --k 10 --emb-type "sae embeddings" \
    --queries-file $RUN/converted/queries_test_en_with_sae.pkl --corpus-dir $RUN/converted \
    --qrels-file $DATA_ROOT/cure-beir/queries_split/qrels_test.tsv \
    --corpus-id-col _id --query-id-col _id --score-col score
  ```

  CURE needs both corpus splits (`--corpus-dir`), an out of domain corpus is one file (`--corpus-file`).
  `--emb-type embeddings` on the raw pickles scores the original embeddings. Id and grade columns are
  `_id`, `_id`, `score` for CURE and NFCorpus, `_id`, `_id`, `relevance` for MIRIAD and `id`, `id`, `score`
  for R2MED.

### 6. Build the retrieval tables

```bash
python -m src.analysis.latex_results_table \
  --sweep scripts/sweep_cure_topk_gemini.sh runs/repo_cure/gemini-embedding-001 \
  --sweep scripts/sweep_cure_topk_bge.sh runs/repo_cure/bge-large-en-v1.5 \
  --sweep scripts/sweep_cure_topk_medembed.sh runs/repo_cure/MedEmbed-large-v0.1 \
  --metric test_ndcg --datasets CURE NFCorpus MIRIAD Bioinformatics Biology Medical-Sciences \
  --percent --blue
python -m src.analysis.latex_grid_table \
  --sweep scripts/sweep_cure_topk_gemini.sh runs/repo_cure/gemini-embedding-001 \
  --sweep scripts/sweep_cure_topk_bge.sh runs/repo_cure/bge-large-en-v1.5 \
  --sweep scripts/sweep_cure_topk_medembed.sh runs/repo_cure/MedEmbed-large-v0.1 \
  --metric test_ndcg --datasets CURE NFCorpus MIRIAD Bioinformatics Biology Medical-Sciences \
  --percent
```

`latex_results_table.py` prints Table 1 (the selected configuration for each k); with
`--filter lambda_ 0` it prints the λ = 0 values behind Tables 7 and 8. `latex_grid_table.py` prints
Table 6 (every k and m). Run the tables after step 5: they read `summary.json`, the `eval.json` files and
the baselines.

### 7. Interpret

The paper interprets one run per embedder: BGE with k = 64 and m = 3072 (Tables 2 and 3, Figure 2), Gemini
with k = 128 and m = 9216 (Table 9) and MedEmbed with k = 64 and m = 3072 (Table 10). Both LLM roles use
`google/medgemma-4b-it`, run locally.

`google/medgemma-4b-it` is gated on the Hugging Face Hub: accept the Health AI Developer Foundations
terms on [its model page](https://huggingface.co/google/medgemma-4b-it), then log in once with
`hf auth login` on the machine that runs the pipeline. The `INTERPRETER_LLM` environment variable
replaces the default model id (a local folder works too).

The pipeline needs, for the run `RUN` (one of the run folders the out of domain script processed):
`<run>/converted/` with `corpus_train`, `corpus_val` and `queries_train_en` (the conversion loop of
step 5), and `<run>/ood_eval/<dataset>/` (step 5).

```bash
bash scripts/run_interpretability_pipeline.sh $RUN
```

It selects the features (`select_neurons.py`), labels them with the Interpreter LLM (`label_neurons.py`),
builds the evaluation texts in domain and for each out of domain corpus (`build_interpret_eval.py`), then
runs the Activation Prediction LLM on the three interpretation types (snippets, TF IDF terms, Interpreter
LLM output), one job per dataset and type spread over the GPUs (`predict_neurons_binary.py`). Outputs go
to `<run>/interpretation/`, and a step whose output exists is skipped. `--llm-model MODEL` changes the
Activation Prediction LLM only (outputs in `interpretation/judge_<model>/`); `--rank-band K_MIN K_MAX`
changes the activation ranks that make a text positive (default 1 20).

* Run alone, `select_neurons.py --entropy-theme corpus` needs `--theme-col` (CURE: `source_folder`;
  queries: `--query-theme-col discipline`). The converted `*_with_sae.pkl` keeps the text and theme
  columns of the embedder pickle, so `--corpus-file` is only needed for a pickle without them, and must
  then be a row aligned pickle.
* `label_neurons.py` and `predict_neurons_binary.py` run the LLM locally with Transformers
  (`--llm-backend local`, the paper setting) or call an OpenAI compatible server such as vLLM
  (`--llm-backend gateway --llm-base-url URL --llm-model NAME`), which sends the corpus text to that
  server. Give its key through the `LITELLM_PROXY_API_KEY` environment variable (`LITELLM_PROXY_API_BASE`
  can replace `--llm-base-url`) rather than `--llm-api-key`, which stays in the shell history.
* Importing `label_neurons.py` loads the first `.env` file found in its folder or a parent folder
  (`python-dotenv`).

### 8. Build the interpretability tables and figures

```bash
I=$RUN/interpretation
python -m src.interpretability.table_classification --pred-dir $I --in-domain CURE \
  --ood NFCorpus Bioinformatics Biology Medical-Sciences MIRIAD --output-dir figures
python -m src.interpretability.table_probability_by_rank --pred-dir $I --rank-bands 1-5 6-10 11-15 16-20 \
  --in-domain CURE --ood NFCorpus Bioinformatics Biology Medical-Sciences MIRIAD --output-dir figures
python -m src.interpretability.table_examples --pred-dir $I --in-domain CURE --ood Medical-Sciences \
  --compact --output-dir figures
python -m src.interpretability.latex_prompt_figures --output-dir figures
```

The other scripts of `src/interpretability/` are additional analyses, not in the paper: AUROC
(`table_auroc.py`), comparisons of Activation Prediction LLMs (`table_by_activation_prediction_llm.py`,
`table_llm_agreement.py`) and of models (`table_classification_comparison.py`), feature matching across
seeds (`match_neurons_across_seeds.py`) and firing rate and projection plots (`plot_*.py`). Each one
documents its usage in `--help`.

## Paper tables and figures

| Paper | Produced by |
| --- | --- |
| Table 1 | `src/analysis/latex_results_table.py`: CURE cells from `summary.json`, out of domain cells from `scripts/eval_ood_cross_domain_sweep.sh`, original rows from `src/utils/generate_baselines.py` |
| Table 2 | `src/interpretability/table_examples.py --compact` |
| Tables 3, 9, 10 | `src/interpretability/table_classification.py`, on the BGE, Gemini and MedEmbed runs |
| Table 4 | No script: sizes of the files built in step 1 |
| Table 5 | Not in this folder |
| Table 6 | `src/analysis/latex_grid_table.py` |
| Tables 7, 8 | `src/analysis/latex_results_table.py --filter lambda_ 0`, minus Table 1 and minus the original rows (the subtraction is not scripted) |
| Tables 11 to 13 | `src/interpretability/table_probability_by_rank.py` |
| Table 14 | Not in this folder |
| Table 15 | Not in this folder; its auxiliary loss runs come from `scripts/sweep_cure_topk_gemini_auxk.sh` |
| Figure 1 | Not generated by this code; the image is `../.github/assets/doctosae_overview.png` |
| Figures 2, 11, 12 | Not in this folder; the `doctosae` package implements the same decomposition (`Interpreter.explain`) |
| Figures 3 to 8 | Not in this folder |
| Figures 9, 10 | `src/interpretability/latex_prompt_figures.py` |

## Licenses

The code is under the Apache 2.0 license of the repository ([`../LICENSE`](../LICENSE)). The data and
models keep their own terms: check them before sharing anything derived from them (checkpoints, feature
labels, snippets).

| Dataset | Role | Source | License |
| --- | --- | --- | --- |
| CURE | Training, in domain test | `clinia/CUREv1` | `cc-by-nc-4.0` (noncommercial) |
| MIRIAD | Out of domain | `tomaarsen/miriad-4.4M-split`, a split of `miriad/miriad-4.4M` | Open Data Commons Attribution 1.0 (`odc-by`), subject to the OpenAI terms of use; research and education only, not for clinical decision making |
| NFCorpus | Out of domain | BEIR `nfcorpus.zip` (`BeIR/nfcorpus`) | `cc-by-sa-4.0` on the Hub; the original release is free for academic use only |
| R2MED Biology, Bioinformatics, Medical Sciences | Out of domain | `R2MED/Biology`, `R2MED/Bioinformatics`, `R2MED/Medical-Sciences` | `cc-by-4.0`; the queries are StackExchange posts (CC BY SA), attribute both |

| Model | Role | License |
| --- | --- | --- |
| `BAAI/bge-large-en-v1.5` | Embedder | MIT |
| `abhinand/MedEmbed-large-v0.1` | Embedder | Apache 2.0 |
| `gemini-embedding-001` | Embedder, through the Vertex AI API | Google Cloud terms |
| `google/medgemma-4b-it` | Interpreter and Activation Prediction LLM | Health AI Developer Foundations terms (gated) |

## Notes

* Results are reproducible up to the randomness of parallel computation (Appendix A of the paper).
* `summary.json` keeps the absolute input paths of a run in its configuration, and the interpretability
  files hold corpus text: review both before sharing a run folder.
* To cite the paper, see the [repository README](../README.md) and [`CITATION.cff`](../CITATION.cff).
