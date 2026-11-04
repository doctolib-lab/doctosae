#!/usr/bin/env bash
# Run OOD eval (every dataset in src/utils/datasets_constants.py's OOD_DATASETS) for the best
# config per k from a sweep. Best configs are selected by the val metric from summary.json (no
# hardcoding): hidden is picked by that metric like any other hyperparameter unless H pins it.
# Dataset paths and id columns also come from datasets_constants.py, the single source of truth
# shared with the Python baseline code (src.utils.baselines), never duplicated here.
# The sweep's own training dataset is excluded on its own (see datasets_constants.ood_datasets_for);
# EXCLUDE lets you drop additional datasets on top of that.
# Parallelized by dataset. Results: <run_dir>/ood_eval/<dataset>/eval.json.
# The script runs from the paper/ folder wherever it is started; path arguments are read from the
# folder it is started in.
#
# Usage:
#   bash scripts/eval_ood_cross_domain_sweep.sh \
#       scripts/sweep_cure_topk_bge.sh test_ndcg
#   bash scripts/eval_ood_cross_domain_sweep.sh \
#       scripts/sweep_cure_topk_medembed.sh test_ndcg 3072 MedEmbed-large-v0.1
set -euo pipefail

SWEEP_SH="${1:?Usage: $0 <sweep.sh> <metric> [h] [emb_subdir] [runs_dir] [exclude] [pins]}"
METRIC="${2:?Usage: $0 <sweep.sh> <metric> [h] [emb_subdir] [runs_dir] [exclude] [pins]}"
H="${3:-}"                          # optional: pin to this hidden dim; omit to pick it by val metric
EMB_SUBDIR="${4:-}"                 # optional: embedder directory of the OOD data (default: the sweep's own)
RUNS_DIR="${5:-}"                   # optional override for runs_dir (default: read from sweep script)
EXCLUDE="${6:-}"                    # extra dataset names to skip, separated by commas, beyond the sweep's own
PINS="${7:-}"                       # key=value hyperparameters separated by commas, pinned for the
                                    # selection, e.g. "lambda_=1e-5,temperature=0.05"; anything
                                    # not pinned is still chosen by the val metric

# Relative path arguments name files from the starting folder: make them absolute, then move to paper/.
SWEEP_SH="$(cd "$(dirname "$SWEEP_SH")" && pwd)/$(basename "$SWEEP_SH")"
[[ -z $RUNS_DIR ]] || RUNS_DIR="$(cd "$RUNS_DIR" && pwd)"
cd "$(dirname "$0")/.."

_excluded() { [[ ",$EXCLUDE," == *",$1,"* ]]; }

# Get best run dirs for each (k, seed), and the sweep's own dataset (for self-exclusion), from Python
BEST_RUNS_AND_DATASET=$(uv run python -c "
import json, sys
from pathlib import Path
from src.utils.sweep_runs import (parse_sweep, load_runs, val_metric_from, best_per_value,
                                  filter_by_hparam, parse_hparam_value)
from src.utils.datasets_constants import dataset_from_sweep

grid = parse_sweep(Path('$SWEEP_SH'))
if '$RUNS_DIR':
    grid['runs_dir'] = Path('$RUNS_DIR')
data = load_runs(grid['runs_dir'], grid['swept_keys'], grid['fixed_config'], grid['swept_values'])
if '$H':
    data = filter_by_hparam(data, 'hidden', $H)
pins = {k: parse_hparam_value(v) for k, v in
        (kv.split('=', 1) for kv in '$PINS'.split(',') if kv)}
val_metric = val_metric_from('$METRIC')
best = best_per_value(data, val_metric, 'k', fixed=pins)

print('DATASET\t' + dataset_from_sweep(grid))
print('EMBEDDER\t' + Path(grid['train_corpus']).parent.name)
for k, sig in sorted(best.items()):
    for seed, summary in sorted(data[sig].items()):
        run_dir = summary['_run_dir']
        print(f'RUN\t{k}\t{seed}\t{run_dir}')
")

SWEEP_DATASET=$(echo "$BEST_RUNS_AND_DATASET" | awk -F'\t' '$1=="DATASET" {print $2}')
BEST_RUNS=$(echo "$BEST_RUNS_AND_DATASET" | awk -F'\t' '$1=="RUN" {print $2"\t"$3"\t"$4}')
[[ -n $EMB_SUBDIR ]] || EMB_SUBDIR=$(echo "$BEST_RUNS_AND_DATASET" | awk -F'\t' '$1=="EMBEDDER" {print $2}')

echo "Sweep's own dataset (self-excluded from OOD): $SWEEP_DATASET"
echo "Embedder directory of the OOD data: $EMB_SUBDIR"
echo "Best run dirs:"
echo "$BEST_RUNS"
echo ""

run_dataset() {
  local DSNAME="$1" CORP_CSV="$2" QRY="$3" QRELS="$4" IDCOL="$5" GPU="$6" SCORECOL="$7"
  echo "[gpu] $DSNAME -> CUDA_VISIBLE_DEVICES=$GPU"
  IFS=',' read -ra CORPS <<< "$CORP_CSV"
  echo "$BEST_RUNS" | while IFS=$'\t' read -r K SEED RUN_DIR; do
    local OOD_DIR="$RUN_DIR/ood_eval/$DSNAME"
    mkdir -p "$OOD_DIR"
    local CONV_CORPS=()
    for CORP in "${CORPS[@]}"; do
      local CONV_CORP="$OOD_DIR/$(basename "$CORP" .pkl)_with_sae.pkl"
      [[ -f "$CONV_CORP" ]] || CUDA_VISIBLE_DEVICES="$GPU" uv run python -m src.training.convert_embeddings \
        --emb-file "$CORP" --experiment-folder "$RUN_DIR" --output-subdir "ood_eval/$DSNAME"
      CONV_CORPS+=("$CONV_CORP")
    done
    local CONV_QRY="$OOD_DIR/$(basename "$QRY" .pkl)_with_sae.pkl"
    [[ -f "$CONV_QRY" ]] || CUDA_VISIBLE_DEVICES="$GPU" uv run python -m src.training.convert_embeddings \
      --emb-file "$QRY" --experiment-folder "$RUN_DIR" --output-subdir "ood_eval/$DSNAME"
    [[ -f "$OOD_DIR/eval.json" ]] && { echo "skip $DSNAME k=$K s=$SEED (exists)"; continue; }
    uv run python -m src.evaluation.eval_ir \
      --queries-file "$CONV_QRY" --corpus-file "${CONV_CORPS[@]}" --qrels-file "$QRELS" \
      --metric ndcg --k 10 --emb-type "sae embeddings" \
      --corpus-id-col "$IDCOL" --query-id-col "$IDCOL" --score-col "$SCORECOL" \
      --output "$OOD_DIR/eval.json"
    echo "done: $DSNAME k=$K s=$SEED"
  done
}

GPUS=(0 1 2 3 4 5 6 7)   # datasets take the GPUs in turn
g=0

# One line per OOD dataset (sweep's own dataset already excluded): name, corpus paths separated
# by commas, queries, qrels, id col (shared by corpus and queries), score col, from datasets_constants.py.
DATASET_LINES=$(uv run python -m src.utils.datasets_constants --embedder "$EMB_SUBDIR" --sweep-dataset "$SWEEP_DATASET")

while IFS=$'\t' read -r DSNAME CORP_CSV QRY QRELS IDCOL SCORECOL; do
  if _excluded "$DSNAME"; then
    echo "skipping $DSNAME (excluded)"
    continue
  fi
  MISSING=()
  IFS=',' read -ra CORPS <<< "$CORP_CSV"
  for p in "${CORPS[@]}" "$QRY" "$QRELS"; do
    [[ -f "$p" ]] || MISSING+=("$p")
  done
  if (( ${#MISSING[@]} > 0 )); then
    echo "skipping $DSNAME ($EMB_SUBDIR): no embeddings found, missing ${MISSING[*]}"
    continue
  fi
  # NOTE: assign GPU in the PARENT shell before `&`; a `g++` inside the argument list of a
  # backgrounded command runs in the child subshell and never increments the parent's g.
  gpu=${GPUS[$((g++ % ${#GPUS[@]}))]}
  run_dataset "$DSNAME" "$CORP_CSV" "$QRY" "$QRELS" "$IDCOL" "$gpu" "$SCORECOL" \
    > "/tmp/ood_eval_${DSNAME}.log" 2>&1 &
done <<< "$DATASET_LINES"

wait
echo ""
echo "ALL OOD EVALS DONE."
