#!/usr/bin/env bash
# Interpretability pipeline on one run folder: select the units, label them, build the evaluation rows for
# CURE and for every dataset under ood_eval/, then run the Activation Prediction LLM on all three
# description types, one job per (dataset, type), spread over the GPUs. Every unit is kept.
#
# Needs <run>/converted/ (convert_embeddings.py; --use-original-dimensions for the original dimensions)
# and <run>/ood_eval/<dataset>/ (eval_ood_cross_domain_sweep.sh). Selection, labelling and the evaluation
# rows are skipped when their files exist, so another predictor scores the same rows.
#
# Usage: bash scripts/run_interpretability_pipeline.sh RUN_DIR [--rank-band K_MIN K_MAX] [--llm-model PATH]
#   --rank-band  positives are the texts where the unit ranks K_MIN..K_MAX (default 1 20); any other band
#                adds a _rank<K_MIN>-<K_MAX> suffix to the evaluation rows and predictions
#   --llm-model  Activation Prediction LLM (default: $INTERPRETER_LLM, else google/medgemma-4b-it, the
#                Interpreter LLM); its predictions go to interpretation/judge_<model name>/
set -euo pipefail
cd "$(dirname "$0")/.."

RUN=${1:?Usage: $0 RUN_DIR [--rank-band K_MIN K_MAX] [--llm-model PATH]}; shift
K_MIN=1; K_MAX=20; MODEL=""
while (( $# )); do
  case $1 in
    --rank-band) K_MIN=$2; K_MAX=$3; shift 3 ;;
    --llm-model) MODEL=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

I=$RUN/interpretation; OUT=$I; LLM_ARGS=""
[[ -n $MODEL ]] && { OUT=$I/judge_$(basename "$MODEL"); LLM_ARGS="--llm-model $MODEL"; }
BAND=""; [[ $K_MIN-$K_MAX != 1-20 ]] && BAND="_rank$K_MIN-$K_MAX"
OOD=$(ls "$RUN/ood_eval")
mkdir -p "$OUT"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

[[ -f $I/neuron_profiles.json ]] || uv run python -m src.interpretability.select_neurons \
  --sae-corpus-file "$RUN/converted/corpus_train_with_sae.pkl" \
  --queries-file "$RUN/converted/queries_train_en_with_sae.pkl" \
  --theme-col source_folder --query-theme-col discipline \
  --fr-min 0.0 --fr-max 1.0 --entropy-theme corpus --output "$I/neuron_profiles.json"
[[ -f $I/labels.json ]] || uv run python -u -m src.interpretability.label_neurons \
  --neuron-profiles "$I/neuron_profiles.json" --llm-backend local --batch-size 20 \
  --output "$I/labels.json" > "$I/label_neurons.log" 2>&1
log "units selected and labelled"

[[ -f $I/eval_rows$BAND.json ]] || uv run python -m src.interpretability.build_interpret_eval \
  --neuron-profiles "$I/neuron_profiles.json" \
  --eval-file "$RUN/converted/corpus_train_with_sae.pkl" "$RUN/converted/corpus_val_with_sae.pkl" \
  --rank-band "$K_MIN" "$K_MAX" --per-neuron 5 --n-negatives 5 --seed 42 --output "$I/eval_rows$BAND.json"
for D in $OOD; do
  EXCLUDE=$(echo $OOD | tr ' ' '\n' | grep -vx "$D" | tr '\n' ' ')
  [[ -f $I/eval_rows${BAND}_$D.json ]] || uv run python -m src.interpretability.build_interpret_eval \
    --neuron-profiles "$I/neuron_profiles.json" --ood-dir "$RUN/ood_eval" --dataset-name CURE \
    --exclude $EXCLUDE --rank-band "$K_MIN" "$K_MAX" --per-neuron 5 --n-negatives 5 --seed 42 \
    --output "$I/eval_rows${BAND}_$D.json"
done
log "evaluation rows built"

NGPU=$(nvidia-smi -L | wc -l)
JOBS=(); for D in CURE $OOD; do for T in llm tfidf snippets; do JOBS+=("$D:$T"); done; done
for g in $(seq 0 $((NGPU - 1))); do
  CMD=""
  for j in $(seq "$g" "$NGPU" $((${#JOBS[@]} - 1))); do
    D=${JOBS[$j]%%:*}; T=${JOBS[$j]##*:}; SUFFIX=$([[ $D = CURE ]] && echo "" || echo "_$D")
    LABELS=""; [[ $T = llm ]] && LABELS="--labels $I/labels.json"
    CMD+="uv run python -u -m src.interpretability.predict_neurons_binary --neuron-profiles $I/neuron_profiles.json $LABELS --description-type $T --eval-rows $I/eval_rows$BAND$SUFFIX.json --positives-per-neuron 3 --negatives-per-neuron 3 --seed 42 --llm-backend local --batch-size 8 $LLM_ARGS --output $OUT/pred_binary_$T$BAND$SUFFIX.json > $OUT/predict_$T$BAND$SUFFIX.log 2>&1; "
  done
  CUDA_VISIBLE_DEVICES=$g bash -c "$CMD" &
done
wait
log "predictions done"
