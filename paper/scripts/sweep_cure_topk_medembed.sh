#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# The script runs from the paper/ folder wherever it is started: RUNS and LOG_DIR are relative to
# it, and DATA_ROOT defaults to ~/data (see README.md). src/utils/sweep_runs.py parses this file
# with regular expressions, so keep each assignment on one line, keep DATA ending in
# `cure-beir/<embedder>`, and keep RUNS a literal path (no variables) ending in the embedder
# directory.
DATA=${DATA_ROOT:-$HOME/data}/cure-beir/MedEmbed-large-v0.1
QRELS=${DATA_ROOT:-$HOME/data}/cure-beir/queries_split
RUNS=runs/repo_cure/MedEmbed-large-v0.1/cure
LOG_DIR=logs/sae_cure/MedEmbed-large-v0.1
mkdir -p "$LOG_DIR" "$(dirname "$RUNS")"

GPUS=(0 1 2 3 4 5 6 7)
NGPUS=${#GPUS[@]}
echo "Using GPUs: ${GPUS[*]}, runs distributed in turn"
BATCH=512
MAX=48
n=0

for K in 64 128 256 1024; do
  for LAM in 0 5e-6 1e-5; do
    for LR in 5e-4; do
      for TEMP in 0.01 0.05 0.1; do
        for SEED in 41 42 43; do
          for HIDDEN in 3072 9216 21504; do
            for ALPHA in 0.1; do
              [[ "$LAM" != 0 || "$TEMP" == "0.1" ]] || continue
              TAG="k${K}_l${LAM}_a${ALPHA}_lr${LR}_temp${TEMP}_s${SEED}_h${HIDDEN}"
              [[ -f "${RUNS}_${TAG}/summary.json" ]] && { echo "skip cure_${TAG} (already done)"; continue; }
              # Sliding window: keep exactly $MAX jobs in flight, and wait for any ONE to
              # finish (not the whole cohort) before launching the next, so a fast run
              # (small hidden) doesn't sit idle waiting on a slow batchmate (large hidden).
              while (( $(jobs -rp | wc -l) >= MAX )); do wait -n; done
              GPU_ID=${GPUS[$((n % NGPUS))]}
              # Each run is written to a job script and started under nohup, so it keeps
              # running when the terminal that launched the sweep closes.
              JOB_SH="$LOG_DIR/${TAG}.sh"
              cat > "$JOB_SH" <<EOF
#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=$GPU_ID uv run python -u -m src.training.train_sae \\
  --train-corpus  "$DATA/corpus_train.pkl" \\
  --val-corpus    "$DATA/corpus_val.pkl" \\
  --train-queries "$DATA/queries_train_en.pkl" \\
  --val-queries   "$DATA/queries_val_en.pkl" \\
  --test-queries  "$DATA/queries_test_en.pkl" \\
  --train-qrels   "$QRELS/qrels_train.tsv" \\
  --val-qrels     "$QRELS/qrels_val.tsv" \\
  --test-qrels    "$QRELS/qrels_test.tsv" \\
  --corpus-id-col _id --query-id-col _id --score-col score \\
  --sparsity-type topknorelu --activation-fn relu --init default \\
  --query-recon-weight "$ALPHA" \\
  --k "$K" --lambda "$LAM" --seed "$SEED" --lr "$LR" \\
  --temperature "$TEMP" --hidden "$HIDDEN" \\
  --batch "$BATCH" --epochs 100 --val-every-steps 100 \\
  --experiment-folder "${RUNS}_${TAG}" \\
  > "$LOG_DIR/${TAG}.log" 2>&1
EOF
              nohup bash "$JOB_SH" > /dev/null 2>&1 &
              echo "launched cure_${TAG} on GPU $GPU_ID (PID $!)"
              n=$((n+1))
            done
          done
        done
      done
    done
  done
done
wait
