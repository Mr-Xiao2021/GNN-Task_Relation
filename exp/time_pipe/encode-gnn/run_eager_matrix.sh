#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
OFA_PREFIX=/data1/xxr_data/new_conda/ofa
CONDA_BIN=/home/xxr/miniconda3/bin/conda
OFA_PYTHON="$OFA_PREFIX/bin/python"
RESULT_DIR=${1:-"$SCRIPT_DIR/results/eager_$(date +%y%m%d%H%M%S)"}
STATUS_CSV="$RESULT_DIR/run_status.csv"

mkdir -p "$RESULT_DIR"

finish() {
  exit_code=$?
  if [ "$exit_code" -eq 0 ]; then
    state=success
  else
    state=failed
  fi
  printf 'state=%s\nexit_code=%s\nfinished_at=%s\n' \
    "$state" "$exit_code" "$(date --iso-8601=seconds)" > "$RESULT_DIR/runner.status"
}
trap finish EXIT

printf 'task,batch_num,status,elapsed_seconds,log_file\n' > "$STATUS_CSV"
printf 'started_at=%s\nresult_dir=%s\n' \
  "$(date --iso-8601=seconds)" "$RESULT_DIR" > "$RESULT_DIR/runner.info"

tasks=(cora_node pubmed_node wikics)
batch_nums=(1 5 10)
failed_runs=0

cd "$PROJECT_ROOT"

for task in "${tasks[@]}"; do
  case "$task" in
    cora_node)
      checkpoint="$PROJECT_ROOT/saved_exp/2026-09-17 17:25:07.738700/full_cdm/e5t7x7gs/checkpoints/epoch=74-step=150.ckpt"
      ;;
    pubmed_node)
      checkpoint="$PROJECT_ROOT/saved_exp/2026-09-17 16:01:00.408337/full_cdm/tn7immv7/checkpoints/epoch=60-step=61.ckpt"
      ;;
    wikics)
      checkpoint="$PROJECT_ROOT/saved_exp/2026-09-17 16:16:11.509321/full_cdm/e5b9gd7e/checkpoints/epoch=71-step=288.ckpt"
      ;;
  esac

  for batch_num in "${batch_nums[@]}"; do
    log_file="$RESULT_DIR/${task}_batch_${batch_num}.log"
    start_seconds=$(date +%s)
    printf '[%s] START task=%s batch_num=%s\n' \
      "$(date --iso-8601=seconds)" "$task" "$batch_num"

    if CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
      "$CONDA_BIN" run --no-capture-output -p "$OFA_PREFIX" \
      python exp/time_pipe/encode-gnn/eager_encode_gnn_time.py \
      --checkpoint "$checkpoint" \
      --split test \
      --loader-index 0 \
      --batch-num "$batch_num" \
      --warmup-batches 1 \
      --device cuda:0 \
      task_names "$task" \
      llm_name ST \
      llm_max_length 500 \
      batch_size 128 \
      llm_b_size 100 \
      num_workers 4 \
      llm_peft false \
      llm_quantization false \
      llm_trainable false \
      > "$log_file" 2>&1; then
      run_status=success
    else
      run_status=failed
      failed_runs=$((failed_runs + 1))
    fi

    elapsed_seconds=$(($(date +%s) - start_seconds))
    printf '%s,%s,%s,%s,%s\n' \
      "$task" "$batch_num" "$run_status" "$elapsed_seconds" "$log_file" \
      >> "$STATUS_CSV"
    printf '[%s] END task=%s batch_num=%s status=%s elapsed=%ss\n' \
      "$(date --iso-8601=seconds)" "$task" "$batch_num" \
      "$run_status" "$elapsed_seconds"
  done
done

if [ "$failed_runs" -ne 0 ]; then
  printf '%s run(s) failed; see %s\n' "$failed_runs" "$STATUS_CSV" >&2
  exit 1
fi

"$OFA_PYTHON" "$SCRIPT_DIR/summarize_eager_results.py" \
  --results-dir "$RESULT_DIR"

printf '[%s] COMPLETE results=%s\n' "$(date --iso-8601=seconds)" "$RESULT_DIR"
