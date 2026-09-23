#!/usr/bin/env bash

set -uo pipefail

PROJECT_ROOT="/data1/xxr_data/GNN/GNN-Task_Relation"
RESULT_DIR="$PROJECT_ROOT/exp/encode/tf_profile/results/offline_tf_profile_260923005943"
PYTHON_BIN="/data1/xxr_data/new_conda/ofa/bin/python"
GPU="1"

TASKS=(cora_node cora_link pubmed_node pubmed_link arxiv WN18RR)

declare -A CHECKPOINTS=(
  [cora_node]="$PROJECT_ROOT/saved_exp/2026-09-17 17:25:07.738700/full_cdm/e5t7x7gs/checkpoints/epoch=74-step=150.ckpt"
  [cora_link]="$PROJECT_ROOT/saved_exp/2026-09-21 18:06:44.432551/full_cdm/8x52rwr9/checkpoints/epoch=48-step=3479.ckpt"
  [pubmed_node]="$PROJECT_ROOT/saved_exp/2026-09-17 16:01:00.408337/full_cdm/tn7immv7/checkpoints/epoch=60-step=61.ckpt"
  [pubmed_link]="$PROJECT_ROOT/saved_exp/2026-09-21 19:05:58.361032/full_cdm/b9rejg4r/checkpoints/epoch=10-step=6479.ckpt"
  [arxiv]="$PROJECT_ROOT/saved_exp/2026-09-21 20:44:10.956571/full_cdm/hk1d9kki/checkpoints/epoch=6-step=4977.ckpt"
  [WN18RR]="$PROJECT_ROOT/saved_exp/2026-09-22 12:45:33.439358/full_cdm/pyb5uw7f/checkpoints/epoch=32-step=22407.ckpt"
)

declare -A CACHE_DIRS=(
  [cora_node]="$PROJECT_ROOT/cache_data/Cora/ST/processed"
  [cora_link]="$PROJECT_ROOT/cache_data/Cora/ST/processed"
  [pubmed_node]="$PROJECT_ROOT/cache_data/Pubmed/ST/processed"
  [pubmed_link]="$PROJECT_ROOT/cache_data/Pubmed/ST/processed"
  [arxiv]="$PROJECT_ROOT/cache_data/arxiv/ST/processed"
  [WN18RR]="$PROJECT_ROOT/cache_data/WN18RR/ST/processed"
)

mkdir -p "$RESULT_DIR"
cd "$PROJECT_ROOT"

printf 'task,status,exit_code,duration_seconds,checkpoint,log\n' > "$RESULT_DIR/run_status.csv"
printf '%s\n' "$$" > "$RESULT_DIR/runner.pid"

timestamp() {
  date --iso-8601=seconds
}

record_status() {
  local task="$1"
  local status="$2"
  local exit_code="$3"
  local duration="$4"
  local checkpoint="$5"
  local log_path="$6"
  printf '%s,%s,%s,%s,%s,%s\n' \
    "$task" "$status" "$exit_code" "$duration" "$checkpoint" "$log_path" \
    >> "$RESULT_DIR/run_status.csv"
}

run_profile() {
  local task="$1"
  local checkpoint="$2"
  local log_path="$RESULT_DIR/${task}.log"
  local started
  local finished
  local exit_code

  started=$(date +%s)
  echo "[$(timestamp)] PROFILE_START task=$task checkpoint=$checkpoint"
  CUDA_VISIBLE_DEVICES="$GPU" \
  TOKENIZERS_PARALLELISM=false \
  PYTHONUNBUFFERED=1 \
  "$PYTHON_BIN" exp/time_pipe/encode-gnn/offline_encode_gnn_time.py \
    --checkpoint "$checkpoint" \
    --split test \
    --loader-index 0 \
    --batch-num -1 \
    --warmup-batches 5 \
    --encode-warmup-batches 1 \
    --repeats 5 \
    --sampling-hops 2 \
    --device cuda:0 \
    task_names "$task" \
    llm_name ST \
    llm_max_length 500 \
    batch_size 128 \
    llm_b_size 100 \
    num_workers 0 \
    > "$log_path" 2>&1
  exit_code=$?
  finished=$(date +%s)

  if [[ "$exit_code" -eq 0 ]] && grep -q '"transformer_profile"' "$log_path"; then
    record_status "$task" success "$exit_code" "$((finished - started))" "$checkpoint" "$log_path"
    echo "[$(timestamp)] PROFILE_END task=$task status=success"
  else
    if [[ "$exit_code" -eq 0 ]]; then
      exit_code=4
    fi
    record_status "$task" failed "$exit_code" "$((finished - started))" "$checkpoint" "$log_path"
    echo "[$(timestamp)] PROFILE_END task=$task status=failed exit=$exit_code"
  fi
}

for task in "${TASKS[@]}"; do
  checkpoint="${CHECKPOINTS[$task]}"
  cache_dir="${CACHE_DIRS[$task]}"
  printf 'running task=%s phase=preflight updated_at=%s\n' \
    "$task" "$(timestamp)" > "$RESULT_DIR/runner.state"

  if [[ ! -f "$checkpoint" && ! -d "$checkpoint" ]]; then
    echo "[$(timestamp)] PREFLIGHT_FAILED task=$task missing_checkpoint=$checkpoint"
    record_status "$task" preflight_failed 2 0 "$checkpoint" none
    continue
  fi
  if [[ ! -f "$cache_dir/texts.pkl" || ! -f "$cache_dir/geometric_data_processed.pt" ]]; then
    echo "[$(timestamp)] PREFLIGHT_FAILED task=$task missing_cache=$cache_dir"
    record_status "$task" preflight_failed 3 0 "$checkpoint" none
    continue
  fi

  printf 'running task=%s phase=profile updated_at=%s\n' \
    "$task" "$(timestamp)" > "$RESULT_DIR/runner.state"
  run_profile "$task" "$checkpoint"
done

printf 'complete updated_at=%s\n' "$(timestamp)" > "$RESULT_DIR/runner.state"
echo "[$(timestamp)] ALL_TASKS_FINISHED result_dir=$RESULT_DIR"
