#!/usr/bin/env bash

set -uo pipefail

PROJECT_ROOT="/data1/xxr_data/GNN/GNN-Task_Relation"
RESULT_DIR="$PROJECT_ROOT/exp/time_pipe/encode-gnn/results/offline_heat_260919234746"
PYTHON_BIN="/data1/xxr_data/new_conda/ofa/bin/python"
GPU="1"

TASKS=(cora_node cora_link pubmed_node pubmed_link arxiv WN18RR)

declare -A CHECKPOINTS=(
  [cora_node]="$PROJECT_ROOT/saved_exp/2026-09-17 17:25:07.738700/full_cdm/e5t7x7gs/checkpoints/epoch=74-step=150.ckpt"
  [cora_link]=""
  [pubmed_node]="$PROJECT_ROOT/saved_exp/2026-09-17 16:01:00.408337/full_cdm/tn7immv7/checkpoints/epoch=60-step=61.ckpt"
  [pubmed_link]=""
  [arxiv]=""
  [WN18RR]=""
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

printf 'task,phase,status,exit_code,duration_seconds,checkpoint,log\n' > "$RESULT_DIR/run_status.csv"
printf '%s\n' "$$" > "$RESULT_DIR/runner.pid"

timestamp() {
  date --iso-8601=seconds
}

cache_ready() {
  local cache_dir="$1"
  [[ -f "$cache_dir/texts.pkl" && -f "$cache_dir/geometric_data_processed.pt" ]]
}

record_status() {
  local task="$1"
  local phase="$2"
  local status="$3"
  local exit_code="$4"
  local duration="$5"
  local checkpoint="$6"
  local log_path="$7"
  printf '%s,%s,%s,%s,%s,%s,%s\n' \
    "$task" "$phase" "$status" "$exit_code" "$duration" \
    "${checkpoint:-none}" "$log_path" >> "$RESULT_DIR/run_status.csv"
}

run_cache_prep() {
  local task="$1"
  local log_path="$RESULT_DIR/${task}_cache_prep.log"
  local started
  local finished
  local exit_code

  started=$(date +%s)
  echo "[$(timestamp)] CACHE_PREP_START task=$task"
  CUDA_VISIBLE_DEVICES="$GPU" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" exp/time_pipe/encode-gnn/offline_encode_gnn_time.py \
      --split test \
      --loader-index 0 \
      --batch-num 1 \
      --warmup-batches 1 \
      --encode-warmup-batches 1 \
      --repeats 1 \
      --sampling-hops 2 \
      --skip-metric \
      --device cuda:0 \
      task_names "$task" \
      llm_name ST \
      llm_max_length 500 \
      batch_size 64 \
      llm_b_size 100 \
      train_sample_size -1 \
      num_workers 0 \
      > "$log_path" 2>&1
  exit_code=$?
  finished=$(date +%s)
  if [[ "$exit_code" -eq 0 ]]; then
    record_status "$task" cache_prep success "$exit_code" "$((finished - started))" "" "$log_path"
    echo "[$(timestamp)] CACHE_PREP_END task=$task status=success"
  else
    record_status "$task" cache_prep failed "$exit_code" "$((finished - started))" "" "$log_path"
    echo "[$(timestamp)] CACHE_PREP_END task=$task status=failed exit=$exit_code"
  fi
  return "$exit_code"
}

run_formal_profile() {
  local task="$1"
  local checkpoint="$2"
  local log_path="$RESULT_DIR/${task}.log"
  local started
  local finished
  local exit_code
  local -a command

  command=(
    "$PYTHON_BIN" exp/time_pipe/encode-gnn/offline_encode_gnn_time.py
    --split test
    --loader-index 0
    --batch-num -1
    --warmup-batches 5
    --encode-warmup-batches 1
    --repeats 5
    --sampling-hops 2
    --device cuda:0
  )
  if [[ -n "$checkpoint" ]]; then
    command+=(--checkpoint "$checkpoint")
  fi
  command+=(
    task_names "$task"
    llm_name ST
    llm_max_length 500
    batch_size 64
    llm_b_size 100
    train_sample_size -1
    num_workers 0
  )

  started=$(date +%s)
  echo "[$(timestamp)] PROFILE_START task=$task checkpoint=${checkpoint:-none}"
  CUDA_VISIBLE_DEVICES="$GPU" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
    "${command[@]}" > "$log_path" 2>&1
  exit_code=$?
  finished=$(date +%s)
  if [[ "$exit_code" -eq 0 ]]; then
    record_status "$task" profile success "$exit_code" "$((finished - started))" "$checkpoint" "$log_path"
    echo "[$(timestamp)] PROFILE_END task=$task status=success"
  else
    record_status "$task" profile failed "$exit_code" "$((finished - started))" "$checkpoint" "$log_path"
    echo "[$(timestamp)] PROFILE_END task=$task status=failed exit=$exit_code"
  fi
  return "$exit_code"
}

for task in "${TASKS[@]}"; do
  checkpoint="${CHECKPOINTS[$task]}"
  cache_dir="${CACHE_DIRS[$task]}"
  printf 'running task=%s phase=preflight updated_at=%s\n' "$task" "$(timestamp)" > "$RESULT_DIR/runner.state"

  if [[ -n "$checkpoint" && ! -f "$checkpoint" && ! -d "$checkpoint" ]]; then
    echo "[$(timestamp)] PREFLIGHT_FAILED task=$task missing_checkpoint=$checkpoint"
    record_status "$task" preflight failed 2 0 "$checkpoint" "none"
    continue
  fi

  if ! cache_ready "$cache_dir"; then
    printf 'running task=%s phase=cache_prep updated_at=%s\n' "$task" "$(timestamp)" > "$RESULT_DIR/runner.state"
    if ! run_cache_prep "$task"; then
      continue
    fi
    if ! cache_ready "$cache_dir"; then
      echo "[$(timestamp)] PREFLIGHT_FAILED task=$task cache_missing_after_prep=$cache_dir"
      record_status "$task" cache_check failed 3 0 "$checkpoint" "none"
      continue
    fi
  fi

  printf 'running task=%s phase=profile updated_at=%s\n' "$task" "$(timestamp)" > "$RESULT_DIR/runner.state"
  run_formal_profile "$task" "$checkpoint" || true
done

printf 'complete updated_at=%s\n' "$(timestamp)" > "$RESULT_DIR/runner.state"
echo "[$(timestamp)] ALL_TASKS_FINISHED result_dir=$RESULT_DIR"
