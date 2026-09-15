#!/usr/bin/env bash
set -euo pipefail

SUPPORTED_TASKS=(cora_node pubmed_node arxiv wikics)

usage() {
  cat <<'EOF'
Usage:
  scripts/run_e2e_node_npu.sh <task_name> [epochs] [config_key config_value ...]

Supported tasks:
  cora_node    Cora node classification (local raw data included)
  pubmed_node  PubMed node classification (local raw data included)
  arxiv        OGBN-Arxiv node classification (downloads data on first run)
  wikics       WikiCS node classification (downloads data on first run)

Examples:
  scripts/run_e2e_node_npu.sh cora_node 200
  scripts/run_e2e_node_npu.sh pubmed_node 200 batch_size 64 num_workers 4
  ASCEND_RT_VISIBLE_DEVICES=1 scripts/run_e2e_node_npu.sh wikics 100
  WANDB_ONLINE=1 WANDB_PROJECT=gnn-task-relation scripts/run_e2e_node_npu.sh cora_node 200
  DRY_RUN=1 scripts/run_e2e_node_npu.sh arxiv 200

The best validation checkpoint and last checkpoint are saved under:
  saved_exp/<run timestamp>/checkpoints/

Set WANDB_ONLINE=1 after `wandb login` to upload metrics and checkpoints.
EOF
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  usage
  exit 0
fi

if [[ ${1:-} == "--list" ]]; then
  printf '%s\n' "${SUPPORTED_TASKS[@]}"
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

TASK_NAME=$1
shift

case " ${SUPPORTED_TASKS[*]} " in
  *" ${TASK_NAME} "*) ;;
  *)
    printf 'Unsupported e2e_node task: %s\n\n' "${TASK_NAME}" >&2
    usage >&2
    exit 2
    ;;
esac

EPOCHS=${EPOCHS:-200}
if [[ ${1:-} =~ ^[0-9]+$ ]]; then
  EPOCHS=$1
  shift
fi

if (( $# % 2 != 0 )); then
  printf 'Additional run_cdm options must be config_key/config_value pairs.\n' >&2
  exit 2
fi

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONDA_ROOT=${CONDA_ROOT:-/train08_data/user4/miniconda3}
CONDA_ENV=${CONDA_ENV:-gnn}
CANN_ROOT=${CANN_ROOT:-/usr/local/Ascend/cann-9.1.0-beta.3}

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
source "${CANN_ROOT}/set_env.sh"

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
export TORCH_NPU_DEVICE_CAPABILITY=${TORCH_NPU_DEVICE_CAPABILITY:-8.0}

WANDB_ARGS=()
if [[ ${WANDB_ONLINE:-0} == "1" ]]; then
  export WANDB_MODE=online
  WANDB_ARGS+=(offline_log false wandb_log_model true)
else
  export WANDB_MODE=${WANDB_MODE:-offline}
fi
if [[ -n ${WANDB_PROJECT:-} ]]; then
  WANDB_ARGS+=(log_project "${WANDB_PROJECT}")
fi
if [[ -n ${WANDB_ENTITY:-} ]]; then
  WANDB_ARGS+=(wandb_entity "${WANDB_ENTITY}")
fi

cd "${PROJECT_DIR}"

COMMAND=(
  python run_cdm.py
  task_names "${TASK_NAME}"
  num_epochs "${EPOCHS}"
  save_model true
  load_best true
)
COMMAND+=("${WANDB_ARGS[@]}")
COMMAND+=("$@")

printf 'Running task=%s epochs=%s NPU=%s\n' \
  "${TASK_NAME}" "${EPOCHS}" "${ASCEND_RT_VISIBLE_DEVICES}"
printf 'W&B mode=%s project=%s\n' \
  "${WANDB_MODE}" "${WANDB_PROJECT:-gnn-task-relation}"
printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

if [[ ${DRY_RUN:-0} == "1" ]]; then
  exit 0
fi

exec "${COMMAND[@]}"
