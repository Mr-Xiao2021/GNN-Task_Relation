#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} =~ ^[0-9]+$ ]]; then
  exec "${SCRIPT_DIR}/run_e2e_node_npu.sh" cora_node "$@"
fi
exec "${SCRIPT_DIR}/run_e2e_node_npu.sh" cora_node "${EPOCHS:-200}" "$@"
