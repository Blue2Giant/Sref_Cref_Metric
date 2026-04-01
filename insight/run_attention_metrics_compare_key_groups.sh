#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/Miniconda/.conda/envs/comfyui/bin/python}"
ROOT_DIR="${ROOT_DIR:-/mnt/jfs/qwen-edit-attn-fullmap-keycolor-save}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/benchmark_metrics/insight/output/qwen_attn_key_group_compare}"
SCRIPT_PATH="/data/benchmark_metrics/insight/attention_metrics_compare_key_groups.py"

ARGS=(
  --root-dir "$ROOT_DIR"
  --output-dir "$OUTPUT_DIR"
)

if [[ "${REUSE_SELECTED_LONG_CSV:-0}" == "1" ]]; then
  ARGS+=(
    --skip-summary-export
    --reuse-selected-long-csv
  )
fi

"$PYTHON_BIN" "$SCRIPT_PATH" "${ARGS[@]}" "$@"
