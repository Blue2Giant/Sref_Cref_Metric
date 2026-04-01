#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="/data/benchmark_metrics/insight/attention_metrics_compare_key_groups_multi.py"
ROOT_DIR="${ROOT_DIR:-/mnt/jfs/qwen-edit-attn-fullmap-keycolor-save-kfull-1-1}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/benchmark_metrics/logs/qwen_attn_key_group_compare_multi-1-1}"
SUMMARY_CSV="${SUMMARY_CSV:-/data/benchmark_metrics/logs/attention_metrics_summary.csv}"
COHORT_DIR="${COHORT_DIR:-/data/benchmark_metrics/insight/key_folder}"

ARGS=(
  --root-dir "$ROOT_DIR"
  --output-dir "$OUTPUT_DIR"
  --cohort-dir "$COHORT_DIR"
  --summary-csv "$SUMMARY_CSV"
)

if [[ "${SKIP_SUMMARY_EXPORT:-0}" == "1" ]]; then
  ARGS+=(--skip-summary-export)
fi

if [[ "${REUSE_SELECTED_LONG_CSV:-0}" == "1" ]]; then
  ARGS+=(--reuse-selected-long-csv)
fi

"$PYTHON_BIN" "$SCRIPT_PATH" "${ARGS[@]}" "$@"
