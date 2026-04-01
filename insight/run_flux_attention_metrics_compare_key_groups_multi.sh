#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="/data/benchmark_metrics/insight/attention_metrics_compare_key_groups_multi_flux.py"
ROOT_DIR="${ROOT_DIR:-/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/flux-klein-9b-attn-fullmap}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/benchmark_metrics/logs/flux_attn_key_group_compare}"
SUMMARY_CSV="${SUMMARY_CSV:-/data/benchmark_metrics/logs/flux_attention_metrics_summary.csv}"
COHORT_DIR="${COHORT_DIR:-/data/benchmark_metrics/insight/key_folder/flux}"

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
