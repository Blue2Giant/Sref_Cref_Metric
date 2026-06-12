#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_PATH="/data/benchmark_metrics/insight/qwen_attention_paper_novelty_figures.py"

SELECTED_LONG_CSV="${SELECTED_LONG_CSV:-/data/benchmark_metrics/logs/qwen_attn_key_group_compare_kfull_1_1_q_latent_range_20260408_remote_root/selected_metrics_long.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/benchmark_metrics/logs/qwen_attention_paper_novelty_qwen_20260412_tmux}"
PRIMARY_GRID_ROOT="${PRIMARY_GRID_ROOT:-/mnt/jfs/qwen-edit-attn-fullmap-keycolor-save-kfull-1-1-q-latent-range}"
FALLBACK_GRID_ROOT="${FALLBACK_GRID_ROOT:-/data/benchmark_metrics/logs/qwen-edit-attn-fullmap-keycolor}"

"$PYTHON_BIN" "$SCRIPT_PATH" \
  --selected-long-csv "$SELECTED_LONG_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --grid-root "$PRIMARY_GRID_ROOT" \
  --grid-root "$FALLBACK_GRID_ROOT" \
  --export-format png \
  --export-format svg \
  "$@"
