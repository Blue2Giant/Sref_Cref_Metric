#!/usr/bin/env bash
set -uo pipefail

ROOT=/data/benchmark_metrics
LOG_DIR="$ROOT/logs"

mkdir -p "$LOG_DIR"
cd "$ROOT"

run_one() {
  local name="$1"
  local script_path="$2"
  local log_path="$3"

  echo "__START__ ${name}"
  bash "$script_path" 2>&1 | tee "$log_path"
  local status=${PIPESTATUS[0]}
  echo "__END__ ${name} status=${status}"
  return 0
}

run_one "TeleStyle_1024x1024" "$ROOT/sref_cref/TeleStyle_demo.sh" "$LOG_DIR/TeleStyle_1024x1024_20260422.log"
run_one "flux_klein_9B_1024x1024" "$ROOT/sref_cref/flux_klein_9B_demo.sh" "$LOG_DIR/flux_klein_9B_1024x1024_20260422.log"
run_one "Qwen_2511_1024x1024" "$ROOT/sref_cref/Qwen_2511_demo.sh" "$LOG_DIR/Qwen_2511_1024x1024_20260422.log"
