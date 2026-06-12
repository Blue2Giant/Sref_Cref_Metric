#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/benchmark_metrics
LOG_DIR="$ROOT/logs"

mkdir -p "$LOG_DIR"
cd "$ROOT"

bash "$ROOT/sref_cref/seedream.sh" 2>&1 | tee "$LOG_DIR/seedream_1024x1024_20260422.log"
