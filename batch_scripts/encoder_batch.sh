#!/bin/bash
set -euo pipefail

RUNNER_PY="/data/benchmark_metrics/benchmark_metrics/encoder_batch_runner.py"
NUM_PROCS="8"

PAIR_DIR_A="/path/to/folderA"
PAIR_DIR_B="/path/to/folderB"

DINOV2_MODEL="/mnt/jfs/model_zoo/dinov2-with-registers-large"
CAS_MODEL="/mnt/jfs/model_zoo/dinov2-base"
ONEIG_MODEL="/mnt/jfs/model_zoo/OneIG-StyleEncoder"

OUT_DINOV2_JSON="/path/to/dinov2_out.json"
OUT_CAS_JSON="/path/to/cas_out.json"
OUT_ONEIG_JSON="/path/to/oneig_out.json"

CLIPCAP_IMAGE_DIR="/path/to/images"
CLIPCAP_PROMPT_JSON="/path/to/prompts.json"
CLIPCAP_MODEL="/mnt/jfs/model_zoo/clip-vit-large-patch14"
OUT_CLIPCAP_JSON="/path/to/clipcap_out.json"

ONEALIGN_IMAGE_DIR="/path/to/images"
ONEALIGN_MODEL="/mnt/jfs/model_zoo/one-align"
ONEALIGN_TASK="aesthetics"
OUT_ONEALIGN_JSON="/path/to/onealign_out.json"

echo "=== dinov2 ===="
python3 "$RUNNER_PY" pair \
  --encoder dinov2 \
  --dir_a "$PAIR_DIR_A" \
  --dir_b "$PAIR_DIR_B" \
  --model "$DINOV2_MODEL" \
  --out_json "$OUT_DINOV2_JSON" \
  --num_procs "$NUM_PROCS"

echo "=== cas ===="
python3 "$RUNNER_PY" pair \
  --encoder cas \
  --dir_a "$PAIR_DIR_A" \
  --dir_b "$PAIR_DIR_B" \
  --model "$CAS_MODEL" \
  --out_json "$OUT_CAS_JSON" \
  --num_procs "$NUM_PROCS"

echo "=== oneig ===="
python3 "$RUNNER_PY" pair \
  --encoder oneig \
  --dir_a "$PAIR_DIR_A" \
  --dir_b "$PAIR_DIR_B" \
  --model "$ONEIG_MODEL" \
  --out_json "$OUT_ONEIG_JSON" \
  --num_procs "$NUM_PROCS"

echo "=== clip cap ==="
python3 "$RUNNER_PY" clip_cap \
  --image_dir "$CLIPCAP_IMAGE_DIR" \
  --prompt_json "$CLIPCAP_PROMPT_JSON" \
  --out_json "$OUT_CLIPCAP_JSON" \
  --model "$CLIPCAP_MODEL" \
  --num_procs "$NUM_PROCS"

echo "=== onealign aesthetics ==="
python3 "$RUNNER_PY" onealign \
  --image_dir "$ONEALIGN_IMAGE_DIR" \
  --out_json "$OUT_ONEALIGN_JSON" \
  --model "$ONEALIGN_MODEL" \
  --task "$ONEALIGN_TASK" \
  --num_procs "$NUM_PROCS"
