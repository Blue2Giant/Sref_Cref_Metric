#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=/data/benchmark_metrics/lora_pipeline/similarity_judge

STYLE_DIR=${STYLE_DIR:-/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/sref}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/0415_qwen_image_sref_noise_query_recap_infer_all_no_attn_ckpt20000_all_key_sref}

run_name_default="$(basename "$OUTPUT_DIR")_content_leakage"
RUN_NAME=${RUN_NAME:-$run_name_default}
OUT_DIR=${OUT_DIR:-/mnt/jfs/logs/$RUN_NAME}

ENDPOINT=${ENDPOINT:-qwen3vlw8a8@http://stepcloud-apisix-gateway-eval.i-stepfun.com/Qwen3-VL-235B-A22B-W8A8/v1}
NUM_PROCS=${NUM_PROCS:-64}
NUM_SAMPLES=${NUM_SAMPLES:-0}
SEED=${SEED:-1234}
MAX_TOKENS=${MAX_TOKENS:-256}
TIMEOUT=${TIMEOUT:-600}
OVERWRITE=${OVERWRITE:-0}
BASENAME_TXT=${BASENAME_TXT:-}
FLUSH_EVERY=${FLUSH_EVERY:-10}

extra_args=()
if [ "$OVERWRITE" = "1" ]; then
    extra_args+=(--overwrite)
fi

mkdir -p "$OUT_DIR"

python3 "$SCRIPT_DIR/style_similarity_dir.py" \
    --judge_mode content_leakage \
    --style_dir "$STYLE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --out_score_json "$OUT_DIR/content_leakage_score.json" \
    --out_reason_json "$OUT_DIR/content_leakage_reason.json" \
    --endpoint "$ENDPOINT" \
    --num_procs "$NUM_PROCS" \
    --num_samples "$NUM_SAMPLES" \
    --seed "$SEED" \
    --max_tokens "$MAX_TOKENS" \
    --timeout "$TIMEOUT" \
    --flush_every "$FLUSH_EVERY" \
    --basename_txt "$BASENAME_TXT" \
    "${extra_args[@]}"
