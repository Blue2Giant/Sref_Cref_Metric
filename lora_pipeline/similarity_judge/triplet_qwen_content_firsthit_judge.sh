#!/usr/bin/env bash
set -euo pipefail

overwrite=${OVERWRITE:-0}
match_threshold=${MATCH_THRESHOLD:-1}
procs_per_endpoint=${PROCS_PER_ENDPOINT:-256}
conn_retry_times=${CONN_RETRY_TIMES:-2}
conn_retry_delay=${CONN_RETRY_DELAY:-1.0}
request_timeout_sec=${REQUEST_TIMEOUT_SEC:-180}
image_cache_size=${IMAGE_CACHE_SIZE:-32}
stats_interval_sec=${STATS_INTERVAL_SEC:-60}
flush_every=${FLUSH_EVERY:-32}

extra_args=()
if [ "$overwrite" = "1" ]; then
    extra_args+=(--overwrite)
fi

# Active qwen tmux server windows checked on 2026-04-08.
ENDPOINTS=(
    # qwen_server_3-65 / qwen_server_4-83
    "Qwen3-VL-30B-A3B-Instruct@http://10.204.18.58:22002/v1"
    "qwen3vlw8a8@http://stepcloud-apisix-gateway-eval.i-stepfun.com/Qwen3-VL-235B-A22B-W8A8/v1"
    "qwen35-27b@http://stepcast-router.shai-core:9200/v1"
)

endpoint_args=()
for endpoint in "${ENDPOINTS[@]}"; do
    endpoint_args+=(--endpoint "$endpoint")
done

common_args=(
    --num-samples 0
    --content_conf_thr 0.5
    --content_judge_times 3
    --content_min_true 2
    --procs_per_endpoint "$procs_per_endpoint"
    --conn_retry_times "$conn_retry_times"
    --conn_retry_delay "$conn_retry_delay"
    --request-timeout-sec "$request_timeout_sec"
    --image-cache-size "$image_cache_size"
    --stats-interval-sec "$stats_interval_sec"
    --flush-every "$flush_every"
    --match_threshold "$match_threshold"
    --per-image
    "${endpoint_args[@]}"
    "${extra_args[@]}"
)

while true; do
    # triplet_jsonl=/data/benchmark_metrics/logs/triplet_jsonl/qwen_0323_dual_lora_images_by_subfolder.jsonl
    # content_jsonl=/data/benchmark_metrics/logs/triplet_jsonl/qwen_selections_original_paths.jsonl
    # OUT_DIR=/mnt/jfs/logs/qwen_triplet_content_firsthit_judge_0325_0.5_2_0403_perimage
    # OUT_DIR=/mnt/jfs/logs/qwen_triplet_content_firsthit_judge_0325_0.5_1_0403_perimage
    # python3 /data/benchmark_metrics/lora_pipeline/similarity_judge/triplet_qwen_content_firsthit_judge.py \
    #     --triplet-jsonl "$triplet_jsonl" \
    #     --content-index-jsonl "$content_jsonl" \
    #     --out-jsonl "${OUT_DIR}/content_firsthit_matched.jsonl" \
    #     --error-log-jsonl "${OUT_DIR}/content_firsthit_errors.jsonl" \
    #     --processed-jsonl "${OUT_DIR}/content_firsthit_processed.jsonl" \
    #     "${common_args[@]}"
    # triplet_jsonl=/data/benchmark_metrics/logs/triplet_jsonl/illustrious_0323_dual_lora_diverse_unique_images.jsonl
    # content_jsonl=/data/benchmark_metrics/logs/triplet_jsonl/illustrious_selections_original_paths.jsonl
    # OUT_DIR=/data/benchmark_metrics/logs/illustrious_triplet_content_firsthit_judge_0325_0.5_2_0403_perimage
    # python3 /data/benchmark_metrics/lora_pipeline/similarity_judge/triplet_qwen_content_firsthit_judge.py \
    #     --triplet-jsonl "$triplet_jsonl" \
    #     --content-index-jsonl "$content_jsonl" \
    #     --out-jsonl "${OUT_DIR}/content_firsthit_matched.jsonl" \
    #     --error-log-jsonl "${OUT_DIR}/content_firsthit_errors.jsonl" \
    #     --processed-jsonl "${OUT_DIR}/content_firsthit_processed.jsonl" \
    #     --pair-key-order style_content \
    #     "${common_args[@]}"

    triplet_jsonl=/data/benchmark_metrics/logs/triplet_jsonl/flux_0323_dual_lora_diverse_save_prompt_0328_images_by_subfolder.jsonl
    content_jsonl=/data/benchmark_metrics/logs/triplet_jsonl/selections_with_origin_content_flux.jsonl
    OUT_DIR=/data/benchmark_metrics/logs/triplet_content_firsthit_judge_0325_0.5_2_0402_perimage
    OUT_DIR=/data/benchmark_metrics/logs/triplet_content_firsthit_judge_0325_0.5_1_0402_perimage
    python3 /data/benchmark_metrics/lora_pipeline/similarity_judge/triplet_qwen_content_firsthit_judge.py \
        --triplet-jsonl "$triplet_jsonl" \
        --content-index-jsonl "$content_jsonl" \
        --out-jsonl "${OUT_DIR}/content_firsthit_matched.jsonl" \
        --error-log-jsonl "${OUT_DIR}/content_firsthit_errors.jsonl" \
        --processed-jsonl "${OUT_DIR}/content_firsthit_processed.jsonl" \
        "${common_args[@]}"
done
