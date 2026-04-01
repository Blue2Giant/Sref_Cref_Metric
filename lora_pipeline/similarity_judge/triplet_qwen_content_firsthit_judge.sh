#!/usr/bin/env bash
triplet_jsonl=/data/benchmark_metrics/logs/triplets_style_and_content_only.jsonl
content_jsonl=/data/benchmark_metrics/logs/selections_with_origin_content_flux.jsonl
overwrite=${OVERWRITE:-0}
match_threshold=${MATCH_THRESHOLD:-1}
extra_args=""
if [ "$overwrite" = "1" ]; then
    extra_args="--overwrite"
fi
endpoint1="Qwen3-VL-30B-A3B-Instruct@http://10.191.4.18:22002/v1"
endpoint2="Qwen3-VL-30B-A3B-Instruct@http://10.201.19.47:22002/v1"
endpoint3="Qwen3-VL-30B-A3B-Instruct@http://10.201.16.19:22002/v1"
endpoint4="Qwen3-VL-30B-A3B-Instruct@http://10.201.17.67:22002/v1"
endpoint5="Qwen3-VL-30B-A3B-Instruct@http://10.201.18.31:22002/v1"
endpoint6="Qwen3-VL-30B-A3B-Instruct@http://10.191.21.51:22002/v1"
OUT_DIR=/data/benchmark_metrics/logs/triplet_content_firsthit_judge_0325_0.5_2
# python3 /data/benchmark_metrics/lora_pipeline/tools/triplet_qwen_content_firsthit_judge.py \
#     --triplet-jsonl $triplet_jsonl \
#     --content-index-jsonl $content_jsonl \
#     --out-jsonl "${OUT_DIR}/style_firsthit.jsonl" \
#     --error-log-jsonl "${OUT_DIR}/style_firsthit_errors.jsonl" \
#     --num-samples 0 \
#     --content_conf_thr 0.5 \
#     --content_judge_times 3 \
#     --content_min_true 2 \
#     --match_threshold 2 \
#     --procs_per_endpoint 4 \
#     --endpoint "$endpoint1" \
#     --endpoint "$endpoint2" \
#     --endpoint "$endpoint3" \
#     --endpoint "$endpoint4" \
#     --endpoint "$endpoint5" \
#     --endpoint "$endpoint6"

#triplet update 0327
OUT_DIR=/data/benchmark_metrics/logs/triplet_content_firsthit_judge_0325_0.5_2
python3 /data/benchmark_metrics/lora_pipeline/similarity_judge/triplet_qwen_content_firsthit_judge.py \
    --triplet-jsonl $triplet_jsonl \
    --content-index-jsonl $content_jsonl \
    --out-jsonl "${OUT_DIR}/style_firsthit.jsonl" \
    --error-log-jsonl "${OUT_DIR}/style_firsthit_errors.jsonl" \
    --num-samples 0 \
    --content_conf_thr 0.5 \
    --content_judge_times 3 \
    --content_min_true 2 \
    --match_threshold 2 \
    --procs_per_endpoint 4 \
    --endpoint "$endpoint1" \
    --endpoint "$endpoint2" \
    --endpoint "$endpoint3" \
    --endpoint "$endpoint4" \
    --endpoint "$endpoint5" \
    --endpoint "$endpoint6"