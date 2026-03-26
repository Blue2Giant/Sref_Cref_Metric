triplet_jsonl=/data/benchmark_metrics/logs/triplets_style_and_content_only.jsonl
style_index_jsonl=/data/benchmark_metrics/logs/selections_with_origin_style_flux0325.jsonl
overwrite=${OVERWRITE:-0}
match_threshold=${MATCH_THRESHOLD:-1}
extra_args=""
if [ "$overwrite" = "1" ]; then
    extra_args="--overwrite"
fi
endpoint1="Qwen3-VL-30B-A3B-Instruct@http://10.201.17.67:22002/v1"
endpoint2="Qwen3-VL-30B-A3B-Instruct@http://10.201.18.31:22002/v1"
endpoint3="Qwen3-VL-30B-A3B-Instruct@http://10.191.21.51:22002/v1"
endpoint4="Qwen3-VL-30B-A3B-Instruct@http://10.201.16.8:22002/v1"
endpoint5="Qwen3-VL-30B-A3B-Instruct@http://10.201.19.47:22002/v1"
endpoint6="Qwen3-VL-30B-A3B-Instruct@http://10.201.16.19:22002/v1"


OUT_DIR=/data/benchmark_metrics/logs/triplet_style_firsthit_judge_0325_0.5_2
python3 /data/benchmark_metrics/lora_pipeline/tools/triplet_qwen_style_firsthit_judge.py \
    --triplet-jsonl "$triplet_jsonl" \
    --style-index-jsonl "$style_index_jsonl" \
    --out-jsonl "${OUT_DIR}/style_firsthit.jsonl" \
    --error-log-jsonl "${OUT_DIR}/style_firsthit_errors.jsonl" \
    --num-samples 0 \
    --style_conf_thr 0.5 \
    --style_judge_times 3 \
    --style_min_true 2 \
    --probe-timeout 3.0 \
    --procs_per_endpoint 64 \
    --endpoint $endpoint1 \
    --endpoint $endpoint2 \
    --endpoint $endpoint3 \
    --endpoint $endpoint4 \
    --endpoint $endpoint5 \
    --endpoint $endpoint6 \
    $extra_args

OUT_DIR=/data/benchmark_metrics/logs/triplet_style_firsthit_judge_0325_0.5_2_2match
python3 /data/benchmark_metrics/lora_pipeline/tools/triplet_qwen_style_firsthit_judge.py \
    --triplet-jsonl "$triplet_jsonl" \
    --style-index-jsonl "$style_index_jsonl" \
    --out-jsonl "${OUT_DIR}/style_firsthit.jsonl" \
    --error-log-jsonl "${OUT_DIR}/style_firsthit_errors.jsonl" \
    --num-samples 0 \
    --style_conf_thr 0.5 \
    --style_judge_times 3 \
    --style_min_true 2 \
    --probe-timeout 3.0 \
    --procs_per_endpoint 64 \
    --endpoint $endpoint1 \
    --endpoint $endpoint2 \
    --endpoint $endpoint3 \
    --endpoint $endpoint4 \
    --endpoint $endpoint5 \
    --endpoint $endpoint6 \
    --match_threshold 2 \
    $extra_args
# OUT_DIR=/data/benchmark_metrics/logs/triplet_style_firsthit_judge_0325_0.4_2
# python3 /data/benchmark_metrics/lora_pipeline/tools/triplet_qwen_style_firsthit_judge.py \
#     --triplet-jsonl "$triplet_jsonl" \
#     --style-index-jsonl "$style_index_jsonl" \
#     --out-jsonl "${OUT_DIR}/style_firsthit.jsonl" \
#     --error-log-jsonl "${OUT_DIR}/style_firsthit_errors.jsonl" \
#     --num-samples 0 \
#     --style_conf_thr 0.5 \
#     --style_judge_times 3 \
#     --style_min_true 2 \
#     --match_threshold "$match_threshold" \
#     --procs_per_endpoint 32 \
#     --endpoint $endpoint1 \
#     --endpoint $endpoint2 \
#     --endpoint $endpoint3 \
#     --endpoint $endpoint4 \
#     --endpoint $endpoint5 \
#     $extra_args


# OUT_DIR=/data/benchmark_metrics/logs/triplet_style_firsthit_judge_0325
# python3 /data/benchmark_metrics/lora_pipeline/tools/triplet_qwen_style_firsthit_judge.py \
#     --triplet-jsonl "$triplet_jsonl" \
#     --style-index-jsonl "$style_index_jsonl" \
#     --out-jsonl "${OUT_DIR}/style_firsthit.jsonl" \
#     --error-log-jsonl "${OUT_DIR}/style_firsthit_errors.jsonl" \
#     --num-samples 0 \
#     --style_conf_thr 0.5 \
#     --style_judge_times 3 \
#     --style_min_true 3 \
#     --match_threshold "$match_threshold" \
#     --procs_per_endpoint 32 \
#     --endpoint $endpoint1 \
#     --endpoint $endpoint2 \
#     --endpoint $endpoint3 \
#     --endpoint $endpoint4 \
#     --endpoint $endpoint5 \
#     $extra_args
#     $extra_args
