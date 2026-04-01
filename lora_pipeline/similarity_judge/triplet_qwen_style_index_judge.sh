triplet_jsonl=/data/benchmark_metrics/logs/triplets_style_and_content_only.jsonl
style_index_jsonl=/data/benchmark_metrics/logs/selections_with_origin_style_flux.jsonl
OUT_DIR=/data/benchmark_metrics/logs/triplet_style_index_judge_0324
overwrite=${OVERWRITE:-0}
extra_args=""
if [ "$overwrite" = "1" ]; then
    extra_args="--overwrite"
fi

python3 /data/benchmark_metrics/lora_pipeline/similarity_judge/triplet_qwen_style_index_judge.py \
    --triplet-jsonl "$triplet_jsonl" \
    --style-index-jsonl "$style_index_jsonl" \
    --out-jsonl "${OUT_DIR}/style_binary.jsonl" \
    --error-log-jsonl "${OUT_DIR}/style_errors.jsonl" \
    --num-samples 0 \
    --style_conf_thr 0.5 \
    --style_ratio 0.6 \
    --procs_per_endpoint 128 \
    --endpoint "Qwen3-VL-30B-A3B-Instruct@http://10.201.18.35:22002/v1" \
    --endpoint "Qwen3-VL-30B-A3B-Instruct@http://10.201.19.47:22002/v1" \
    --endpoint "Qwen3-VL-30B-A3B-Instruct@http://10.201.16.19:22002/v1" \
    $extra_args
