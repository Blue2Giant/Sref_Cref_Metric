triplet_jsonl=/data/benchmark_metrics/logs/triplets_style_and_content_only.jsonl
style_index_jsonl=/data/benchmark_metrics/logs/selections_with_origin_style_flux0325.jsonl
overwrite=${OVERWRITE:-0}
extra_args=""
if [ "$overwrite" = "1" ]; then
    extra_args="--overwrite"
fi

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
    --procs_per_endpoint 128 \
    --endpoint "Qwen3-VL-30B-A3B-Instruct@http://10.201.16.8:22002/v1" \
    --endpoint "Qwen3-VL-30B-A3B-Instruct@http://10.201.19.47:22002/v1" \
    --endpoint "Qwen3-VL-30B-A3B-Instruct@http://10.201.16.19:22002/v1" \
    $extra_args

OUT_DIR=/data/benchmark_metrics/logs/triplet_style_firsthit_judge_0325_0.4_2
python3 /data/benchmark_metrics/lora_pipeline/tools/triplet_qwen_style_firsthit_judge.py \
    --triplet-jsonl "$triplet_jsonl" \
    --style-index-jsonl "$style_index_jsonl" \
    --out-jsonl "${OUT_DIR}/style_firsthit.jsonl" \
    --error-log-jsonl "${OUT_DIR}/style_firsthit_errors.jsonl" \
    --num-samples 0 \
    --style_conf_thr 0.5 \
    --style_judge_times 3 \
    --style_min_true 2 \
    --procs_per_endpoint 32 \
    --endpoint "Qwen3-VL-30B-A3B-Instruct@http://10.201.16.8:22002/v1" \
    --endpoint "Qwen3-VL-30B-A3B-Instruct@http://10.201.19.47:22002/v1" \
    --endpoint "Qwen3-VL-30B-A3B-Instruct@http://10.201.16.19:22002/v1" \
    $extra_args


OUT_DIR=/data/benchmark_metrics/logs/triplet_style_firsthit_judge_0325
python3 /data/benchmark_metrics/lora_pipeline/tools/triplet_qwen_style_firsthit_judge.py \
    --triplet-jsonl "$triplet_jsonl" \
    --style-index-jsonl "$style_index_jsonl" \
    --out-jsonl "${OUT_DIR}/style_firsthit.jsonl" \
    --error-log-jsonl "${OUT_DIR}/style_firsthit_errors.jsonl" \
    --num-samples 0 \
    --style_conf_thr 0.5 \
    --style_judge_times 3 \
    --style_min_true 3 \
    --procs_per_endpoint 32 \
    --endpoint "Qwen3-VL-30B-A3B-Instruct@http://10.201.17.67:22002/v1" \
    --endpoint "Qwen3-VL-30B-A3B-Instruct@http://10.201.19.47:22002/v1" \
    --endpoint "Qwen3-VL-30B-A3B-Instruct@http://10.201.16.19:22002/v1" \
    $extra_args
