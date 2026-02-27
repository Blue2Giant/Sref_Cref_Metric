#!/bin/bash
mkdir -p "$OUT_DIR"
while true;do
    content_dir="/mnt/jfs/bench-bucket/sref_bench/sample_1500_bench_cref_sref/cref"
    style_dir="/mnt/jfs/bench-bucket/sref_bench/sample_1500_bench_cref_sref/sref"
    result_dir="/mnt/jfs/bench-bucket/sref_bench/sample_1500_bench_cref_sref/qwen_output_resize"
    output_json_content="s3://lanjinghong-data/sample_1500_bench_cref_sref/qwen_resize_output_content.json"
    output_json_style="s3://lanjinghong-data/sample_1500_bench_cref_sref/qwen_resize_output_style.json" 
    python3 /data/benchmark_metrics/triplet_similarity/triplet_qwen_dual_judge.py \
        --content_dir "$content_dir" \
        --style_dir "$style_dir" \
        --result_dir "$result_dir" \
        --num_samples 2000 \
        --output_content_json $output_json_content \
        --output_style_json $output_json_style \
        --endpoint "qwen3vlw8a8@http://stepcloud-apisix-gateway-eval.i-stepfun.com/Qwen3-VL-235B-A22B-W8A8/v1" \
        --num_samples 100 \
        --procs_per_endpoint 32
done