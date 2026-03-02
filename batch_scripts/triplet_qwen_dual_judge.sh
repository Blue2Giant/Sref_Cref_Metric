#!/bin/bash
content_dir="/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/cref"
style_dir="/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/sref"
result_dir="/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/qwen-edit"
output_json_content="/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/qwen-edit/qwen_reject_cref.json"
output_json_style="/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/qwen-edit/qwen_reject_sref.json" 
python3 /data/benchmark_metrics/vlm_similarity/triplet_qwen_dual_judge.py \
    --content_dir "$content_dir" \
    --style_dir "$style_dir" \
    --result_dir "$result_dir" \
    --output_content_json $output_json_content \
    --output_style_json $output_json_style \
    --endpoint "qwen3vlw8a8@http://stepcloud-apisix-gateway-eval.i-stepfun.com/Qwen3-VL-235B-A22B-W8A8/v1" \
    --procs_per_endpoint 32 \
    --overwrite