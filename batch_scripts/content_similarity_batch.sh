content_dir="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/cref"
result_dir="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/qwen-edit"
output_json_content_discrete="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/qwen-edit/qwen_resize_output_content_descrete.json"
reason_json_content_discrete="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/qwen-edit/qwen_resize_output_content_reason_descrete.json"
xingpeng_ip=http://stepcloud-apisix-gateway-eval.i-stepfun.com/Qwen3-VL-235B-A22B-W8A8/v1
xingpeng_model=qwen3vlw8a8
python3 /data/benchmark_metrics/vlm_similarity/content_similarity_dir.py \
  --content_dir $content_dir \
  --output_dir $result_dir \
  --out_json $output_json_content_discrete \
  --out_reason_json $reason_json_content_discrete \
  --base_url $xingpeng_ip \
  --model $xingpeng_model \
  --num_procs 64 \
  --overwrite