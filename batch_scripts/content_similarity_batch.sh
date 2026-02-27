content_dir="/mnt/jfs/bench-bucket/sref_bench/sample_1500_bench_cref_sref/cref"
style_dir="/mnt/jfs/bench-bucket/sref_bench/sample_1500_bench_cref_sref/sref"
result_dir="/mnt/jfs/bench-bucket/sref_bench/sample_1500_bench_cref_sref/qwen_output_resize"
output_json_content_discrete="s3://lanjinghong-data/sample_1500_bench_cref_sref/qwen_resize_output_content_descrete.json"
reason_json_content_discrete="s3://lanjinghong-data/sample_1500_bench_cref_sref/qwen_resize_output_content_reason_descrete.json"
xingpeng_ip=http://stepcloud-apisix-gateway-eval.i-stepfun.com/Qwen3-VL-235B-A22B-W8A8/v1
xingpeng_model=qwen3vlw8a8
python3 /data/benchmark_metrics/vlm_similarity/content_similarity_dir.py \
  --content_dir $content_dir \
  --output_dir $result_dir \
  --out_json $output_json_content_discrete \
  --out_reason_json $reason_json_content_discrete \
  --base_url $xingpeng_ip \
  --model $xingpeng_model \
  --num_samples 100 \
  --num_procs 64