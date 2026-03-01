python /data/benchmark_metrics/caption_pipe/gpt-4o-haoling_core_batch.py \
  --prompts_json /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/prompts.json \
  --cref_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/cref \
  --sref_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/sref \
  --out_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/gpt4o-edit \
  --model gpt-4o-all \
  --base_url https://models-proxy.stepfun-inc.com/v1 \
  --api_key YOUR_KEY \
  --num_procs 8