export OPENAI_API_KEY=ak-dhco9tfkhr9sb5n2zkgoy0alyeodm3ig
sref_dir=/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content
sref_dir=/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content
python /data/benchmark_metrics/caption_pipe/gpt-4o-haoling_core_batch.py \
  --prompts_json $sref_dir/prompts.json \
  --cref_dir $sref_dir/cref \
  --sref_dir $sref_dir/sref \
  --out_dir $sref_dir/gpt4o-edit \
  --model gpt-4o-all \
  --base_url https://models-proxy.stepfun-inc.com/v1 \
  --num_procs 16