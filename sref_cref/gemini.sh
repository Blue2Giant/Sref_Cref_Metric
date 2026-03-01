export GEMINI_API_KEY=ak-dhco9tfkhr9sb5n2zkgoy0alyeodm3ig
python /data/benchmark_metrics/caption_pipe/gemini_image_min_batch.py \
  --prompts_json /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/prompts.json \
  --cref_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/cref \
  --sref_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/sref \
  --out_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/gemini-edit \
  --model_id gemini-2.5-flash-image-native \
  --num_generate 100 \
  --num_procs 8