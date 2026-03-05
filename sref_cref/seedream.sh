export SEEDREAM_API_KEY=ak-dhco9tfkhr9sb5n2zkgoy0alyeodm3ig
# python /data/benchmark_metrics/sref_cref/seeddream_mino.py \
#   --cref /data/benchmark_metrics/assets/jiegeng.png \
#   --sref /data/benchmark_metrics/assets/style.webp \
#   --model doubao-seedream-4.0 \
#   --prompt "transfer the first image style to the style of the second image" \
#   --out /data/benchmark_metrics/logs/seedream.png
sref_dir=/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content
python /data/benchmark_metrics/sref_cref/seeddream_batch.py \
  --cref_dir $sref_dir/cref \
  --sref_dir $sref_dir/sref \
  --prompts_json $sref_dir/prompts.json \
  --out_dir $sref_dir/seedream \
  --workers 8
cref_sref_dir=/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content
python /data/benchmark_metrics/sref_cref/seeddream_batch.py \
  --cref_dir $cref_sref_dir/cref \
  --sref_dir $cref_sref_dir/sref \
  --prompts_json $cref_sref_dir/prompts.json \
  --out_dir $cref_sref_dir/seedream \
  --resolution 1024x1024