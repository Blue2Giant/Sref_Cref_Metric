export SEEDREAM_API_KEY=ak-dhco9tfkhr9sb5n2zkgoy0alyeodm3ig
PYTHON_BIN=/home/i-lanjinghong/miniconda3/envs/Sref/bin/python
# python /data/benchmark_metrics/sref_cref/seeddream_mino.py \
#   --cref /data/benchmark_metrics/assets/jiegeng.png \
#   --sref /data/benchmark_metrics/assets/style.webp \
#   --model doubao-seedream-4.0 \
#   --prompt "transfer the first image style to the style of the second image" \
#   --out /data/benchmark_metrics/logs/seedream.png
sref_dir=/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content
# python /data/benchmark_metrics/sref_cref/seeddream_batch.py \
#   --cref_dir $sref_dir/cref \
#   --sref_dir $sref_dir/sref \
#   --prompts_json $sref_dir/prompts.json \
#   --out_dir $sref_dir/seedream \
#   --workers 2
cref_sref_dir=/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content
$PYTHON_BIN /data/benchmark_metrics/sref_cref/seeddream_batch.py \
  --cref_dir $cref_sref_dir/cref \
  --sref_dir $cref_sref_dir/sref \
  --prompts_json /mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content/prompts.json \
  --out_dir $cref_sref_dir/seedream_1024x1024 \
  --resolution 2048x2048 \
  --save_resolution 1024x1024 \
  --workers 2
