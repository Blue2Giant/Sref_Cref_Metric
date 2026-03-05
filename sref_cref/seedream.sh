export SEEDREAM_API_KEY=ak-dhco9tfkhr9sb5n2zkgoy0alyeodm3ig
python /data/benchmark_metrics/sref_cref/seedream.py \
  --cref /data/benchmark_metrics/assets/jiegeng.png \
  --sref /data/benchmark_metrics/assets/style.webp \
  --prompt "transfer the first image style to the style of the second image" \
  --out /data/benchmark_metrics/logs/seedream.png