cd /data/benchmark_metrics/benchmark_metrics
python /data/benchmark_metrics/benchmark_metrics/csd_cosine_demo.py \
  --image_a /data/benchmark_metrics/assets/stylized.png\
  --image_b /data/benchmark_metrics/assets/style.webp \
  --model_path /data/benchmark_metrics/logs/csd.pth  \
  --clip_model_path /data/benchmark_metrics/logs/ViT-L-14.pt  \
  --device cuda:0