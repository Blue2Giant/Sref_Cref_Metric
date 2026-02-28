python3 /data/benchmark_metrics/stream_extract_images_from_tar.py \
  --tar s3://xp-base/datasets/sft/20250221-lofter-aes/human-4+/lofteriqa-m9gZ-0_000011.tar \
  --out-dir /data/benchmark_metrics/logs/human \
  --num 500
python3 /data/benchmark_metrics/stream_extract_images_from_tar.py \
  --tar s3://xp-base/datasets/sft/20250121-lofter/aesthetics-human/aesthetics-nohuman/lofter-0_000011.tar \
  --out-dir /data/benchmark_metrics/logs/nonhuman \
  --num 100