python /data/benchmark_metrics/insight/qwen_2511_single_image_demo.py \
  --prompts_json /mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/prompts.json \
  --cref_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/cref \
  --sref_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/sref \
  --out_dir /data/benchmark_metrics/logs/qwen-edit-single \
  --model_name /mnt/jfs/model_zoo/qwen/Qwen-Image-Edit-2511/ \
  --gpus 0 \
  --key_txt /data/benchmark_metrics/insight/key.txt