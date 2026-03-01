export DIFFSYNTH_MODEL_BASE_PATH=/mnt/jfs/model_zoo
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export TELESTYLE_DIR=/mnt/jfs/model_zoo/Tele-AI/TeleStyle
#single running
#python /data/benchmark_metrics/sref_cref/TeleStyle_demo.py

#batch running
python /data/benchmark_metrics/sref_cref/TeleStyle_batch.py \
  --cref_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_new/cref \
  --sref_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_new/sref \
  --prompts_json /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_new/prompts.json \
  --output_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_new/TeleStyle \
  --steps 4 \
  --minedge 1024