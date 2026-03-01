export DIFFSYNTH_MODEL_BASE_PATH=/mnt/jfs/model_zoo
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export TELESTYLE_DIR=/mnt/jfs/model_zoo/Tele-AI/TeleStyle
#single running
#python /data/benchmark_metrics/sref_cref/TeleStyle_demo.py
sref_root=/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content
#batch running
python /data/benchmark_metrics/sref_cref/TeleStyle_batch.py \
  --cref_dir $sref_root/cref \
  --sref_dir $sref_root/sref \
  --prompts_json $sref_root/prompts.json \
  --output_dir $sref_root/TeleStyle \
  --steps 4 \
  --minedge 1024