export DIFFSYNTH_MODEL_BASE_PATH=/mnt/jfs/model_zoo
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export TELESTYLE_DIR=/mnt/jfs/model_zoo/Tele-AI/TeleStyle
PYTHON_BIN=/data/Miniconda/.conda/envs/diffsynth/bin/python
#single running
#python /data/benchmark_metrics/sref_cref/TeleStyle_demo.py
sref_root=/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content
#batch running
$PYTHON_BIN /data/benchmark_metrics/sref_cref/TeleStyle_batch.py \
  --cref_dir $sref_root/cref \
  --sref_dir $sref_root/sref \
  --prompts_json /mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content/prompts.json \
  --output_dir $sref_root/TeleStyle_1024x1024 \
  --steps 4 \
  --minedge 1024 \
  --output_resolution 1024x1024
