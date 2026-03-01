cd /data/benchmark_metrics/TeleStyle
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/jfs/model_zoo
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
python telestyleimage_inference.py  
