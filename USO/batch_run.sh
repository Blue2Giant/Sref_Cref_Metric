#!/usr/bin/env bash
set -euo pipefail

# USO batch inference for the cref+sref benchmark.
# Requirements for this run:
#   - output resolution is forced to 1024x1024
#   - input reference images are passed at their original pixel size (no resize/crop)
#   - by default use a single GPU (GPU 0); override GPU_IDS/NUM_SHARDS if needed

sref_root=${1:-/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content}
out_dir=${OUT_DIR:-$sref_root/uso}
num_shards=${NUM_SHARDS:-1}
gpu_csv=${GPU_IDS:-0}

IFS=',' read -r -a gpu_ids <<< "$gpu_csv"
if (( ${#gpu_ids[@]} < num_shards )); then
  echo "[error] GPU_IDS=$gpu_csv has fewer entries than NUM_SHARDS=$num_shards" >&2
  exit 2
fi

mkdir -p "$out_dir"
log_dir=${LOG_DIR:-/data/benchmark_metrics/logs/uso_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$log_dir"

printf '[info] input root: %s\n' "$sref_root"
printf '[info] output dir: %s\n' "$out_dir"
printf '[info] logs: %s\n' "$log_dir"
printf '[info] GPUs: %s; shards: %s\n' "$gpu_csv" "$num_shards"
printf '[info] output size forced to 1024x1024; --instruct-edit is intentionally disabled.\n'
printf '[info] content refs use --no-preprocess-ref, so their pixel sizes are not resized/cropped by this script.\n'

pids=()
for shard in $(seq 0 $((num_shards - 1))); do
  gpu=${gpu_ids[$shard]}
  log_file="$log_dir/gpu${gpu}_shard${shard}.log"
  echo "[launch] shard=$shard/$num_shards gpu=$gpu log=$log_file"
  (
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHONUNBUFFERED=1
    python3 /data/benchmark_metrics/USO/batch_simple_demo.py \
      --input-dir "$sref_root" \
      --prompts-json "$sref_root/prompts.json" \
      --out-dir "$out_dir" \
      --overwrite \
      --width 1024 \
      --height 1024 \
      --no-preprocess-ref \
      --num-shards "$num_shards" \
      --shard-index "$shard" \
      --sref-only \
      --use-siglip
  ) >"$log_file" 2>&1 &
  pids+=("$!")
done

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[done] shard=$i pid=${pids[$i]} succeeded"
  else
    rc=$?
    echo "[error] shard=$i pid=${pids[$i]} failed rc=$rc" >&2
    status=$rc
  fi
done

if (( status == 0 )); then
  echo "[done] all shards completed -> $out_dir"
else
  echo "[error] one or more shards failed; inspect $log_dir" >&2
fi
exit "$status"
