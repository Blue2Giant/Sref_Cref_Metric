#!/usr/bin/env bash
set -euo pipefail
SREF_ROOT=/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content
OUT_DIR=${OUT_DIR:-$SREF_ROOT/uso}
LOG_DIR=${LOG_DIR:-/data/benchmark_metrics/logs/uso_gpu_temp2_single_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$LOG_DIR"

echo "[info] host=$(hostname)"
echo "[info] start_time=$(date '+%F %T %Z')"
echo "[info] sref_root=$SREF_ROOT"
echo "[info] out_dir=$OUT_DIR"
echo "[info] log_dir=$LOG_DIR"

# Ensure tokenizer dependency exists in this one-GPU worker image.
python3 - <<'PY' || { echo '[info] installing sentencepiece...'; pip install -q sentencepiece; }
import sentencepiece
print('[info] sentencepiece ok', sentencepiece.__version__)
PY

# Stop the old monitor/supervisor that was occupying this single GPU, if present.
# NOTE: pkill -f can match this wrapper's own command line (because this script
# contains the literal pattern), so use bracketed regexes that do not self-match.
pkill -TERM -f '[s]upervise_5030_follow_cref_gt2[.]sh' 2>/dev/null || true
pkill -TERM -f '[m]ulti_cref_eval[.]py' 2>/dev/null || true
sleep 5
if command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF{print $1}' | sort -u)
  if (( ${#gpu_pids[@]} > 0 )); then
    echo "[info] stopping existing GPU compute pids: ${gpu_pids[*]}"
    kill -TERM "${gpu_pids[@]}" 2>/dev/null || true
    sleep 8
    mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF{print $1}' | sort -u)
    if (( ${#gpu_pids[@]} > 0 )); then
      echo "[info] force-stopping remaining GPU compute pids: ${gpu_pids[*]}"
      kill -KILL "${gpu_pids[@]}" 2>/dev/null || true
      sleep 2
    fi
  fi
  echo '[info] GPU before USO:'
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
fi

cd /data/benchmark_metrics/USO
python3 -m py_compile batch_simple_demo.py uso/flux/pipeline.py uso/flux/for_replace.py

echo '[info] launching USO single-GPU inference: output 1024x1024, no input content ref resize/crop'
GPU_IDS=0 NUM_SHARDS=1 OUT_DIR="$OUT_DIR" LOG_DIR="$LOG_DIR" ./batch_run.sh "$SREF_ROOT"

echo "[info] finished_time=$(date '+%F %T %Z')"
python3 - <<'PY'
from pathlib import Path
from collections import Counter
from PIL import Image
root=Path('/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content/uso')
dims=Counter(); bad=[]
for p in root.glob('*.png'):
    try:
        size=Image.open(p).size
        dims[size]+=1
        if size != (1024,1024): bad.append((p.name,size))
    except Exception as e:
        bad.append((p.name,str(e)))
print('[verify] count=', sum(dims.values()), 'dims=', dims.most_common(20), 'bad_count=', len(bad), 'bad_examples=', bad[:5])
PY
