#!/usr/bin/env bash
set -uo pipefail

ROOT=${ROOT:-/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content}
GPU_A=${GPU_A:-1}
GPU_B=${GPU_B:-5}
GPUS_PHYS="${GPU_A},${GPU_B}"
VISIBLE_GPLUS="0,1"
RES=${RES:-1024x1024}
RUN_TAG=${RUN_TAG:-1024x1024_bucketinput_2gpu_0612}
RUN_ROOT="$ROOT/three_model_monitor_${RUN_TAG}"
MONITOR_LOG="$RUN_ROOT/monitor.log"
mkdir -p "$RUN_ROOT"

exec 9>"$RUN_ROOT/monitor.lock"
if ! flock -n 9; then
  echo "[$(date '+%F %T %Z')] another monitor is already running: $RUN_ROOT/monitor.lock" | tee -a "$MONITOR_LOG"
  exit 0
fi

PY_TEL=${PY_TEL:-/data/Miniconda/.conda/envs/diffsynth/bin/python}
PY_SREF=${PY_SREF:-/data/Miniconda/.conda/envs/diffsynth/bin/python}
TEL_SCRIPT=${TEL_SCRIPT:-/data/benchmark_metrics/sref_cref/TeleStyle_batch.py}
FLUX_SCRIPT=${FLUX_SCRIPT:-/data/benchmark_metrics/sref_cref/flux_klein_9B.py}
QWEN_SCRIPT=${QWEN_SCRIPT:-/data/benchmark_metrics/sref_cref/qwen_infer.py}
PROMPTS_JSON=${PROMPTS_JSON:-$ROOT/prompts.json}
CREF_DIR=${CREF_DIR:-$ROOT/cref}
SREF_DIR=${SREF_DIR:-$ROOT/sref}

TEL_OUT=${TEL_OUT:-$ROOT/TeleStyle_${RUN_TAG}}
FLUX_OUT=${FLUX_OUT:-$ROOT/flux-klein-9b_${RUN_TAG}}
QWEN_OUT=${QWEN_OUT:-$ROOT/qwen-edit_${RUN_TAG}}
SHARD_DIR=${SHARD_DIR:-$ROOT/telestyle_shards_0612}

export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/jfs/model_zoo}
export DIFFSYNTH_SKIP_DOWNLOAD=${DIFFSYNTH_SKIP_DOWNLOAD:-true}
export DIFFSYNTH_DOWNLOAD_SOURCE=${DIFFSYNTH_DOWNLOAD_SOURCE:-huggingface}
export TELESTYLE_DIR=${TELESTYLE_DIR:-/mnt/jfs/model_zoo/Tele-AI/TeleStyle}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

log() {
  echo "[$(date '+%F %T %Z')] $*" | tee -a "$MONITOR_LOG"
}

count_png() {
  local out="$1"
  find "$out" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l | awk '{print $1}'
}

total_prompts() {
  "$PY_TEL" - <<PY
import json
print(len(json.load(open('$PROMPTS_JSON', 'r', encoding='utf-8'))))
PY
}

TOTAL=$(total_prompts)

pid_csv_from_dir() {
  local dir="$1"
  if compgen -G "$dir/logs/*.pid" >/dev/null; then
    cat "$dir"/logs/*.pid 2>/dev/null | paste -sd, -
  fi
}

alive_csv() {
  local csv="$1"
  local out=""
  IFS=',' read -ra arr <<< "$csv"
  for p in "${arr[@]}"; do
    [[ -n "${p:-}" ]] || continue
    if kill -0 "$p" 2>/dev/null; then
      if [[ -z "$out" ]]; then out="$p"; else out="$out,$p"; fi
    fi
  done
  echo "$out"
}

status_snapshot() {
  local model="$1"
  local out="$2"
  local pids="$3"
  local cnt
  cnt=$(count_png "$out")
  log "STATUS model=$model png=$cnt/$TOTAL out=$out pids=${pids:-none}"
  if [[ -n "${pids:-}" ]]; then
    ps -p "$pids" -o pid,stat,etime,pcpu,pmem,cmd 2>&1 | sed 's/^/[ps] /' | tee -a "$MONITOR_LOG" || true
  fi
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>&1 | sed 's/^/[gpu] /' | tee -a "$MONITOR_LOG" || true
}

make_telestyle_shards() {
  mkdir -p "$SHARD_DIR"
  "$PY_TEL" - <<PY | tee -a "$MONITOR_LOG"
import json
from pathlib import Path
root=Path('$ROOT')
prompts=json.load(open('$PROMPTS_JSON','r',encoding='utf-8'))
keys=sorted(map(str,prompts.keys()))
shards=[keys[0::2], keys[1::2]]
out=Path('$SHARD_DIR')
out.mkdir(parents=True, exist_ok=True)
for i,ks in enumerate(shards):
    data={k:prompts[k] for k in ks}
    p=out/f'prompts_shard{i}.json'
    json.dump(data, open(p,'w',encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'[shard] telestyle shard{i}: {len(data)} -> {p}')
PY
}

launch_telestyle() {
  make_telestyle_shards
  mkdir -p "$TEL_OUT/logs"
  log "LAUNCH TeleStyle on physical GPUs ${GPU_A},${GPU_B}; out=$TEL_OUT"
  CUDA_VISIBLE_DEVICES="$GPU_A" stdbuf -oL -eL "$PY_TEL" "$TEL_SCRIPT" \
    --cref_dir "$CREF_DIR" \
    --sref_dir "$SREF_DIR" \
    --prompts_json "$SHARD_DIR/prompts_shard0.json" \
    --output_dir "$TEL_OUT" \
    --steps 4 \
    --minedge 1024 \
    --output_resolution "$RES" \
    >> "$TEL_OUT/logs/shard0_gpu${GPU_A}.log" 2>&1 & echo $! > "$TEL_OUT/logs/shard0_gpu${GPU_A}.pid"
  CUDA_VISIBLE_DEVICES="$GPU_B" stdbuf -oL -eL "$PY_TEL" "$TEL_SCRIPT" \
    --cref_dir "$CREF_DIR" \
    --sref_dir "$SREF_DIR" \
    --prompts_json "$SHARD_DIR/prompts_shard1.json" \
    --output_dir "$TEL_OUT" \
    --steps 4 \
    --minedge 1024 \
    --output_resolution "$RES" \
    >> "$TEL_OUT/logs/shard1_gpu${GPU_B}.log" 2>&1 & echo $! > "$TEL_OUT/logs/shard1_gpu${GPU_B}.pid"
  sleep 5
}

wait_telestyle() {
  local retries=0
  mkdir -p "$TEL_OUT/logs"
  while true; do
    local cnt pids alive
    cnt=$(count_png "$TEL_OUT")
    pids=$(pid_csv_from_dir "$TEL_OUT")
    alive=$(alive_csv "$pids")
    status_snapshot "TeleStyle" "$TEL_OUT" "$alive"
    if (( cnt >= TOTAL )); then
      log "DONE TeleStyle png=$cnt/$TOTAL"
      return 0
    fi
    if [[ -z "$alive" ]]; then
      if (( retries >= 2 )); then
        log "ERROR TeleStyle stopped with png=$cnt/$TOTAL after retries=$retries"
        tail -n 80 "$TEL_OUT"/logs/*.log 2>/dev/null | sed 's/^/[telestyle-tail] /' | tee -a "$MONITOR_LOG" || true
        return 1
      fi
      retries=$((retries+1))
      log "WARN TeleStyle not alive with png=$cnt/$TOTAL; restart attempt $retries/2"
      launch_telestyle
    fi
    sleep 300
  done
}

run_single_model_with_retries() {
  local name="$1"
  local out="$2"
  shift 2
  local retries=0
  mkdir -p "$out/logs"
  while true; do
    local cnt
    cnt=$(count_png "$out")
    if (( cnt >= TOTAL )); then
      log "DONE $name already complete png=$cnt/$TOTAL"
      return 0
    fi
    if (( retries > 2 )); then
      log "ERROR $name failed/incomplete png=$cnt/$TOTAL after retries=$((retries-1))"
      tail -n 120 "$out"/logs/run.log 2>/dev/null | sed "s/^/[$name-tail] /" | tee -a "$MONITOR_LOG" || true
      return 1
    fi
    retries=$((retries+1))
    log "LAUNCH $name attempt=$retries on visible physical GPUs $GPUS_PHYS out=$out current_png=$cnt/$TOTAL"
    CUDA_VISIBLE_DEVICES="$GPUS_PHYS" stdbuf -oL -eL "$@" >> "$out/logs/run.log" 2>&1 &
    local pid=$!
    echo "$pid" > "$out/logs/run.pid"
    while kill -0 "$pid" 2>/dev/null; do
      status_snapshot "$name" "$out" "$pid"
      sleep 300
    done
    wait "$pid"
    local rc=$?
    cnt=$(count_png "$out")
    log "EXIT $name pid=$pid rc=$rc png=$cnt/$TOTAL"
    tail -n 80 "$out"/logs/run.log 2>/dev/null | sed "s/^/[$name-tail] /" | tee -a "$MONITOR_LOG" || true
    if (( cnt >= TOTAL )); then
      log "DONE $name png=$cnt/$TOTAL"
      return 0
    fi
    log "WARN $name incomplete after exit; will retry if retries remain"
    sleep 120
  done
}

run_flux() {
  mkdir -p "$FLUX_OUT/logs"
  run_single_model_with_retries "Flux" "$FLUX_OUT" "$PY_SREF" "$FLUX_SCRIPT" \
    --prompts_json "$PROMPTS_JSON" \
    --cref_dir "$CREF_DIR" \
    --sref_dir "$SREF_DIR" \
    --out_dir "$FLUX_OUT" \
    --model_name /mnt/jfs/model_zoo/FLUX.2-klein-9B/ \
    --steps 4 \
    --guidance_scale 1.0 \
    --gpus "$VISIBLE_GPLUS" \
    --output_resolution "$RES" \
    --save_jsonl
}

run_qwen() {
  mkdir -p "$QWEN_OUT/logs"
  run_single_model_with_retries "Qwen" "$QWEN_OUT" "$PY_SREF" "$QWEN_SCRIPT" \
    --prompts_json "$PROMPTS_JSON" \
    --cref_dir "$CREF_DIR" \
    --sref_dir "$SREF_DIR" \
    --out_dir "$QWEN_OUT" \
    --model_name /mnt/jfs/model_zoo/qwen/Qwen-Image-Edit-2511/ \
    --gpus "$VISIBLE_GPLUS" \
    --output_resolution "$RES" \
    --save_jsonl
}

main() {
  log "START three-model monitor tag=$RUN_TAG total=$TOTAL root=$ROOT GPUs=$GPUS_PHYS"
  log "outputs: TEL=$TEL_OUT FLUX=$FLUX_OUT QWEN=$QWEN_OUT"
  wait_telestyle || exit 10
  log "Cooling down 60s before Flux"
  sleep 60
  run_flux || exit 20
  log "Cooling down 60s before Qwen"
  sleep 60
  run_qwen || exit 30
  log "ALL_DONE three models completed. TEL=$(count_png "$TEL_OUT") FLUX=$(count_png "$FLUX_OUT") QWEN=$(count_png "$QWEN_OUT") total=$TOTAL"
}

main "$@"
