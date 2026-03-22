#!/bin/bash
# /data/benchmark_metrics/lora_pipeline/meta/comfyui_rlaunch_flexible.sh -G maintain -c 80 -m 720000 -g 4 -t H100,H800
set -euo pipefail

DEFAULT_GROUP="l40s_yangtong"
DEFAULT_CPU="40"
DEFAULT_GPU_COUNT="8"
DEFAULT_GPU_MODEL="L40S"
DEFAULT_MEMORY="360000"
DEFAULT_IMAGE="hub.i.basemind.com/text2image/comfyui-server:0.1"
DEFAULT_MOUNT="juicefs+s3://oss.i.shaipower.com/lanjinghong-data:/mnt/jfs"
DEFAULT_VOLUME="/data:/data"
DEFAULT_START_SCRIPT="/data/benchmark_metrics/lora_pipeline/meta/comfyui_start_new_server.sh"
DEFAULT_RETRY_SLEEP="3"

usage() {
  cat <<'EOF'
用法:
  comfyui_rlaunch_flexible.sh [选项]

选项:
  -G <group>           资源组名，默认: l40s_yangtong
  -c <cpu>             CPU 数量(正整数)，默认: 40
  -g <gpu>             GPU 配置，支持:
                       1) 仅数量: 8
                       2) 型号+数量: L40S:8
                       默认: L40S:8
  -m <memory>          内存大小(正整数，单位与 brainctl 保持一致)，默认: 360000
  -t <tag1,tag2...>    正向标签，支持逗号分隔多值，也可重复传入 -t
                       默认: L40S
  --once               仅执行一次 launch（默认无限重试）
  --dry-run            仅打印最终 brainctl 命令，不实际执行
  -h                   显示帮助

示例:
  bash comfyui_rlaunch_flexible.sh
  bash comfyui_rlaunch_flexible.sh -G maintain -c 64 -g H100:8 -m 500000 -t H100,RDMA
  bash comfyui_rlaunch_flexible.sh -g 4 -t L40S --once
EOF
}

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_valid_group() {
  [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]]
}

GROUP="$DEFAULT_GROUP"
CPU="$DEFAULT_CPU"
GPU_COUNT="$DEFAULT_GPU_COUNT"
GPU_MODEL="$DEFAULT_GPU_MODEL"
MEMORY="$DEFAULT_MEMORY"
TAGS=()
USER_SET_TAGS=false
ONCE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -G)
      [[ $# -ge 2 ]] || { echo "错误: -G 需要参数" >&2; usage; exit 2; }
      GROUP="$2"
      shift 2
      ;;
    -c)
      [[ $# -ge 2 ]] || { echo "错误: -c 需要参数" >&2; usage; exit 2; }
      CPU="$2"
      shift 2
      ;;
    -g)
      [[ $# -ge 2 ]] || { echo "错误: -g 需要参数" >&2; usage; exit 2; }
      gpu_arg="$2"
      if [[ "$gpu_arg" =~ ^[1-9][0-9]*$ ]]; then
        GPU_COUNT="$gpu_arg"
      elif [[ "$gpu_arg" =~ ^([A-Za-z0-9._-]+):([1-9][0-9]*)$ ]]; then
        GPU_MODEL="${BASH_REMATCH[1]}"
        GPU_COUNT="${BASH_REMATCH[2]}"
      else
        echo "错误: -g 格式无效，需为 <count> 或 <MODEL:count>" >&2
        usage
        exit 2
      fi
      shift 2
      ;;
    -m)
      [[ $# -ge 2 ]] || { echo "错误: -m 需要参数" >&2; usage; exit 2; }
      MEMORY="$2"
      shift 2
      ;;
    -t)
      [[ $# -ge 2 ]] || { echo "错误: -t 需要参数" >&2; usage; exit 2; }
      USER_SET_TAGS=true
      IFS=',' read -r -a _parts <<< "$2"
      for p in "${_parts[@]}"; do
        tag="$(echo "$p" | xargs)"
        [[ -n "$tag" ]] && TAGS+=("$tag")
      done
      shift 2
      ;;
    --once)
      ONCE=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "错误: 未知参数 $1" >&2
      usage
      exit 2
      ;;
  esac
done

is_valid_group "$GROUP" || { echo "错误: group 无效: $GROUP" >&2; exit 2; }
is_positive_int "$CPU" || { echo "错误: cpu 必须是正整数: $CPU" >&2; exit 2; }
is_positive_int "$GPU_COUNT" || { echo "错误: gpu 数量必须是正整数: $GPU_COUNT" >&2; exit 2; }
is_positive_int "$MEMORY" || { echo "错误: memory 必须是正整数: $MEMORY" >&2; exit 2; }
[[ "$GPU_MODEL" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "错误: GPU 型号无效: $GPU_MODEL" >&2; exit 2; }

if [[ "$USER_SET_TAGS" == false ]]; then
  TAGS=("$GPU_MODEL")
fi

has_gpu_model=false
for t in "${TAGS[@]}"; do
  if [[ "$t" == "$GPU_MODEL" ]]; then
    has_gpu_model=true
    break
  fi
done
if [[ "$has_gpu_model" == false ]]; then
  TAGS+=("$GPU_MODEL")
fi

dedup_tags=()
for t in "${TAGS[@]}"; do
  exists=false
  for d in "${dedup_tags[@]}"; do
    if [[ "$d" == "$t" ]]; then
      exists=true
      break
    fi
  done
  [[ "$exists" == false ]] && dedup_tags+=("$t")
done
TAGS=("${dedup_tags[@]}")
POSITIVE_TAGS="$(IFS=','; echo "${TAGS[*]}")"

launch_once() {
  local cmd=(
    brainctl launch
    "--charged-group=$GROUP"
    --private-machine=yes
    "--cpu=$CPU"
    "--gpu=$GPU_COUNT"
    "--memory=$MEMORY"
    "--positive-tags" "$POSITIVE_TAGS"
    "--image=$DEFAULT_IMAGE"
    "--mount=$DEFAULT_MOUNT"
    --volume "$DEFAULT_VOLUME"
    --i-know-i-am-wasting-resource=false
    --host-network=true
    --custom-resources rdma/mlnx_shared=8
    --entrypoint=""
    --
    bash "$DEFAULT_START_SCRIPT"
  )
  if [[ "$DRY_RUN" == true ]]; then
    echo "DRY_RUN CMD: ${cmd[*]}"
    return 0
  fi
  "${cmd[@]}"
}

while true; do
  launch_once
  if [[ "$ONCE" == true ]]; then
    break
  fi
  echo "rlaunch 任务已退出，${DEFAULT_RETRY_SLEEP}秒后重试..."
  sleep "$DEFAULT_RETRY_SLEEP"
done
