#!/bin/bash

set -uo pipefail
trap '' HUP

DEFAULT_GROUP="buffer"
DEFAULT_CPU="40"
DEFAULT_GPU="0"
DEFAULT_MEMORY="800000"
DEFAULT_RETRY_SLEEP="3"

IMAGE="hub.i.basemind.com/stepcast/stepcast:openvllm-qwen3vl-0925"
MOUNT="juicefs+s3://oss.i.shaipower.com/lanjinghong-data:/mnt/jfs"
VOLUME="/data/:/data/"
START_SCRIPT="/data/benchmark_metrics/lora_pipeline/meta/start_qwen_server.sh"
POSITIVE_TAGS_DEFAULT="H100,H200,H800,L40S"
POSITIVE_TAGS_PINNED="gpu-h800-0187.host.platform.shaipower.com"
NEGATIVE_TAGS_PINNED="gpu-h100-0107.host.platform.shaipower.com"

GROUP="$DEFAULT_GROUP"
CPU="$DEFAULT_CPU"
GPU="$DEFAULT_GPU"
MEMORY="$DEFAULT_MEMORY"
shutdown_requested=false
current_pid=""

usage() {
    cat <<EOF
用法:
  bash qwen3_30B_server_demo.sh [-G group] [-c cpu] [-g gpu] [-m memory]

选项:
  -G <group>    指定 charged group，默认: ${DEFAULT_GROUP}
  -c <cpu>      指定 CPU 数量，默认: ${DEFAULT_CPU}
  -g <gpu>      指定 GPU 数量，默认: ${DEFAULT_GPU}
  -m <memory>   指定内存大小，默认: ${DEFAULT_MEMORY}
  -h            显示帮助

示例:
  bash qwen3_30B_server_demo.sh
  bash qwen3_30B_server_demo.sh -G maintain -c 80 -g 8 -m 900000
EOF
}

is_positive_int() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_valid_group() {
    [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]]
}

stop_current_job() {
    shutdown_requested=true
    echo "收到中断信号，准备停止当前任务并退出..." >&2
    if [[ -n "$current_pid" ]] && kill -0 "$current_pid" 2>/dev/null; then
        kill -TERM -- "-${current_pid}" 2>/dev/null || kill -TERM "$current_pid" 2>/dev/null || true
        sleep 1
        if kill -0 "$current_pid" 2>/dev/null; then
            kill -KILL -- "-${current_pid}" 2>/dev/null || kill -KILL "$current_pid" 2>/dev/null || true
        fi
    fi
}

while getopts ":G:c:g:m:h" opt; do
    case "$opt" in
        G)
            GROUP="$OPTARG"
            ;;
        c)
            CPU="$OPTARG"
            ;;
        g)
            GPU="$OPTARG"
            ;;
        m)
            MEMORY="$OPTARG"
            ;;
        h)
            usage
            exit 0
            ;;
        :)
            echo "错误: -$OPTARG 需要参数" >&2
            usage
            exit 2
            ;;
        \?)
            echo "错误: 未知参数 -$OPTARG" >&2
            usage
            exit 2
            ;;
    esac
done
shift $((OPTIND - 1))

if [[ $# -gt 0 ]]; then
    echo "错误: 存在未知位置参数: $*" >&2
    usage
    exit 2
fi

is_valid_group "$GROUP" || { echo "错误: group 无效: $GROUP" >&2; exit 2; }
is_positive_int "$CPU" || { echo "错误: cpu 必须是正整数: $CPU" >&2; exit 2; }
is_positive_int "$GPU" || { echo "错误: gpu 必须是正整数: $GPU" >&2; exit 2; }
is_positive_int "$MEMORY" || { echo "错误: memory 必须是正整数: $MEMORY" >&2; exit 2; }

trap stop_current_job INT TERM

launch_once() {
    local cmd=(
        rlaunch
        "--charged-group=${GROUP}"
        --private-machine=yes
        "--cpu=${CPU}"
        "--gpu=${GPU}"
        "--memory=${MEMORY}"
        --positive-tags "${POSITIVE_TAGS_DEFAULT}"
        --negative-tags "${NEGATIVE_TAGS_PINNED}"
        --positive-tags "${POSITIVE_TAGS_PINNED}"
        "--image=${IMAGE}"
        "--mount=${MOUNT}"
        --volume "${VOLUME}"
        --host-network=true
        --enable-sshd=false
        --entrypoint=""
        --
        bash "${START_SCRIPT}"
    )

    echo "启动参数: group=${GROUP} cpu=${CPU} gpu=${GPU} memory=${MEMORY}"
    setsid "${cmd[@]}" &
    current_pid=$!

    if wait "$current_pid"; then
        current_pid=""
        return 0
    fi

    local code=$?
    current_pid=""
    return "$code"
}

while true; do
    if [[ "$shutdown_requested" == true ]]; then
        break
    fi

    if launch_once; then
        echo "brainctl 任务完成，准备下一轮。"
    else
        code=$?
        if [[ "$shutdown_requested" == true ]]; then
            break
        fi
        echo "brainctl 任务异常退出(ExitCode=${code})，按原参数重试..."
    fi

    if [[ "$shutdown_requested" == true ]]; then
        break
    fi

    echo "brainctl 任务已退出，${DEFAULT_RETRY_SLEEP}秒后重试..."
    sleep "${DEFAULT_RETRY_SLEEP}" &
    current_pid=$!
    wait "$current_pid" 2>/dev/null || true
    current_pid=""
done

echo "脚本已停止。"
