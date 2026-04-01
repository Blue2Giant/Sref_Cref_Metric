#!/bin/bash

DEFAULT_GROUPS=("l40s_yangtong" "buffer" "maintain")
DEFAULT_GROUP="buffer"
DEFAULT_GPU=8
DEFAULT_CPU=64
DEFAULT_MEMORY=1000000

show_help() {
    echo "Usage: $0 [-G group] [-g gpu] [-c cpu] [-m memory]"
    echo "  -G: charged-group, one of: ${DEFAULT_GROUPS[*]} (default: ${DEFAULT_GROUP})"
    echo "  -g: GPU count (default: ${DEFAULT_GPU})"
    echo "  -c: CPU count (default: ${DEFAULT_CPU})"
    echo "  -m: memory in MB (default: ${DEFAULT_MEMORY})"
    exit 0
}

while getopts "G:g:c:m:h" opt; do
    case "$opt" in
        G) GROUP="$OPTARG" ;;
        g) GPU="$OPTARG" ;;
        c) CPU="$OPTARG" ;;
        m) MEMORY="$OPTARG" ;;
        h) show_help ;;
        *) show_help ;;
    esac
done

GPU=${GPU:-$DEFAULT_GPU}
CPU=${CPU:-$DEFAULT_CPU}
MEMORY=${MEMORY:-$DEFAULT_MEMORY}
GROUP=${GROUP:-$DEFAULT_GROUP}

valid_group=false
for g in "${DEFAULT_GROUPS[@]}"; do
    if [ "$GROUP" = "$g" ]; then
        valid_group=true
        break
    fi
done

if [ "$valid_group" != true ]; then
    echo "invalid group: $GROUP, must be one of: ${DEFAULT_GROUPS[*]}" >&2
    exit 1
fi


echo "charged-group : $GROUP"
echo "GPU           : $GPU"
echo "CPU           : $CPU"
echo "Memory (MB)   : $MEMORY"

while true; do
    brainctl launch --charged-group="$GROUP" --private-machine=yes \
        --cpu="$CPU" --gpu="$GPU" --memory="$MEMORY" --positive-tags "H100,H800" \
        --negative-tags gpu-h100-0290.host.platform.shaipower.com \
        --negative-tags gpu-h100-0452.host.platform.shaipower.com \
        --mount=juicefs+s3://oss.i.shaipower.com/lanjinghong-data:/mnt/jfs \
        --i-know-i-am-wasting-resource=false \
        --custom-resources rdma/mlnx_shared=8 --entrypoint="" -- bash   
    echo "rlaunch 任务已退出，3秒后重试..."
    sleep 3
done

