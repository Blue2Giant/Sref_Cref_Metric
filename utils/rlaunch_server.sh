#!/bin/bash
# -----------------------------------------------
# 默认值
DEFAULT_GROUPS=("buffer" "maintain")
DEFAULT_GPU="4"
DEFAULT_CPU="64"
DEFAULT_MEMORY="1000000" # 保持原始脚本的单位，这里是MB
DEFAULT_PRIORITY="1"

# 显示帮助信息
function show_help {
    echo "Usage: $0 [-g gpu] [-c cpu] [-m memory] [-p priority] [-x]"
    echo "  -g: 指定 GPU 数量（默认: ${DEFAULT_GPU}）"
    echo "  -c: 指定 CPU 数量（默认: ${DEFAULT_CPU}）"
    echo "  -m: 指定内存大小（默认: ${DEFAULT_MEMORY} MB）"
    echo "  -p: 指定机器数量（默认: ${DEFAULT_PRIORITY}）"
    echo "  -x: 显示默认组并交互选择"
    exit 0
}

# 解析命令行参数
while getopts "g:c:m:p:xh" opt; do
  case $opt in
    g) GPU="$OPTARG" ;;
    c) CPU="$OPTARG" ;;
    m) MEMORY="$OPTARG" ;;
    p) PRIORITY="$OPTARG" ;;
    x) SHOW_GROUPS=true ;;
    h) show_help ;;
    *) echo "Invalid option: -$OPTARG" >&2
       exit 1 ;;
  esac
done

# 如果没有设置参数，则使用默认值
GPU="${GPU:-$DEFAULT_GPU}"
CPU="${CPU:-$DEFAULT_CPU}"
MEMORY="${MEMORY:-$DEFAULT_MEMORY}"
PRIORITY="${PRIORITY:-$DEFAULT_PRIORITY}"

# 如果启用 -x 标签，显示默认组并交互选择
if [[ $SHOW_GROUPS == true ]]; then
    echo "请选择要执行的组："
    select GROUP in "${DEFAULT_GROUPS[@]}"; do
        if [[ -n $GROUP ]]; then
            echo "已选择组: $GROUP"
            break
        else
            echo "无效选择，请重新选择。"
        fi
    done
else
    # 如果没有启用 -x 标签，则使用默认的第一个组
    GROUP="${DEFAULT_GROUPS[0]}"
fi

echo "提交任务到组: $GROUP"
echo "GPU 数量: $GPU"
echo "CPU 数量: $CPU"
echo "内存大小: $MEMORY"
echo "机器数量: $PRIORITY"

output=$(rlaunch --group "$GROUP" --positive-tags H100,H200,H800 --predict-only --predict-node-num=50 2>&1)
# 检查 rlaunch 命令本身的退出状态码
if [ $? -ne 0 ]; then
    echo "ERROR: rlaunch --predict-only command failed. Output:" >&2
    echo "$output" >&2
    exit 1
fi

selected_node=$(echo "$output" | awk -v req_gpu="$GPU" -v req_cpu="$CPU" -v req_mem="$MEMORY" '
BEGIN {
    min_gpu = 999999
    selected = ""
}
/^Node: / {
    # 提取节点信息
    node = ""
    cpu_val = 0
    gpu_val = 0
    mem_val = 0

    # 解析节点行
    split($0, parts, ",")
    for (i in parts) {
        if (parts[i] ~ /^Node: /) {
            sub(/^Node: /, "", parts[i])
            sub(/ $/, "", parts[i])   # 移除尾部空格
            node = parts[i]
        }
        else if (parts[i] ~ /CPU: /) {
            split(parts[i], cpu_parts, /: /)
            cpu_val = cpu_parts[2] + 0   # 转换为数字
        }
        else if (parts[i] ~ /GPU: /) {
            split(parts[i], gpu_parts, /: /)
            gpu_val = gpu_parts[2] + 0   # 转换为数字
        }
        else if (parts[i] ~ /Memory: /) {
            split(parts[i], mem_parts, /: /)
            # 移除" GiB"后缀并转换为数字（单位GiB）
            sub(/ GiB$/, "", mem_parts[2])
            mem_val = mem_parts[2] + 0   # 转换为数字
        }
    }

    # 检查资源是否满足要求（req_mem是MB，mem_val是GiB）
    if (gpu_val >= req_gpu && cpu_val >= req_cpu && (mem_val * 1024) >= req_mem) {
        # 更新最小GPU节点
        if (gpu_val < min_gpu) {
            min_gpu = gpu_val
            selected = node
        }
    }
}
END {
    if (selected != "") {
        print selected
    }
    else {
        # 如果没有找到合适的节点，打印错误到stderr并退出
        print "ERROR: No suitable node found based on requirements (GPU=" req_gpu ", CPU=" req_cpu ", Memory=" req_mem " MB)." > "/dev/stderr"
        exit 1
    }
}')

if [ -z "$selected_node" ]; then
    echo "WARNING: selected_node is empty after awk processing. Using default tags." >&2
    selected_node="H200,H800,H100"
    echo "selected_node fallback: $selected_node"
else
    # 转换为 'node/nodename' 格式，因为 rlaunch 期望这样
    selected_node="node/$selected_node"
    echo "Selected node for task: $selected_node"
fi

# =========================
# 下面这块改成循环启动
# =========================

for ((i=1; i<=PRIORITY; i++)); do
    echo "=============================="
    echo "启动第 $i/$PRIORITY 个任务..."
    echo "使用节点标签: $selected_node"
    echo "=============================="

    # 把最终使用的变量打印出来（仅做展示）
    echo """
rlaunch --charged-group=\"$GROUP\" --private-machine=yes \\
  --cpu=\"$CPU\" --gpu=\"$GPU\" --memory=\"$MEMORY\" --positive-tags \"$selected_node\" \\
  --image hub.i.basemind.com/stepcast/stepcast:openvllm-qwen3vl-0925 \\\\n
  --negative-tags node/gpu-h100-0791.host.platform.shaipower.com \\\\n
  --mount=juicefs+s3://oss.i.shaipower.com/lanjinghong-data:/mnt/jfs \\\\n
  --volume /data/Depth-Anything:/data/Depth-Anything \\\\n
  --host-network=true \\\\n
  --preemptible=yes \\\\n
  --entrypoint=\"\" -- bash /data/LoraPipeline/utils/qwen3_label_and_generation_prompt.sh
"""

    # 真正提交任务
    rlaunch --charged-group="$GROUP" --private-machine=yes \
      --cpu="$CPU" --gpu="$GPU" --memory="$MEMORY" --positive-tags "$selected_node" \
      --image=hub.i.basemind.com/stepcast/stepcast:openvllm-qwen3vl-0925 \
      --positive-tags gpu-h800-0599.host.platform.shaipower.com \
      --mount=juicefs+s3://oss.i.shaipower.com/lanjinghong-data:/mnt/jfs \
      --volume /data/Depth-Anything:/data/Depth-Anything \
      --volume /data/LoraPipeline:/data/LoraPipeline \
      --host-network=true \
      --preemptible=yes \
      --enable-sshd=false \
      --entrypoint="" -- bash /data/LoraPipeline/utils/qwen3_label_and_generation_prompt.sh

    # 如果你希望在 rlaunch 失败时也退出脚本
    if [ $? -ne 0 ]; then
        echo "ERROR: rlaunch command failed at iteration $i with exit code $?. Aborting." >&2
        exit 1
    fi
done

# # pip install openai
# python /data/Depth-Anything/qwen-3/request_qwen3_loop.py
