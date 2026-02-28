#!/bin/bash

# 定义要查询的组
groups=("l40s_yangtong" "maintain" "buffer")

# 初始化关联数组来存储每个组的 GPU 统计信息
declare -A group_gpu_info
# --positive-tags H100
# 遍历每个组
for group in "${groups[@]}"; do
    # 保存 rlaunch 的输出
    output=$(brainctl launch --group $group --positive-tags H200,H100,H800 --predict-only --predict-node-num=50)
    #--positive-tags 
    # 初始化当前组的 GPU 统计信息
    declare -A gpu_count=()
    declare -A gpu_cpu_min=()
    declare -A gpu_cpu_max=()
    declare -A gpu_memory_min=()
    declare -A gpu_memory_max=()

    # 处理每一行
    while IFS= read -r line; do
        # 提取 GPU 数量
        gpu=$(echo "$line" | grep -oP 'GPU: \K\d+')
        # 跳过 GPU 为 0 的行
        # if [[ -z $gpu || $gpu -eq 0 ]]; then
        #     continue
        # fi

        # 提取 CPU 数量
        cpu=$(echo "$line" | grep -oP 'CPU: \K\d+')
        # 提取内存大小（假设单位为 GiB）
        memory=$(echo "$line" | grep -oP 'Memory: \K[\d.]+')

        # 如果 CPU 或内存为空，则跳过该行
        if [[ -z $cpu || -z $memory ]]; then
            continue
        fi

        # 初始化 GPU 的统计信息（如果尚未初始化）
        if [[ -z ${gpu_count[$gpu]} ]]; then
            gpu_count[$gpu]=0
            gpu_cpu_min[$gpu]=$cpu
            gpu_cpu_max[$gpu]=$cpu
            gpu_memory_min[$gpu]=$memory
            gpu_memory_max[$gpu]=$memory
        fi

        # 增加对应 GPU 数量的机器计数
        ((gpu_count[$gpu]++))

        # 更新 CPU 区间
        if [[ $cpu -lt ${gpu_cpu_min[$gpu]} ]]; then
            gpu_cpu_min[$gpu]=$cpu
        fi
        if [[ $cpu -gt ${gpu_cpu_max[$gpu]} ]]; then
            gpu_cpu_max[$gpu]=$cpu
        fi

        # 更新内存区间
        if (( $(echo "$memory < ${gpu_memory_min[$gpu]}" | bc -l) )); then
            gpu_memory_min[$gpu]=$memory
        fi
        if (( $(echo "$memory > ${gpu_memory_max[$gpu]}" | bc -l) )); then
            gpu_memory_max[$gpu]=$memory
        fi
    done <<< "$output"

    # 将当前组的 GPU 统计信息保存到全局数组中
    group_gpu_info["$group"]=$(for gpu in "${!gpu_count[@]}"; do
        echo "$gpu:${gpu_count[$gpu]}:${gpu_cpu_min[$gpu]}:${gpu_cpu_max[$gpu]}:${gpu_memory_min[$gpu]}:${gpu_memory_max[$gpu]}"
    done | sort -n -t: -k1)
done

# 输出所有组的 GPU 统计信息（按 GPU 数量排序）
echo "Detailed Summary of GPU, CPU, and Memory counts for all groups:"
echo "=============================================================================================="
printf "%-15s | %-10s | %-13s | %-15s | %-20s\n" "Group" "GPU Count" "Machine Count" "CPU Range" "Memory Range (GiB)"
echo "=============================================================================================="

for group in "${groups[@]}"; do
    # 获取当前组的统计信息
    stats="${group_gpu_info[$group]}"
    if [[ -z $stats ]]; then
        echo "No data available for group: $group"
        continue
    fi

    # 输出当前组的统计信息
    while IFS=: read -r gpu machine_count cpu_min cpu_max memory_min memory_max; do
        # 如果 GPU 数量为 8，则加粗显示
        if [[ $gpu -eq 8 ]]; then
            printf "\033[1m%-15s | %-10s | %-13s | %-15s | %-20s\033[0m\n" "$group" "$gpu" "$machine_count" "$cpu_min - $cpu_max" "$memory_min - $memory_max"
        else
            printf "%-15s | %-10s | %-13s | %-15s | %-20s\n" "$group" "$gpu" "$machine_count" "$cpu_min - $cpu_max" "$memory_min - $memory_max"
        fi
    done <<< "$stats"
    echo "---------------------------------------------------------------------------------------------"
done
echo "=============================================================================================="
