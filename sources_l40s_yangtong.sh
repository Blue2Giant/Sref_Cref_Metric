#!/bin/bash

# 仅探测 l40s_yangtong 组
groups=("l40s_yangtong")

# 初始化关联数组来存储每个组的 GPU 统计信息
declare -A group_gpu_info

for group in "${groups[@]}"; do
    output=$(brainctl launch --group "$group" --positive-tags L40S --predict-only --predict-node-num=50)

    declare -A gpu_count=()
    declare -A gpu_cpu_min=()
    declare -A gpu_cpu_max=()
    declare -A gpu_memory_min=()
    declare -A gpu_memory_max=()

    while IFS= read -r line; do
        gpu=$(echo "$line" | grep -oP 'GPU: \K\d+')
        cpu=$(echo "$line" | grep -oP 'CPU: \K\d+')
        memory=$(echo "$line" | grep -oP 'Memory: \K[\d.]+')

        if [[ -z $cpu || -z $memory || -z $gpu ]]; then
            continue
        fi

        if [[ -z ${gpu_count[$gpu]} ]]; then
            gpu_count[$gpu]=0
            gpu_cpu_min[$gpu]=$cpu
            gpu_cpu_max[$gpu]=$cpu
            gpu_memory_min[$gpu]=$memory
            gpu_memory_max[$gpu]=$memory
        fi

        ((gpu_count[$gpu]++))

        if [[ $cpu -lt ${gpu_cpu_min[$gpu]} ]]; then
            gpu_cpu_min[$gpu]=$cpu
        fi
        if [[ $cpu -gt ${gpu_cpu_max[$gpu]} ]]; then
            gpu_cpu_max[$gpu]=$cpu
        fi

        if (( $(echo "$memory < ${gpu_memory_min[$gpu]}" | bc -l) )); then
            gpu_memory_min[$gpu]=$memory
        fi
        if (( $(echo "$memory > ${gpu_memory_max[$gpu]}" | bc -l) )); then
            gpu_memory_max[$gpu]=$memory
        fi
    done <<< "$output"

    group_gpu_info["$group"]=$(for gpu in "${!gpu_count[@]}"; do
        echo "$gpu:${gpu_count[$gpu]}:${gpu_cpu_min[$gpu]}:${gpu_cpu_max[$gpu]}:${gpu_memory_min[$gpu]}:${gpu_memory_max[$gpu]}"
    done | sort -n -t: -k1)
done

echo "Detailed Summary of GPU, CPU, and Memory counts for l40s_yangtong (L40S only):"
echo "=============================================================================================="
printf "%-15s | %-10s | %-13s | %-15s | %-20s\n" "Group" "GPU Count" "Machine Count" "CPU Range" "Memory Range (GiB)"
echo "=============================================================================================="

for group in "${groups[@]}"; do
    stats="${group_gpu_info[$group]}"
    if [[ -z $stats ]]; then
        echo "No data available for group: $group"
        continue
    fi
    while IFS=: read -r gpu machine_count cpu_min cpu_max memory_min memory_max; do
        if [[ $gpu -eq 8 ]]; then
            printf "\033[1m%-15s | %-10s | %-13s | %-15s | %-20s\033[0m\n" "$group" "$gpu" "$machine_count" "$cpu_min - $cpu_max" "$memory_min - $memory_max"
        else
            printf "%-15s | %-10s | %-13s | %-15s | %-20s\n" "$group" "$gpu" "$machine_count" "$cpu_min - $cpu_max" "$memory_min - $memory_max"
        fi
    done <<< "$stats"
done
echo "=============================================================================================="
