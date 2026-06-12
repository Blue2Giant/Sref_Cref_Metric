#!/usr/bin/env bash
set -euo pipefail

model=${MODEL:-qwen3vlw8a8@}
base_url=${BASE_URL:-http://stepcloud-apisix-gateway-eval.i-stepfun.com/Qwen3-VL-235B-A22B-W8A8/v1}
num_procs=${NUM_PROCS:-256}
judge_times=${JUDGE_TIMES:-3}
min_true=${MIN_TRUE:-2}
num_samples=${NUM_SAMPLES:-0}
max_images_per_key=${MAX_IMAGES_PER_KEY:-0}
keep_empty_keys=${KEEP_EMPTY_KEYS:-1}
probe_timeout=${PROBE_TIMEOUT:-3.0}
out_root=${OUT_ROOT:-/mnt/jfs/logs/illustrious_similar_people_binary_judge_20260410}
overwrite=${OVERWRITE:-0}

script_dir=/data/benchmark_metrics/lora_pipeline/similarity_judge
py_script="${script_dir}/qwen_similar_people_binary_judge.py"

# Active qwen tmux server windows checked on 2026-04-08.
ENDPOINTS=(
    "Qwen3-VL-30B-A3B-Instruct@http://10.204.18.58:22002/v1"
    "qwen3vlw8a8@http://stepcloud-apisix-gateway-eval.i-stepfun.com/Qwen3-VL-235B-A22B-W8A8/v1"
    "qwen35-27b@http://stepcast-router.shai-core:9200/v1"
)

endpoint_args=()
for endpoint in "${ENDPOINTS[@]}"; do
    endpoint_args+=(--endpoint "$endpoint")
done

inputs=(
    /data/benchmark_metrics/lora_pipeline/meta/triplet_jsonls/illustrious_content_one_lora.jsonl
    /data/benchmark_metrics/lora_pipeline/meta/triplet_jsonls/illustrious_dual_lora_style_content_filtered.jsonl
)

extra_args=()
if [ "$keep_empty_keys" = "1" ]; then
    extra_args+=(--keep_empty_keys)
fi

mkdir -p "$out_root"

for input_jsonl in "${inputs[@]}"; do
    name=$(basename "$input_jsonl" .jsonl)
    out_dir="${out_root}/${name}"
    expected_outputs=(
        "${out_dir}/similar_people_all.json"
        "${out_dir}/similar_people_bad.json"
        "${out_dir}/similar_people_good.json"
        "${out_dir}/similar_people_true.jsonl"
        "${out_dir}/similar_people_false.jsonl"
        "${out_dir}/similar_people_error.jsonl"
        "${out_dir}/similar_people_detail.json"
    )
    existing_count=0
    for output_path in "${expected_outputs[@]}"; do
        if [ -f "$output_path" ]; then
            existing_count=$((existing_count + 1))
        fi
    done

    if [ "$overwrite" = "1" ]; then
        if [ -d "$out_dir" ]; then
            echo "[OVERWRITE] removing ${out_dir}"
            rm -rf "$out_dir"
        fi
        mkdir -p "$out_dir"
    else
        if [ "$existing_count" -eq "${#expected_outputs[@]}" ]; then
            echo "[SKIP] input_jsonl=${input_jsonl}"
            echo "[SKIP] all outputs already exist under ${out_dir}"
            continue
        fi
        if [ "$existing_count" -gt 0 ]; then
            echo "[ERROR] partial outputs already exist under ${out_dir}"
            echo "[ERROR] found ${existing_count}/${#expected_outputs[@]} expected files"
            echo "[ERROR] set OVERWRITE=1 to rerun from scratch"
            exit 1
        fi
        mkdir -p "$out_dir"
    fi

    echo "[RUN] input_jsonl=${input_jsonl}"
    echo "[RUN] out_dir=${out_dir}"
    echo "[RUN] overwrite=${overwrite}"

    python3 "$py_script" \
        --input-jsonl "$input_jsonl" \
        --jsonl-task-mode aggregate_key \
        --num_samples "$num_samples" \
        --max_images_per_key "$max_images_per_key" \
        --judge_times "$judge_times" \
        --min_true "$min_true" \
        --model "$model" \
        --base_url "$base_url" \
        --probe-timeout "$probe_timeout" \
        --num_procs "$num_procs" \
        "${endpoint_args[@]}" \
        --out_all "${out_dir}/similar_people_all.json" \
        --out_pos "${out_dir}/similar_people_bad.json" \
        --out_neg "${out_dir}/similar_people_good.json" \
        --out_true_jsonl "${out_dir}/similar_people_true.jsonl" \
        --out_false_jsonl "${out_dir}/similar_people_false.jsonl" \
        --out_error_jsonl "${out_dir}/similar_people_error.jsonl" \
        --out_detail "${out_dir}/similar_people_detail.json" \
        "${extra_args[@]}"
done

echo "[DONE] out_root=${out_root}"
