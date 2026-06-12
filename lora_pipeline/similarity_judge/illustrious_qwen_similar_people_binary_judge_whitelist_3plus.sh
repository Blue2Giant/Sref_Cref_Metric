#!/usr/bin/env bash
set -euo pipefail

num_procs=${NUM_PROCS:-256}
judge_times=${JUDGE_TIMES:-3}
min_true=${MIN_TRUE:-2}
min_similar_people=${MIN_SIMILAR_PEOPLE:-3}
probe_timeout=${PROBE_TIMEOUT:-3.0}
source_out_root=${SOURCE_OUT_ROOT:-/mnt/jfs/logs/illustrious_similar_people_binary_judge_20260410}
out_root=${OUT_ROOT:-/mnt/jfs/logs/illustrious_similar_people_binary_judge_20260410_whitelist_3plus_20260411}
overwrite=${OVERWRITE:-0}
whitelist_ids=${WHITELIST_IDS:-1284773,1294263,620449,707224}

script_dir=/data/benchmark_metrics/lora_pipeline/similarity_judge
judge_py_script="${script_dir}/qwen_similar_people_binary_judge.py"
helper_py_script="/data/benchmark_metrics/lora_pipeline/tools/rejudge_similar_people_outputs_with_whitelist.py"

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

mkdir -p "$out_root" "$out_root/_rerun_whitelist_3plus" "$out_root/_subsets" "$out_root/_summaries"

for input_jsonl in "${inputs[@]}"; do
    name=$(basename "$input_jsonl" .jsonl)
    source_dir="${source_out_root}/${name}"
    output_dir="${out_root}/${name}"
    rerun_dir="${out_root}/_rerun_whitelist_3plus/${name}"
    subset_jsonl="${out_root}/_subsets/${name}.whitelist_bad_keys.jsonl"
    summary_json="${out_root}/_summaries/${name}.summary.json"

    expected_outputs=(
        "${output_dir}/similar_people_all.json"
        "${output_dir}/similar_people_bad.json"
        "${output_dir}/similar_people_good.json"
        "${output_dir}/similar_people_true.jsonl"
        "${output_dir}/similar_people_false.jsonl"
        "${output_dir}/similar_people_error.jsonl"
        "${output_dir}/similar_people_detail.json"
        "${summary_json}"
    )

    existing_count=0
    for output_path in "${expected_outputs[@]}"; do
        if [ -f "$output_path" ]; then
            existing_count=$((existing_count + 1))
        fi
    done

    if [ "$overwrite" = "1" ]; then
        rm -rf "$output_dir" "$rerun_dir"
        rm -f "$subset_jsonl" "$summary_json"
    else
        if [ "$existing_count" -eq "${#expected_outputs[@]}" ]; then
            echo "[SKIP] input_jsonl=${input_jsonl}"
            echo "[SKIP] all outputs already exist under ${output_dir}"
            continue
        fi
        if [ "$existing_count" -gt 0 ]; then
            echo "[ERROR] partial outputs already exist for ${name}"
            echo "[ERROR] found ${existing_count}/${#expected_outputs[@]} expected files"
            echo "[ERROR] set OVERWRITE=1 to rerun from scratch"
            exit 1
        fi
    fi

    echo "[RUN] input_jsonl=${input_jsonl}"
    echo "[RUN] source_dir=${source_dir}"
    echo "[RUN] output_dir=${output_dir}"
    echo "[RUN] whitelist_ids=${whitelist_ids}"
    echo "[RUN] min_similar_people=${min_similar_people}"

    python3 "$helper_py_script" \
        --input-jsonl "$input_jsonl" \
        --source-output-dir "$source_dir" \
        --output-dir "$output_dir" \
        --rerun-dir "$rerun_dir" \
        --subset-jsonl "$subset_jsonl" \
        --summary-json "$summary_json" \
        --judge-script "$judge_py_script" \
        --whitelist-ids "$whitelist_ids" \
        --judge-times "$judge_times" \
        --min-true "$min_true" \
        --min-similar-people "$min_similar_people" \
        --num-procs "$num_procs" \
        --probe-timeout "$probe_timeout" \
        "${endpoint_args[@]}"
done

python3 - <<'PY' "$out_root"
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
summary_dir = out_root / "_summaries"
overview = []
for path in sorted(summary_dir.glob("*.summary.json")):
    with path.open("r", encoding="utf-8") as f:
        overview.append(json.load(f))
out_path = out_root / "whitelist_rejudge_overview.json"
with out_path.open("w", encoding="utf-8") as f:
    json.dump(overview, f, ensure_ascii=False, indent=2)
print(f"[OVERVIEW] -> {out_path}")
PY

echo "[DONE] out_root=${out_root}"
