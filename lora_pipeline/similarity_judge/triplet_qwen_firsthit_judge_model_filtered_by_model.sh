#!/usr/bin/env bash
set -euo pipefail

repo_root=/data/benchmark_metrics
python_bin=${PYTHON_BIN:-python}
targets_csv=${TARGETS:-illustrious,flux,qwen}
prepared_dir=${PREPARED_TRIPLET_DIR:-${repo_root}/logs/triplet_jsonl/model_filtered_firsthit}
judge_out_base_dir=${JUDGE_OUT_BASE_DIR:-/mnt/jfs/logs/model_filtered_firsthit_by_model}
prepare_script_sh="${repo_root}/lora_pipeline/similarity_judge/prepare_model_filtered_firsthit_triplets.sh"
aggregate_py="${repo_root}/lora_pipeline/tools/aggregate_firsthit_perimage_to_model_jsonl.py"

mode=${MODE:-${1:-both}}
if [[ $# -gt 0 && "$1" =~ ^(content|style|both)$ ]]; then
    shift
fi
if [[ "$mode" != "content" && "$mode" != "style" && "$mode" != "both" ]]; then
    echo "invalid MODE: ${mode}. expected one of: content, style, both" >&2
    exit 1
fi

prepare_only=${PREPARE_ONLY:-0}
skip_prepare=${SKIP_PREPARE:-0}
include_empty=${INCLUDE_EMPTY:-0}

overwrite=${OVERWRITE:-0}
match_threshold=${MATCH_THRESHOLD:-2}
procs_per_endpoint=${PROCS_PER_ENDPOINT:-128}
conn_retry_times=${CONN_RETRY_TIMES:-2}
conn_retry_delay=${CONN_RETRY_DELAY:-1.0}
request_timeout_sec=${REQUEST_TIMEOUT_SEC:-180}
image_cache_size=${IMAGE_CACHE_SIZE:-32}
stats_interval_sec=${STATS_INTERVAL_SEC:-60}
flush_every=${FLUSH_EVERY:-32}
probe_timeout=${PROBE_TIMEOUT:-3.0}

content_conf_thr=${CONTENT_CONF_THR:-0.5}
content_judge_times=${CONTENT_JUDGE_TIMES:-3}
content_min_true=${CONTENT_MIN_TRUE:-2}

style_conf_thr=${STYLE_CONF_THR:-0.5}
style_judge_times=${STYLE_JUDGE_TIMES:-3}
style_min_true=${STYLE_MIN_TRUE:-2}

content_judge_py="${repo_root}/lora_pipeline/similarity_judge/triplet_qwen_content_firsthit_judge.py"
style_judge_py="${repo_root}/lora_pipeline/similarity_judge/triplet_qwen_style_firsthit_judge.py"

extra_args=()
if [[ "${overwrite}" == "1" ]]; then
    extra_args+=(--overwrite)
fi

# Active qwen tmux server windows checked on 2026-04-08.
ENDPOINTS=(
    "Qwen3-VL-30B-A3B-Instruct@http://10.204.18.79:22002/v1"
    "Qwen3-VL-30B-A3B-Instruct@http://10.204.24.79:22002/v1"
    "Qwen3-VL-30B-A3B-Instruct@http://10.204.18.85:22002/v1"
    "Qwen3-VL-30B-A3B-Instruct@http://10.204.18.79:22002/v1"
    "qwen3vlw8a8@http://stepcloud-apisix-gateway-eval.i-stepfun.com/Qwen3-VL-235B-A22B-W8A8/v1"
    "qwen35-27b@http://stepcast-router.shai-core:9200/v1"
)

endpoint_args=()
for endpoint in "${ENDPOINTS[@]}"; do
    endpoint_args+=(--endpoint "${endpoint}")
done

content_common_args=(
    --num-samples 0
    --content_conf_thr "${content_conf_thr}"
    --content_judge_times "${content_judge_times}"
    --content_min_true "${content_min_true}"
    --procs_per_endpoint "${procs_per_endpoint}"
    --conn_retry_times "${conn_retry_times}"
    --conn_retry_delay "${conn_retry_delay}"
    --request-timeout-sec "${request_timeout_sec}"
    --image-cache-size "${image_cache_size}"
    --stats-interval-sec "${stats_interval_sec}"
    --flush-every "${flush_every}"
    --match_threshold "${match_threshold}"
    --per-image
    "${endpoint_args[@]}"
    "${extra_args[@]}"
)

style_common_args=(
    --num-samples 0
    --style_conf_thr "${style_conf_thr}"
    --style_judge_times "${style_judge_times}"
    --style_min_true "${style_min_true}"
    --probe-timeout "${probe_timeout}"
    --procs_per_endpoint "${procs_per_endpoint}"
    --conn_retry_times "${conn_retry_times}"
    --conn_retry_delay "${conn_retry_delay}"
    --request-timeout-sec "${request_timeout_sec}"
    --image-cache-size "${image_cache_size}"
    --stats-interval-sec "${stats_interval_sec}"
    --flush-every "${flush_every}"
    --match_threshold "${match_threshold}"
    --per-image
    "${endpoint_args[@]}"
    "${extra_args[@]}"
)

prepare_all_if_needed() {
    if [[ "${skip_prepare}" == "1" ]]; then
        return
    fi
    PREPARED_TRIPLET_DIR="${prepared_dir}" PYTHON_BIN="${python_bin}" bash "${prepare_script_sh}"
}

aggregate_outputs() {
    local prepared_jsonl="$1"
    local raw_matched_jsonl="$2"
    local raw_all_similar_jsonl="$3"
    local raw_processed_jsonl="$4"
    local raw_error_jsonl="$5"
    local model_true_jsonl="$6"
    local model_fail_jsonl="$7"
    local model_error_jsonl="$8"

    agg_args=()
    if [[ "${include_empty}" == "1" ]]; then
        agg_args+=(--include-empty)
    fi

    "${python_bin}" "${aggregate_py}" \
        --prepared-triplet-jsonl "${prepared_jsonl}" \
        --matched-jsonl "${raw_matched_jsonl}" \
        --all-similar-jsonl "${raw_all_similar_jsonl}" \
        --processed-jsonl "${raw_processed_jsonl}" \
        --error-log-jsonl "${raw_error_jsonl}" \
        --out-true-jsonl "${model_true_jsonl}" \
        --out-fail-jsonl "${model_fail_jsonl}" \
        --out-error-jsonl "${model_error_jsonl}" \
        "${agg_args[@]}"
}

run_content_target() {
    local dataset="$1"
    local prepared_jsonl="${prepared_dir}/${dataset}_content_model_filtered_triplet.jsonl"
    local index_jsonl=""
    local out_dir="${judge_out_base_dir}/${dataset}_content_firsthit_by_model"

    case "${dataset}" in
        illustrious)
            index_jsonl="${repo_root}/logs/triplet_jsonl/illustrious_selections_original_paths.jsonl"
            ;;
        flux)
            index_jsonl="${repo_root}/logs/triplet_jsonl/selections_with_origin_content_flux.jsonl"
            ;;
        qwen)
            index_jsonl="${repo_root}/logs/triplet_jsonl/qwen_selections_original_paths.jsonl"
            ;;
        *)
            echo "unknown content dataset: ${dataset}" >&2
            exit 1
            ;;
    esac

    mkdir -p "${out_dir}"
    raw_matched_jsonl="${out_dir}/content_firsthit_matched_perimage.jsonl"
    raw_all_similar_jsonl="${out_dir}/content_firsthit_all_similar_perimage.jsonl"
    raw_error_jsonl="${out_dir}/content_firsthit_errors_perimage.jsonl"
    raw_processed_jsonl="${out_dir}/content_firsthit_processed_perimage.jsonl"
    model_true_jsonl="${out_dir}/content_firsthit_true_by_model.jsonl"
    model_fail_jsonl="${out_dir}/content_firsthit_fail_by_model.jsonl"
    model_error_jsonl="${out_dir}/content_firsthit_error_by_model.jsonl"

    if [[ "${prepare_only}" != "1" ]]; then
        echo "[content] dataset=${dataset} prepared_jsonl=${prepared_jsonl}"
        "${python_bin}" "${content_judge_py}" \
            --triplet-jsonl "${prepared_jsonl}" \
            --content-index-jsonl "${index_jsonl}" \
            --out-jsonl "${raw_matched_jsonl}" \
            --all-similar-out-jsonl "${raw_all_similar_jsonl}" \
            --error-log-jsonl "${raw_error_jsonl}" \
            --processed-jsonl "${raw_processed_jsonl}" \
            "${content_common_args[@]}"
        aggregate_outputs \
            "${prepared_jsonl}" \
            "${raw_matched_jsonl}" \
            "${raw_all_similar_jsonl}" \
            "${raw_processed_jsonl}" \
            "${raw_error_jsonl}" \
            "${model_true_jsonl}" \
            "${model_fail_jsonl}" \
            "${model_error_jsonl}"
    fi
}

run_style_target() {
    local dataset="$1"
    local prepared_jsonl="${prepared_dir}/${dataset}_style_model_filtered_triplet.jsonl"
    local index_jsonl=""
    local out_dir="${judge_out_base_dir}/${dataset}_style_firsthit_by_model"

    case "${dataset}" in
        illustrious)
            index_jsonl="${repo_root}/logs/triplet_jsonl/illustrious_selections_original_paths.jsonl"
            ;;
        flux)
            index_jsonl="${repo_root}/logs/triplet_jsonl/selections_with_origin_style_flux0325.jsonl"
            ;;
        qwen)
            index_jsonl="${repo_root}/logs/triplet_jsonl/qwen_selections_original_paths.jsonl"
            ;;
        *)
            echo "unknown style dataset: ${dataset}" >&2
            exit 1
            ;;
    esac

    mkdir -p "${out_dir}"
    raw_matched_jsonl="${out_dir}/style_firsthit_matched_perimage.jsonl"
    raw_all_similar_jsonl="${out_dir}/style_firsthit_all_similar_perimage.jsonl"
    raw_error_jsonl="${out_dir}/style_firsthit_errors_perimage.jsonl"
    raw_processed_jsonl="${out_dir}/style_firsthit_processed_perimage.jsonl"
    model_true_jsonl="${out_dir}/style_firsthit_true_by_model.jsonl"
    model_fail_jsonl="${out_dir}/style_firsthit_fail_by_model.jsonl"
    model_error_jsonl="${out_dir}/style_firsthit_error_by_model.jsonl"

    if [[ "${prepare_only}" != "1" ]]; then
        echo "[style] dataset=${dataset} prepared_jsonl=${prepared_jsonl}"
        "${python_bin}" "${style_judge_py}" \
            --triplet-jsonl "${prepared_jsonl}" \
            --style-index-jsonl "${index_jsonl}" \
            --out-jsonl "${raw_matched_jsonl}" \
            --all-similar-out-jsonl "${raw_all_similar_jsonl}" \
            --error-log-jsonl "${raw_error_jsonl}" \
            --processed-jsonl "${raw_processed_jsonl}" \
            "${style_common_args[@]}"
        aggregate_outputs \
            "${prepared_jsonl}" \
            "${raw_matched_jsonl}" \
            "${raw_all_similar_jsonl}" \
            "${raw_processed_jsonl}" \
            "${raw_error_jsonl}" \
            "${model_true_jsonl}" \
            "${model_fail_jsonl}" \
            "${model_error_jsonl}"
    fi
}

prepare_all_if_needed

IFS=',' read -r -a targets <<< "${targets_csv}"
for raw_target in "${targets[@]}"; do
    target="$(echo "${raw_target}" | xargs)"
    [[ -z "${target}" ]] && continue
    if [[ "${mode}" == "content" || "${mode}" == "both" ]]; then
        run_content_target "${target}"
    fi
    if [[ "${mode}" == "style" || "${mode}" == "both" ]]; then
        run_style_target "${target}"
    fi
done
