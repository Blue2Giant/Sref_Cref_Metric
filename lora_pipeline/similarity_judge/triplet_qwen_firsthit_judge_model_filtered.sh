#!/usr/bin/env bash
set -euo pipefail

repo_root=/data/benchmark_metrics
python_bin=${PYTHON_BIN:-python3}
prepare_only=${PREPARE_ONLY:-0}
targets_csv=${TARGETS:-illustrious,flux,qwen}
prepared_dir=${PREPARED_TRIPLET_DIR:-${repo_root}/logs/triplet_jsonl/model_filtered_firsthit}
out_base_dir=${OUT_BASE_DIR:-/mnt/jfs/logs/model_filtered_firsthit}

mode=${MODE:-${1:-both}}
if [[ $# -gt 0 && "$1" =~ ^(content|style|both)$ ]]; then
    shift
fi
if [[ "$mode" != "content" && "$mode" != "style" && "$mode" != "both" ]]; then
    echo "invalid MODE: ${mode}. expected one of: content, style, both" >&2
    exit 1
fi

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

prepare_script="${repo_root}/lora_pipeline/tools/prepare_single_model_triplet_jsonl_for_firsthit.py"
content_judge_py="${repo_root}/lora_pipeline/similarity_judge/triplet_qwen_content_firsthit_judge.py"
style_judge_py="${repo_root}/lora_pipeline/similarity_judge/triplet_qwen_style_firsthit_judge.py"
model_id_dir="${repo_root}/lora_pipeline/meta/model_ids/models"

mkdir -p "${prepared_dir}" "${out_base_dir}"

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

prepare_triplet_jsonl() {
    local source_jsonl="$1"
    local model_id_txt="$2"
    local out_jsonl="$3"
    local missing_txt="$4"

    echo "[prepare] source_jsonl=${source_jsonl}"
    echo "[prepare] model_id_txt=${model_id_txt}"
    "${python_bin}" "${prepare_script}" \
        --source-jsonl "${source_jsonl}" \
        --model-id-txt "${model_id_txt}" \
        --out-jsonl "${out_jsonl}" \
        --missing-model-id-txt "${missing_txt}" \
        --progress-every 200
}

run_content_target() {
    local dataset="$1"
    local source_jsonl=""
    local model_id_txt=""
    local index_jsonl=""
    local prepared_jsonl=""
    local missing_txt=""
    local out_dir=""

    case "${dataset}" in
        illustrious)
            source_jsonl="${repo_root}/logs/triplet_jsonl/illustrious_0321_two_lora_and_illustrious_s3_eval_images_by_model.jsonl"
            model_id_txt="${model_id_dir}/illustrious_content.txt"
            index_jsonl="${repo_root}/logs/triplet_jsonl/illustrious_selections_original_paths.jsonl"
            ;;
        flux)
            source_jsonl="${repo_root}/logs/triplet_jsonl/flux_0321_0326_and_s3_eval_images_by_model.jsonl"
            model_id_txt="${model_id_dir}/flux_content.txt"
            index_jsonl="${repo_root}/logs/triplet_jsonl/selections_with_origin_content_flux.jsonl"
            ;;
        qwen)
            source_jsonl="${repo_root}/logs/triplet_jsonl/qwen_0323_one_lora_and_qwen_s3_eval_images_by_model.jsonl"
            model_id_txt="${model_id_dir}/qwen_content.txt"
            index_jsonl="${repo_root}/logs/triplet_jsonl/qwen_selections_original_paths.jsonl"
            ;;
        *)
            echo "unknown content dataset: ${dataset}" >&2
            exit 1
            ;;
    esac

    prepared_jsonl="${prepared_dir}/${dataset}_content_model_filtered_triplet.jsonl"
    missing_txt="${prepared_dir}/${dataset}_content_model_filtered_missing_model_ids.txt"
    out_dir="${out_base_dir}/${dataset}_content_firsthit_model_filtered"

    prepare_triplet_jsonl "${source_jsonl}" "${model_id_txt}" "${prepared_jsonl}" "${missing_txt}"
    if [[ "${prepare_only}" == "1" ]]; then
        return
    fi

    mkdir -p "${out_dir}"
    echo "[content] dataset=${dataset} prepared_jsonl=${prepared_jsonl} out_dir=${out_dir}"
    "${python_bin}" "${content_judge_py}" \
        --triplet-jsonl "${prepared_jsonl}" \
        --content-index-jsonl "${index_jsonl}" \
        --out-jsonl "${out_dir}/content_firsthit_matched.jsonl" \
        --all-similar-out-jsonl "${out_dir}/content_firsthit_all_similar.jsonl" \
        --error-log-jsonl "${out_dir}/content_firsthit_errors.jsonl" \
        --processed-jsonl "${out_dir}/content_firsthit_processed.jsonl" \
        "${content_common_args[@]}"
}

run_style_target() {
    local dataset="$1"
    local source_jsonl=""
    local model_id_txt=""
    local index_jsonl=""
    local prepared_jsonl=""
    local missing_txt=""
    local out_dir=""

    case "${dataset}" in
        illustrious)
            source_jsonl="${repo_root}/logs/triplet_jsonl/illustrious_0321_two_lora_and_illustrious_s3_eval_images_by_model.jsonl"
            model_id_txt="${model_id_dir}/illustrious_style.txt"
            index_jsonl="${repo_root}/logs/triplet_jsonl/illustrious_selections_original_paths.jsonl"
            ;;
        flux)
            source_jsonl="${repo_root}/logs/triplet_jsonl/flux_0321_0326_and_s3_eval_images_by_model.jsonl"
            model_id_txt="${model_id_dir}/flux_style.txt"
            index_jsonl="${repo_root}/logs/triplet_jsonl/selections_with_origin_style_flux0325.jsonl"
            ;;
        qwen)
            source_jsonl="${repo_root}/logs/triplet_jsonl/qwen_0323_one_lora_and_qwen_s3_eval_images_by_model.jsonl"
            model_id_txt="${model_id_dir}/qwen_style.txt"
            index_jsonl="${repo_root}/logs/triplet_jsonl/qwen_selections_original_paths.jsonl"
            ;;
        *)
            echo "unknown style dataset: ${dataset}" >&2
            exit 1
            ;;
    esac

    prepared_jsonl="${prepared_dir}/${dataset}_style_model_filtered_triplet.jsonl"
    missing_txt="${prepared_dir}/${dataset}_style_model_filtered_missing_model_ids.txt"
    out_dir="${out_base_dir}/${dataset}_style_firsthit_model_filtered"

    prepare_triplet_jsonl "${source_jsonl}" "${model_id_txt}" "${prepared_jsonl}" "${missing_txt}"
    if [[ "${prepare_only}" == "1" ]]; then
        return
    fi

    mkdir -p "${out_dir}"
    echo "[style] dataset=${dataset} prepared_jsonl=${prepared_jsonl} out_dir=${out_dir}"
    "${python_bin}" "${style_judge_py}" \
        --triplet-jsonl "${prepared_jsonl}" \
        --style-index-jsonl "${index_jsonl}" \
        --out-jsonl "${out_dir}/style_firsthit_matched.jsonl" \
        --all-similar-out-jsonl "${out_dir}/style_firsthit_all_similar.jsonl" \
        --error-log-jsonl "${out_dir}/style_firsthit_errors.jsonl" \
        --processed-jsonl "${out_dir}/style_firsthit_processed.jsonl" \
        "${style_common_args[@]}"
}

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
