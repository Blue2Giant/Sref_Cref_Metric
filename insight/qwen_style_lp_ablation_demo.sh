#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PY="/data/benchmark_metrics/insight/qwen_2511_style_lp_guided_demo.py"
SUM_PY="/data/benchmark_metrics/insight/qwen_style_lp_metrics_summary.py"

PROMPTS_JSON="${PROMPTS_JSON:-/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/prompts.json}"
CREF_DIR="${CREF_DIR:-/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/cref}"
SREF_DIR="${SREF_DIR:-/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/sref}"
MODEL_NAME="${MODEL_NAME:-/mnt/jfs/model_zoo/qwen/Qwen-Image-Edit-2511/}"
KEY_TXT="${KEY_TXT:-/data/benchmark_metrics/insight/key.txt}"
OUT_ROOT="${OUT_ROOT:-/data/benchmark_metrics/logs/qwen-style-lp-ablation}"
GPUS="${GPUS:-0}"
SEED="${SEED:-42}"
STEPS="${STEPS:-28}"
TRUE_CFG="${TRUE_CFG:-4.0}"

COMMON_ARGS=(
  --prompts_json "${PROMPTS_JSON}"
  --cref_dir "${CREF_DIR}"
  --sref_dir "${SREF_DIR}"
  --model_name "${MODEL_NAME}"
  --gpus "${GPUS}"
  --key_txt "${KEY_TXT}"
  --steps "${STEPS}"
  --true-cfg-scale "${TRUE_CFG}"
  --seed "${SEED}"
  --lp-factor 4
  --max-sequence-length 256
  --attention-slicing max
  --device-map balanced
  --max-memory-gpu 70GiB,70GiB
  --max-memory-cpu 800GiB
  --enable-vae-slicing
  --enable-vae-tiling
  --offload-image-latents-to-cpu
  --offload-prompt-embeds-to-cpu
  --empty-cache-per-step 4
)

run_case() {
  local case_name="$1"
  shift
  local out_dir="${OUT_ROOT}/${case_name}"
  local metrics_jsonl="${out_dir}/metrics.jsonl"
  mkdir -p "${out_dir}"
  echo "[RUN] case=${case_name}"
  "${PYTHON_BIN}" "${RUN_PY}" \
    "${COMMON_ARGS[@]}" \
    --out_dir "${out_dir}" \
    --metrics_jsonl "${metrics_jsonl}" \
    "$@"
}

run_case "raw" \
  --experiment raw

run_case "suppress_lp_beta0.2" \
  --experiment suppress_lp \
  --beta-const 0.2

run_case "suppress_hp_alpha0.5" \
  --experiment suppress_hp \
  --alpha-hp 0.5

run_case "lp_restore_e035_b0002_05" \
  --experiment lp_restore \
  --beta-schedule piecewise \
  --early-ratio 0.35 \
  --beta-early 0.0 \
  --beta-mid 0.2 \
  --beta-late 0.5

run_case "lp_restore_aggressive_e050_b0002_10" \
  --experiment lp_restore \
  --beta-schedule piecewise \
  --early-ratio 0.5 \
  --beta-early 0.0 \
  --beta-mid 0.2 \
  --beta-late 1.0

echo "[SUMMARY] 生成rho汇总: ${OUT_ROOT}/rho_summary.csv"
"${PYTHON_BIN}" "${SUM_PY}" \
  --root_dir "${OUT_ROOT}" \
  --out_csv "${OUT_ROOT}/rho_summary.csv"

echo "[DONE] ablation outputs at ${OUT_ROOT}"
