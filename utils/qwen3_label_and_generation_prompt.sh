#!/usr/bin/env bash
set -euo pipefail

########################################
# 1. 启动 Qwen3 vLLM 服务（后台）
########################################

MODEL_PATH="/mnt/jfs/model_zoo/Qwen3-VL-235B-A22B-Instruct"
SERVED_MODEL_NAME="Qwen3-VL-235B-A22B-Instruct"   # 暴露给 OpenAI API 的 model 名
HOST="0.0.0.0"
PORT=22002
BASE_URL="http://127.0.0.1:${PORT}/v1"                 # 等会打标脚本就用这个

LOG_DIR="/data/LoraPipeline/logs"
mkdir -p "${LOG_DIR}"

echo "[INFO] 启动 Qwen3 vLLM 服务..."
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size 4 \
  --mm-encoder-tp-mode data \
  --enable-expert-parallel \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.70 \
  --distributed-executor-backend mp \
  > "${LOG_DIR}/qwen3_vllm_server.log" 2>&1 &

SERVER_PID=$!
echo "[INFO] vLLM server PID=${SERVER_PID}, 日志: ${LOG_DIR}/qwen3_vllm_server.log"

########################################
# 2. 健康检查：等服务 ready
########################################

echo "[INFO] 等待 Qwen3 服务就绪: ${BASE_URL}"

MAX_RETRY=60   # 最多等 60 * 5 = 300 秒
SLEEP_SEC=5

for ((i=1; i<=MAX_RETRY; i++)); do
  # 先确认进程还活着
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[ERROR] vLLM 进程已退出，启动失败，查看日志：${LOG_DIR}/qwen3_vllm_server.log"
    exit 1
  fi

  # 尝试请求 /models 或 /v1/models
  if curl -sS "${BASE_URL}/models" > /dev/null 2>&1; then
    echo "[INFO] Qwen3 服务已就绪！"
    break
  fi

  echo "[INFO] 第 ${i}/${MAX_RETRY} 次探测未通过，${SLEEP_SEC} 秒后重试..."
  sleep "${SLEEP_SEC}"
done

# 超时保护
if ! curl -sS "${BASE_URL}/models" > /dev/null 2>&1; then
  echo "[ERROR] 等待超时，Qwen3 服务一直没就绪，退出。"
  kill "${SERVER_PID}" || true
  exit 1
fi

########################################
# 3. 启动打标脚本
########################################

echo "[INFO] 开始运行打标脚本..."

cd /data/LoraPipeline

civitai_loras="s3://collect-data-datasets/202510/civitai_file/"
shakker_loras="s3://collect-data-datasets/202510/shakker_file/"
liblib_loras="s3://collect-data-datasets/202510/liblib_file/"
tensorart_loras="s3://collect-data-datasets/202510/tensorart_file/"

civitai_bucket_qwen="s3://lanjinghong-data/civitai_label_binary_classfication_using_prompt_example_filtered_qwen/"
civitai_qwen_loras="s3://collect-data-datasets/202510/civitai_file/Qwen"
output_civitai_qwen="s3://lanjinghong-data/loras_eval_qwen"

civitai_bucket_flux="s3://lanjinghong-data/civitai_label_binary_classfication_using_prompt_example_filtered_flux/"
civitai_flux_loras="s3://collect-data-datasets/202510/civitai_file/Flux.1 D"
output_civitai_flux="s3://lanjinghong-data/loras_eval_flux"

# ⚠️ 注意：这里的 --model 必须等于上面 SERVED_MODEL_NAME，或者你改 vLLM 的 --served-model-name 来对齐
python /data/LoraPipeline/qwen_prompt_generation_lora.py \
  --meta-root "${civitai_bucket_flux}" \
  --output-root "${output_civitai_flux}" \
  --weight-root "${civitai_flux_loras}" \
  --max-tokens 32768 \
  --num-calls 1 \
  --num-prompts 100 \
  --model "${SERVED_MODEL_NAME}" \
  --base-url "${BASE_URL}"

LABEL_EXIT_CODE=$?

########################################
# 4. 打标结束后，停掉 vLLM（可选）
########################################

echo "[INFO] 打标脚本退出码: ${LABEL_EXIT_CODE}，准备关闭 Qwen3 vLLM 服务..."
kill "${SERVER_PID}" 2>/dev/null || true
echo "[INFO] 已发送 kill 给 vLLM 进程 (PID=${SERVER_PID})"

exit "${LABEL_EXIT_CODE}"
