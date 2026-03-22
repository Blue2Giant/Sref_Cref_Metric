# Efficient inference with FP8 checkpoint
# Requires NVIDIA H100+ and CUDA 12+
vllm serve /mnt/jfs/model_zoo/Qwen3-VL-30B-A3B-Instruct/ \
  --tensor-parallel-size 4 \
  --enable-prefix-caching False \
  --async-scheduling \
  --host 0.0.0.0 \
  --port 22002 \
  --gpu-memory-utilization 0.8 \
  --served-model-name "Qwen3-VL-30B-A3B-Instruct" \
  --mm-processor-cache-gb 0
  # --enable-chunked-prefill \
  # --no-enforce-eager \
  # --served-model-name "Qwen3-VL-30B-A3B-Instruct"
#  --enable-prefix-caching --no-enforce-eager \
python3 /usr/bin/signal_proxy.py "vllm serve /mnt/jfs/model_zoo/Qwen3-VL-30B-A3B-Instruct --served-model-name Qwen3-VL-30B-A3B-Instruct --port 22002 --tensor-parallel-size 4  --gpu-memory-utilization 0.6"