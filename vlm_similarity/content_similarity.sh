xingpeng_ip=http://stepcloud-apisix-gateway-eval.i-stepfun.com/Qwen3-VL-235B-A22B-W8A8/v1
xingpeng_model=qwen3vlw8a8
python /data/benchmark_metrics/vlm_similarity/content_similarity.sh \
  --img-a /data/benchmark_metrics/assets/content.webp \
  --img-b /data/benchmark_metrics/assets/style.webp \
  --model $xingpeng_model \
  --base-url $xingpeng_ip