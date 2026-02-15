xingpeng_ip=http://stepcloud-apisix-gateway-eval.i-stepfun.com/Qwen3-VL-235B-A22B-W8A8/v1
xingpeng_model=qwen3vlw8a8
python /data/benchmark_metrics/triplet_similarity/content_similarity.py \
  --img-a /data/benchmark_metrics/assets/content_5/kitchen.jpg \
  --img-b /data/benchmark_metrics/assets/content_5/person.jpg \
  --model $xingpeng_model \
  --base-url $xingpeng_ip