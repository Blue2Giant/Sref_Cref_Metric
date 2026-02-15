own_ip=http://10.204.2.11:22002/v1
own_model=Qwen3-VL-30B-A3B-Instruct
xingpeng_ip=http://stepcast-router.shai-core:9200/v1
xingpeng_model=Qwen3VL30BA3B-Image-Edit
while true; do
  python /data/LoraPipeline/triplet_similarity/vlm_content_similarity_batches.py \
    --root s3://lanjinghong-data/loras_eval_flux \
    --probe-mode demo \
    --base-url $xingpeng_ip \
    --model $xingpeng_model \
    --num-workers 4 \
    --content-dir-name eval_images_with_negative \
    --output-name vlm_content_similarity_with_negative.json \
    --overwrite
done