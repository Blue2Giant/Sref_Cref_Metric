while true;do
    python /data/LoraPipeline/triplet_similarity/content_logits.py \
    --refs /data/LoraPipeline/data/similarity_stats_triplet/content_similarity_hist.png /data/LoraPipeline/data/similarity_stats_triplet/content_similarity_hist.png \
    --stylized /data/LoraPipeline/data/similarity_stats_triplet/content_similarity_hist.png \
    --base-url http://stepcast-router.shai-core:9200/v1 \
    --model Qwen3VL30BA3B-Image-Edit
done