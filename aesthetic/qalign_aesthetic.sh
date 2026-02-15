# export PYTHONPATH=/data/LoraPipeline/third_party/aesthetic-scorer:$PYTHONPATH
# cd /data/LoraPipeline/aesthetic
# python /data/LoraPipeline/aesthetic/aesthetic_scorer_9grid_aesthetic_rank.py \
#   --input-dir /data/LoraPipeline/output/flux_0111_triplets_subset_for_human_content \
#   --output-txt /data/LoraPipeline/output/aesthetic_scorer_rank.txt \
#   --output-json /data/LoraPipeline/output/aesthetic_scorer_rank.json
# CUDA_VISIBLE_DEVICES="0" python /data/LoraPipeline/aesthetic/qalign_9grid_aesthetic_rank.py \
#   --input-dir /data/LoraPipeline/output/flux_0111_triplets_subset_for_human_content \
#   --output-json /data/LoraPipeline/output/flux_9grid_rank_content_human.json \
#   --filter-txt /data/LoraPipeline/assets/flux_content_human.txt

CUDA_VISIBLE_DEVICES="0" python /data/LoraPipeline/aesthetic/qalign_9grid_aesthetic_rank.py \
  --input-dir /mnt/jfs/9grid/illustrious_9grid \
  --output-json /data/LoraPipeline/output/sdxl_9grid_rank_content_sample.json \
  --filter-txt /data/LoraPipeline/assets/illustrious_content_ids.txt