
while true;do
    python /data/LoraPipeline/sref_pipeline/qwen3_rejudge_direct_reject.py \
    --image_txt /mnt/jfs/xhs_style_dir/image_paths.txt \
    --json_dir  /mnt/jfs/style_check_results_labeled/not_confuse \
    --output_dir /mnt/jfs/style_check_results_labeled_model_new_direct_reject \
    --model Qwen3-VL-30B-A3B-Instruct \
    --base_url http://10.191.2.11:22002/v1 \
    --overwrite
done

while true;do
    python /data/LoraPipeline/sref_pipeline/qwen3_style_rejudge_pairs.py \
    --image_txt /mnt/jfs/xhs_style_dir/image_paths.txt \
    --json_dir  /mnt/jfs/style_check_results_labeled/not_confuse \
    --output_dir /mnt/jfs/style_check_results_labeled_model_new \
    --model Qwen3VL30BA3B-Image-Edit \
    --base_url http://stepcast-router.shai-core:9200/v1
    --overwrite
done
    # --model Qwen3-VL-30B-A3B-Instruct \
    # --base_url http://10.191.22.49:22002/v1
    #   --model Qwen3VL30BA3B-Image-Edit \
    #   --base_url http://stepcast-router.shai-core:9200/v1
# direct copy
while true;do
    python /data/LoraPipeline/sref_pipeline/qwen3_style_rejudge_pairs_direct.py \
        --image_txt /mnt/jfs/xhs_style_dir/image_paths.txt \
        --json_dir  /mnt/jfs/style_check_results_labeled/not_confuse \
        --output_dir /mnt/jfs/style_check_results_labeled_model_direct \
        --model Qwen3VL30BA3B-Image-Edit \
        --base_url http://stepcast-router.shai-core:9200/v1 \
        --overwrite
done