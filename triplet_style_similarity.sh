export SIGLIP_PATH=/data/USO/weights/siglip
cd /data/LoraPipeline
# python /data/LoraPipeline/triplet_style_similarity.py \
#   --root s3://lanjinghong-data/loras_eval_qwen_two_lora_with_trigger/new_with_txt_show \
#   --pt_style siglip,styleshot,oneig \
#   --model_path /data/Sref_Cref/CSD/pretrainedmodels/vit-b-300ep.pth.tar \
#   --styleshot_clip_path /mnt/jfs/model_zoo/CLIP-ViT-H-14-laion2B-s32B-b79K \
#   --styleshot_weight_path /mnt/jfs/model_zoo/StyleShot/StyleShot/pretrained_weight/style_aware_encoder.bin \
#   --oneig_path /mnt/jfs/model_zoo/OneIG-StyleEncoder \
#   --encoder_weights 0.0,0.35,0.65 \
#   --output-name triplet_style_similarity.json \
#   --gpu-ids 0,1,2,3

python triplet_style_similarity.py \
    --root s3://lanjinghong-data/loras_eval_qwen_two_lora_with_trigger_new_with_txt \
    --model_path /data/Sref_Cref/CSD/pretrainedmodels/vit-b-300ep.pth.tar \
    --styleshot_clip_path /mnt/jfs/model_zoo/CLIP-ViT-H-14-laion2B-s32B-b79K \
    --styleshot_weight_path /mnt/jfs/model_zoo/StyleShot/StyleShot/pretrained_weight/style_aware_encoder.bin \
    --oneig_path /mnt/jfs/model_zoo/OneIG-StyleEncoder \
    --style-root s3://lanjinghong-data/loras_eval_qwen_filtered/ \
    --output-json s3://lanjinghong-data/loras_eval_qwen_two_lora_with_trigger_new_with_txt/loras_triplets_style_similarity.json \
    --id-list  /data/LoraPipeline/similarity_stats/qwen_style_ids.txt \
    --pt-style oneig,styleshot \
    --encoder-weights "oneig:0.35,styleshot:0.65" \
    --num-workers 4 \
    --gpu-ids 0,1,2,3