python /data/vgo/xingpeng/new_vgo/Sref_Cref_MiniVGO/qwen_2511_attention_fullmap_kfull.py \
  --model_name /mnt/jfs/model_zoo/qwen/Qwen-Image-Edit-2511/ \
  --transformer_ckpt /mnt/jfs/model_zoo/checkpoint-12000 \
  --prompts_json /mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/prompts.json \
  --cref_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/cref \
  --sref_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/sref \
  --key_txt /data/benchmark_metrics/insight/key_folder/analysis_key.txt \
  --out_dir /mnt/jfs/logs/ours_analysis_key \
  --gpus 0