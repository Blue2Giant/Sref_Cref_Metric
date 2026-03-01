sref_root=/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content
python /data/benchmark_metrics/sref_cref/qwen_infer.py \
    --prompts_json $sref_root/prompts.json \
    --cref_dir $sref_root/cref \
    --sref_dir $sref_root/sref \
    --out_dir $sref_root/qwen-edit \
    --model_name /mnt/jfs/model_zoo/qwen/Qwen-Image-Edit-2511/ \
    --gpus 0