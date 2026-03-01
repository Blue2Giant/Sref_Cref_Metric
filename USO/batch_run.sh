sref_root=/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content
python3 /data/benchmark_metrics/USO/batch_simple_demo.py \
  --input-dir $sref_root  \
  --prompts-json $sref_root/prompts.json \
  --out-dir $sref_root/uso \
  --instruct-edit \
  --sref-only \
  --use-siglip
#instruct-edit控制宽高是否和content图一样
#--sref-only 强制prompt为空