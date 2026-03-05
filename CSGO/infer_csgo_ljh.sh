cd /data/benchmark_metrics/CSGO
export PYTHONPATH=$PYTHONPATH:/data/benchmark_metrics/CSGO
# python /data/benchmark_metrics/CSGO/infer_csgo_ljh.py
sref_dir="/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/"
python /data/benchmark_metrics/CSGO/infer_csgo_ljh_batch.py \
  --cref_dir $sref_dir/cref \
  --sref_dir $sref_dir/sref \
  --out_dir $sref_dir/csgo \
  --skip_existing