data_root=/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content
model=uso
python /data/benchmark_metrics/batch_temp/json_means_to_csv.py \
  --jsons \
  $data_root/$model/dinov2_out.json \
  $data_root/$model/cas_out.json \
  $data_root/$model/oneig_out.json \
  $data_root/$model/clipcap_out.json \
  $data_root/$model/csd_out.json \
  $data_root/$model/laion_scores.json \
  $data_root/$model/v25_scores.json \
  $data_root/$model/qwen_resize_output_style_descrete.json \
  $data_root/$model/qwen_resize_output_content_descrete.json \
  $data_root/$model/follow_scores.json \
  $data_root/$model/qwen_reject_cref.json \
  $data_root/$model/qwen_reject_sref.json \
  --out_csv $data_root/$model/metrics_mean.csv
echo "$data_root/$model/metrics_mean.csv"