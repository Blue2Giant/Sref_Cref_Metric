sref_root=/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content
model=uso
#内容：
python /data/benchmark_metrics/concat2seeMetric.py \
  --folders $sref_root/cref $sref_root/$model \
  --jsons $sref_root/$model/dinov2_out.json $sref_root/$model/cas_out.json \
  --out_dir $sref_root/$model/vis_content \
  --long_side 512
echo $sref_root/$model/vis_content



#风格：
python /data/benchmark_metrics/concat2seeMetric.py \
  --folders $sref_root/sref $sref_root/$model \
  --jsons $sref_root/$model/csd_out.json  $sref_root/$model/oneig_out.json \
  --out_dir $sref_root/$model/vis_style \
  --long_side 512
echo $sref_root/$model/vis_style



#其他：
python /data/benchmark_metrics/concat2seeMetric.py \
  --folders $sref_root/$model \
  --jsons $sref_root/$model/clipcap_out.json $sref_root/$model/laion_scores.json $sref_root/$model/v25_scores.json \
  --out_dir $sref_root/$model/vis_other \
  --long_side 512 \
  --caption_json $sref_root/prompts.json
echo $sref_root/$model/vis_other


#指令遵循可视化：
python /data/benchmark_metrics/concat2seeMetric.py \
  --folders $sref_root/$model \
  --jsons $sref_root/$model/follow_scores.json \
  --out_dir $sref_root/$model/vis_follow \
  --long_side 512 \
  --caption_json $sref_root/prompts.json \
   $sref_root/$model/follow_scores.json \
   $sref_root/$model/follow_reasons.json
echo $sref_root/$model/vis_follow



#vlm风格相似度可视化：
python /data/benchmark_metrics/concat2seeMetric.py \
  --folders $sref_root/sref $sref_root/$model \
  --jsons $sref_root/$model/qwen_resize_output_style_descrete.json\
  --out_dir $sref_root/$model/vis_style_vlm \
  --long_side 512 \
  --caption_json $sref_root/$model/qwen_resize_output_style_reason_descrete.json
echo $sref_root/$model/vis_style_vlm


#vlm内容相似度可视化：
python /data/benchmark_metrics/concat2seeMetric.py \
  --folders $sref_root/cref  $sref_root/$model \
  --jsons  $sref_root/$model/qwen_resize_output_content_descrete.json \
  --out_dir  $sref_root/$model/vis_content_vlm \
  --long_side 512 \
  --caption_json  $sref_root/$model/qwen_resize_output_content_reason_descrete.json
echo $sref_root/$model/vis_content_vlm

#dual judege相似度可视化
python /data/benchmark_metrics/concat2seeMetric.py \
  --folders $sref_root/cref $sref_root/sref $sref_root/$model \
  --jsons "$sref_root/$model/qwen_reject_cref.json" "$sref_root/$model/qwen_reject_sref.json" \
  --out_dir $sref_root/$model/vis_dual_judge \
  --long_side 512
echo "$sref_root/$model/vis_dual_judge"