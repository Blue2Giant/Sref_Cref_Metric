python /data/LoraPipeline/scripts/generate_prompts_from_csv.py \
  --input_csv /data/benchmark_metrics/lora_pipeline/meta/CHARACTER_UNIVERSE_TRIGGER.csv \
  --output_txt /data/benchmark_metrics/lora_pipeline/meta/CHARACTER_UNIVERSE_TRIGGER.txt \
  --num_prompts 200000 \
  --min_columns 2 \
  --max_columns 5 \
  --min_terms_per_column 0 \
  --max_terms_per_column 1 \
  --has_header \
  --seed 42 \
  --replace-space-with-underscore

python /data/LoraPipeline/scripts/generate_prompts_from_csv.py \
  --input_csv /data/benchmark_metrics/lora_pipeline/meta/OTHER_UNIVERSE_TRIGGER.csv \
  --output_txt /data/benchmark_metrics/lora_pipeline/meta/OTHER_UNIVERSE_TRIGGER.txt \
  --num_prompts 200000 \
  --min_columns 2 \
  --max_columns 5 \
  --min_terms_per_column 0 \
  --max_terms_per_column 1 \
  --has_header \
  --seed 42 \
    --replace-space-with-underscore

python /data/LoraPipeline/scripts/generate_prompts_from_csv.py \
  --input_csv /data/benchmark_metrics/lora_pipeline/meta/STYLE_UNIVERSE_TRIGGER.csv \
  --output_txt /data/benchmark_metrics/lora_pipeline/meta/STYLE_UNIVERSE_TRIGGER.txt \
  --num_prompts 200000 \
  --min_columns 2 \
  --max_columns 5 \
  --min_terms_per_column 0 \
  --max_terms_per_column 1 \
  --has_header \
  --seed 42 \
    --replace-space-with-underscore
