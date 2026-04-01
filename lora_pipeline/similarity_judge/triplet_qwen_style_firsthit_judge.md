# triplet_qwen_style_firsthit_judge.py 参数说明

## 新增参数

- `--match_threshold`：整数，默认 `1`
  - 含义：至少命中多少张 style 参考图才判定该 pair 通过
  - 取值范围：`>= 0`
  - 结果规则：
    - 命中数 `>= match_threshold`：输出 value 为路径列表，长度为 `match_threshold`
    - 命中数 `< match_threshold`：输出 value 为 `[]`

## 示例

```bash
python /data/benchmark_metrics/lora_pipeline/tools/triplet_qwen_style_firsthit_judge.py \
  --triplet-jsonl /data/benchmark_metrics/logs/triplets_style_and_content_only.jsonl \
  --style-index-jsonl /data/benchmark_metrics/logs/selections_with_origin_style_flux0325.jsonl \
  --out-jsonl /data/benchmark_metrics/logs/triplet_style_firsthit_judge_0325/style_firsthit.jsonl \
  --error-log-jsonl /data/benchmark_metrics/logs/triplet_style_firsthit_judge_0325/style_firsthit_errors.jsonl \
  --style_conf_thr 0.5 \
  --style_judge_times 3 \
  --style_min_true 2 \
  --match_threshold 2 \
  --endpoint "Qwen3-VL-30B-A3B-Instruct@http://10.201.17.67:22002/v1"
```
