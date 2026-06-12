# 0426 Lora Triplet 开源方案

## 结论

之前按“每个 triplet 单独存一份图片”的方案是错的。

原因不是实现细节，而是数据语义本身不对：

1. 一个 triplet 由三张图组成：
   - `cref_0`: content lora 生成图
   - `sref_0`: style lora 生成图
   - `target`: content + style dual-lora 生成图
2. 同一张 content 图、style 图、target 图会在很多 vault sequence 里重复出现。
3. 所以如果按 sequence / triplet 单独落盘，会把同一张图重复导出很多次，既浪费空间，也破坏数据关系。

正确方案应该是：

1. 图片去重后放进共享图片池。
2. 用 `triplets.csv` 表达“这一条 vault sequence 用了哪三张图”。
3. 另外用单独的图片级 CSV 保存每张图自身的生成 prompt。

## 适用范围

本方案只处理 0426 config 里非零权重 source 中的 lora triplet 部分：

- `cref_sref_flux_lora_part1`
- `cref_sref_illustrious_lora_part1`
- `cref_sref_qwen_lora_part1`

`oneig` 的 5 个 source 仍然是另一类数据，不应强行套进这个 normalized triplet 结构里。它们后续应单独导出。

## 输出结构

建议开源目录：

```text
<output-dir>/
  README.md
  summary.json
  triplets.csv
  content_images.csv
  style_images.csv
  target_images.csv
  _state/
    manifest.json
    triplets.jsonl
    content_images.jsonl
    style_images.jsonl
    target_images.jsonl
  images/
    content/
      flux/
      illustrious/
      qwen/
    style/
      flux/
      illustrious/
      qwen/
    target/
      flux/
      illustrious/
      qwen/
```

说明：

1. `images/content/...`、`images/style/...`、`images/target/...` 存的都是 vault 里真实训练图像字节，不是原始 one-lora / dual-lora 目录里的 PNG 直接拷贝。
2. 同一张 vault 图按像素哈希去重，只导出一次。
3. `triplets.csv` 一行对应一条 vault sequence。
4. `_state/` 是内部断点续跑状态，正式整理阶段必须保留；最终上传到 Hugging Face 时可按需要决定是否保留。

## CSV 设计

### 1. `triplets.csv`

每行描述一条 vault sequence 与三张图之间的关系。核心字段：

- `sequence_id`
- `source`
- `base_model`
- `pair_key`
- `content_model_id`
- `style_model_id`
- `content_image_path`
- `style_image_path`
- `target_image_path`
- `content_original_path`
- `style_original_path`
- `target_original_path`
- `content_generation_prompt`
- `style_generation_prompt`
- `target_generation_prompt`
- `vault_sample_instruction_en_123`
- `vault_primary_instruction_en_123`
- `vault_sample_instruction_cn_123`
- `vault_primary_instruction_cn_123`
- `vault_captions_scene_1`
- `vault_captions_scene_1_en`
- `vault_captions_scene_2`
- `vault_captions_scene_2_en`
- `vault_captions_scene_3`
- `vault_captions_scene_3_en`
- `vault_content_trigger_words`
- `vault_style_trigger_words`
- `vault_target_caption`
- `vault_texts_json`

这里的原则是：

1. triplet 级 CSV 主要记录“sequence 用了哪三张图”和“vault 里这条 sequence 的全部文本 prompt / caption 信息”。
2. 原图自身的生成 prompt 也会尽量写进去，但它依赖原始路径匹配是否成功。

### 2. `content_images.csv`

每行对应一张去重后的 content 图。核心字段：

- `exported_image_path`
- `base_model`
- `model_id`
- `pixel_sha256`
- `original_path`
- `prompt_index`
- `generation_prompt`
- `base_prompt`
- `trigger_word`
- `prompt_payload_path`
- `match_status`
- `prompt_status`
- `sequence_count`
- `sequence_ids_json`

### 3. `style_images.csv`

和 `content_images.csv` 结构一致，只是 role 是 style。

### 4. `target_images.csv`

每行对应一张去重后的 dual-lora target 图。核心字段：

- `exported_image_path`
- `base_model`
- `pair_key`
- `content_model_id`
- `style_model_id`
- `pixel_sha256`
- `original_path`
- `prompt_index`
- `generation_prompt`
- `base_prompt`
- `content_trigger`
- `style_trigger`
- `prompt_payload_path`
- `match_status`
- `prompt_status`
- `sequence_count`
- `sequence_ids_json`

## 原始 prompt 的恢复逻辑

### 1. one-lora content / style

如果匹配到原始图片路径，例如：

- `/mnt/jfs/.../<model_id>/eval_images_with_negative_new/00014_0.png`

优先在同目录读取：

- `selected_prompts_diverse.json`

如果不存在，再回退到：

- `selected_prompts.json`

并用图片文件名前缀索引恢复：

- `prompts[idx]`
- `base_prompts[idx]`
- `indices[idx]`

说明：

1. `illustrious` 的 one-lora S3 目录实测常见的是 `selected_prompts.json`，不是 `selected_prompts_diverse.json`。
2. 对于 `/mnt/jfs/loras_combine/illustrious_0321_two_lora/...` 这种本地路径，当前脚本已经支持在本地文件缺失时回退读取对应的 `s3://lanjinghong-data/loras_eval_illustrious_one_img_magic/...` 图和 prompt。

### 2. dual-lora target

如果匹配到原始 target 路径，例如：

- `/mnt/jfs/.../<pair_key>/eval_images_with_negative_new/00003_0.png`

则在 pair 根目录读取：

- `selected_prompts_final.json`

并恢复：

- `selected_prompts[idx]`
- `selected_base_prompts[idx]`
- `content_trigger`
- `style_trigger`

## vault 图回原始路径的匹配策略

### 不能做的事

不能再假设：

1. `triplet_jsonls` 里的那几张图就是 vault 里实际存进去的图。
2. vault 图和原始 PNG 字节完全一致。
3. `pair_key` 下只有一张 target。

真实检查结果已经证明这些假设不成立。

### 当前可行策略

使用保守的“视觉哈希精确匹配”：

1. 从 vault 取真实训练图像字节。
2. 计算 `(aHash, dHash)` 视觉键。
3. 在原始 one-lora / dual-lora 候选池中，对候选图也计算同样的视觉键。
4. 只有当某张候选图唯一命中同一个视觉键时，才认定匹配成功。

这一步故意保守，原因是：

1. 错配 prompt 比留空更危险。
2. `illustrious` 和少量 `qwen` 样本说明候选池并不总是完整可访问。

## 候选池来源

当前正式实现已经切到 `meta/triplet_jsonls` 这套“实际 triplet 使用到的原始图片子集”，不再扫更大的日志候选池。

这样做的原因是：

1. 这套子集更符合 triplet 的真实语义。
2. 候选规模明显更小，更适合正式导出。
3. 可以显著降低全量导出时的原图匹配成本。

### flux

- content pool:
  - `/data/benchmark_metrics/lora_pipeline/meta/triplet_jsonls/flux_content_one_lora.jsonl`
- style pool:
  - `/data/benchmark_metrics/lora_pipeline/meta/triplet_jsonls/flux_style_one_lora.jsonl`
- dual pool:
  - `/data/benchmark_metrics/lora_pipeline/meta/triplet_jsonls/flux_dual_lora_style_content_filtered.jsonl`

### qwen

- content pool:
  - `/data/benchmark_metrics/lora_pipeline/meta/triplet_jsonls/qwen_content_one_lora.jsonl`
- style pool:
  - `/data/benchmark_metrics/lora_pipeline/meta/triplet_jsonls/qwen_style_one_lora.jsonl`
- dual pool:
  - `/data/benchmark_metrics/lora_pipeline/meta/triplet_jsonls/qwen_dual_lora_style_content_filtered.jsonl`

### illustrious

- content pool:
  - `/data/benchmark_metrics/lora_pipeline/meta/triplet_jsonls/illustrious_content_one_lora.jsonl`
- style pool:
  - `/data/benchmark_metrics/lora_pipeline/meta/triplet_jsonls/illustrious_style_one_lora.jsonl`
- dual pool:
  - `/data/benchmark_metrics/lora_pipeline/meta/triplet_jsonls/illustrious_dual_lora_style_content_filtered.jsonl`

## 当前实现

已新增专用脚本：

- `/data/vgo/xingpeng/new_vgo/Sref_Cref_MiniVGO/tools/export_lora_triplets_normalized.py`

脚本行为：

1. 读取 0426 data config。
2. 默认只处理其中非零权重的 lora source。
3. 直接从 vault `train.db` 读 sequence。
4. 从 vault 中导出真实训练图片。
5. 将图片去重写入 `images/content|style|target/...`。
6. 写出：
   - `triplets.csv`
   - `content_images.csv`
   - `style_images.csv`
   - `target_images.csv`
   - `summary.json`
   - `README.md`
7. 原始路径和原始 prompt 采用保守匹配：
   - `match_status=matched`: 唯一视觉键命中
   - `match_status=unmatched`: 候选池存在，但没有唯一精确命中
   - `match_status=no_candidates`: 本地没有可访问候选图
8. 同时维护 `_state/*.jsonl` 和 `_state/manifest.json`，用于机器中断后的断点续跑。

## 断点续跑策略

当前正式导出不再依赖“任务一次性完整跑完”。

恢复策略如下：

1. 运行时会把 sequence 级状态追加写入 `_state/triplets.jsonl`。
2. 每张唯一图片的元信息会分别追加写入：
   - `_state/content_images.jsonl`
   - `_state/style_images.jsonl`
   - `_state/target_images.jsonl`
3. 每处理一批 sequence，会刷新状态文件并更新 `_state/manifest.json`。
4. 如果机器中途中断，再次运行同一个输出目录并使用 `--resume`，脚本会读取 `_state`，跳过已经完成的 sequence，只继续处理剩余部分。
5. 全部处理完成后，脚本会基于 `_state` 重新生成正式的：
   - `triplets.csv`
   - `content_images.csv`
   - `style_images.csv`
   - `target_images.csv`
   - `summary.json`

已新增自动判断启动模式的包装脚本：

- `/data/vgo/xingpeng/new_vgo/Sref_Cref_MiniVGO/tools/run_lora_triplet_export.sh`

用法示例：

```bash
bash /data/vgo/xingpeng/new_vgo/Sref_Cref_MiniVGO/tools/run_lora_triplet_export.sh \
  cref_sref_qwen_lora_part1 \
  /mnt/jfs/vgo_hf_exports/0426_lora_triplet_normalized/qwen
```

这个脚本会自动判断：

1. 如果目标目录里还没有 `_state/manifest.json`，则首次启动并使用 `--overwrite`。
2. 如果 `_state/manifest.json` 已存在，则自动切换到 `--resume` 续跑。

## 已做的小测试结论

我在 `cpu_vgo-12` 上对真实 vault row 做了小规模验证。

### flux

测试 pair：

- `1041877__1001511`

结果：

- 4 条真实 sequence
- 12 张图全部唯一命中原始路径
- `target / cref_0 / sref_0` 都能恢复原始 prompt

覆盖率：

- `12 / 12`

### qwen

测试 pair：

- `1594223__1134477`

结果：

- 前 5 条真实 sequence
- 15 张图里 13 张唯一命中
- 少量 reference / target 仍未命中

覆盖率：

- `13 / 15`

### illustrious

测试 pair：

- `1035515__1044706`

结果：

- 5 条真实 sequence
- content 命中 `5 / 5`
- style 命中 `3 / 5`
- target 命中 `5 / 5`
- 两张 style 仍未命中
- post-patch 后，3 张 matched style 图里有 2 张 prompt 已恢复为 `resolved`
- 剩余 1 张 matched style 图是 `missing_prompt_entry`
- 其原因是 `selected_prompts.json` 只有 9 条 prompt，且不存在 `00012_0.json` 这样的单图 sidecar prompt 文件
- S3 one-lora prompt 可从 `selected_prompts.json` 恢复

覆盖率：

- `13 / 15`

## 为什么 `illustrious` 会差

根因主要是原始候选池不完整可访问，而不是 triplet 归档结构本身有问题：

1. `illustrious` one-lora 候选池由两部分组成：
   - 20 张 `/mnt/jfs/loras_combine/...` 本地图
   - 9 张 `s3://lanjinghong-data/loras_eval_illustrious_one_img_magic/...` 图
2. 现在 S3 credentials 已补上，所以这 9 张 S3 图可访问，且 prompt 也能恢复。
3. 但当前机器仍然访问不到 `/mnt/jfs/loras_combine/...` 那 20 张本地图。
4. 对 `1035515` 这组 smoke case 来说，剩余 2 张未命中的 style 图在现有可访问的 9 张 S3 候选里没有视觉键命中，因此不能安全补全。

因此：

1. `target` 这部分当前 smoke case 已经能恢复。
2. `content / style` 已经明显改善，但仍不是 100%。
3. `illustrious` 现在是“部分残留问题”，不是“整体对不上”。

结论是：

如果你要求 `illustrious` 这部分也高覆盖恢复“每张原图自己的生成 prompt”和原始路径，需要补一项前置条件：

- 给当前执行环境提供 `/mnt/jfs/loras_combine/...` 的可访问镜像，或者提供能覆盖这部分图片的额外候选池。

## 正式导出前的建议

### 如果当前目标是先把方案定下来并做最小可用验证

可以先：

1. 用新脚本导出 `flux` 和 `qwen` 的小样本。
2. 检查输出结构和 CSV 格式。
3. 再决定是否补 `illustrious` 的剩余元信息缺口。

### 如果当前目标是正式全量开源

现在的执行策略可以直接分 source 正式整理，输出先落在 `/mnt/jfs` 下，后续再同步到 Hugging Face。

需要接受的一点是：

1. `illustrious` 仍可能残留一部分 `unmatched_original_path`。
2. 少量 matched 图片仍可能出现 `missing_prompt_entry` 这类真实元信息缺口。

## 建议命令

### 小测试

```bash
python /data/vgo/xingpeng/new_vgo/Sref_Cref_MiniVGO/tools/export_lora_triplets_normalized.py \
  --config /data/vgo/xingpeng/new_vgo/Sref_Cref_MiniVGO/configs/data/0426_cref_sref_full_diffusion.yaml \
  --vault-root /mnt/chengwei/vault/traindata/04-26 \
  --output-dir /data/benchmark_metrics/.tmp/lora_triplet_normalized_smoke \
  --source cref_sref_flux_lora_part1 \
  --pair-key 1041877__1001511 \
  --limit 4 \
  --progress-every 1 \
  --overwrite
```

### 按 source 分开正式导出

```bash
bash /data/vgo/xingpeng/new_vgo/Sref_Cref_MiniVGO/tools/run_lora_triplet_export.sh \
  cref_sref_flux_lora_part1 \
  /mnt/jfs/vgo_hf_exports/0426_lora_triplet_normalized/flux
```

```bash
bash /data/vgo/xingpeng/new_vgo/Sref_Cref_MiniVGO/tools/run_lora_triplet_export.sh \
  cref_sref_qwen_lora_part1 \
  /mnt/jfs/vgo_hf_exports/0426_lora_triplet_normalized/qwen
```

```bash
bash /data/vgo/xingpeng/new_vgo/Sref_Cref_MiniVGO/tools/run_lora_triplet_export.sh \
  cref_sref_illustrious_lora_part1 \
  /mnt/jfs/vgo_hf_exports/0426_lora_triplet_normalized/illustrious
```

## 最终建议

正式上传前，最值得先确认的不是脚本，而是数据权限边界：

1. 你是否接受 `illustrious` 部分出现 `unmatched_original_path`。
2. 如果不接受，就需要先补 S3 credentials 或本地镜像。
3. `flux` 和 `qwen` 可以先独立推进，不必被 `illustrious` 卡死。
