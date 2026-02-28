
# 📊 Vault 样本级标注工具

## 🎯 什么是 Sample Annotations？

**Sample Annotations（样本级标注）**是 Vault 数据库中每个样本独有的标注值，例如：
- **评分类**：美学评分 (7.2分)、CLIP相似度 (0.85)、质量分数 (0.92)
- **测量类**：图像清晰度、颜色饱和度、对比度指标
- **模型输出**：目标检测结果、OCR结果、分类置信度
- **人工评价**：质量打分、偏好排名、审核结果

### 🔄 与共享标注的区别

| 标注类型 | Sample Annotations | 共享 Annotations |
|---------|-------------------|-------------------|
| **数据存储** | 每个样本独立存储 | 多个样本共享同一标注对象 |
| **适合场景** | 评分、测量值、模型输出 | 标签、分类、数据来源 |
| **查询效率** | 支持数值范围查询 | 适合标签批量查询 |
| **典型示例** | `aesthetic_score: 7.2` | `dataset: coco` |


## 🚀 快速开始

### 1. 验证环境（30秒）
```bash
python test_sample_annotations.py quick_test
```

### 2. 添加美学评分（1分钟）
```bash
python add_sample_annotations_tutorial.py add_aesthetic_scores \
    --vault_path=/your/vault/path \
    --num_workers=8
```

### 3. 查询高评分图像（30秒）
```python
from vault.storage.lanceduck.multimodal import MultiModalStorager

storager = MultiModalStorager("/your/vault/path", read_only=True)
with storager.meta_handler as handler:
    results = handler.query_batch("""
        SELECT i.uri, sa.value_float as score
        FROM images i
        JOIN sample_annotation_elements sae ON i.id = sae.element_id
        JOIN sample_annotations sa ON sae.sample_annotation_id = sa.id
        WHERE sa.name = 'aesthetic_score' AND sa.value_float > 7.0
        ORDER BY sa.value_float DESC
        LIMIT 10
    """)
```

## 📋 数据结构说明

### Sample Annotations 表结构
- **`sample_annotations`**：标注定义表
  - `id`: 标注唯一标识
  - `name`: 标注名称（如 'aesthetic_score'）
  - `value_float`: 浮点数值（用于评分）
  - `value_json`: JSON数据（用于复杂结构如检测框）
  - `sequence_id`: 可选，关联到特定序列

- **`sample_annotation_elements`**：元素关联表
  - `element_id`: 指向 image 或 text 的ID
  - `element_type`: 'image' 或 'text'
  - `role`: 元素角色（如 'source_image', 'target_image', 'caption'）

## 🎚️ 五大使用场景详解

> 💡 **使用提示**：如果你是新手，建议先运行上面的快速开始，再根据需要查看详细场景。

### 场景1：单图像标注（美学评分）
为每张图像添加独立的美学评分。

```bash
python add_sample_annotations_tutorial.py add_aesthetic_scores \
    --vault_path=/path/to/vault \
    --source_filter=my_dataset  # 可选：只处理特定来源
```

**查询示例**：
```sql
-- 查找高质量图像（评分 > 7.0）
SELECT i.id, i.uri, sa.value_float as score
FROM images i
JOIN sample_annotation_elements sae ON i.id = sae.element_id
JOIN sample_annotations sa ON sae.sample_annotation_id = sa.id
WHERE sa.name = 'aesthetic_score' AND sa.value_float > 7.0
ORDER BY sa.value_float DESC
```

### 场景2：图文对标注（CLIP相似度）
为图文对添加匹配度评分。

```bash
python add_sample_annotations_tutorial.py add_clip_scores \
    --vault_path=/path/to/vault
```

**数据结构**：
```python
SampleAnnotation(
    name="clip_score",
    value=0.85,
    participants=(
        (image_id, IMAGE, "image"),
        (text_id, TEXT, "caption"),
    )
)
```

### 场景3：多元素标注（复杂JSON）
为包含多个元素的样本添加结构化标注。

```python
SampleAnnotation.create(
    name="edit_quality",
    value={  # 复杂 JSON，存入 value_json
        "overall_quality": 0.85,
        "instruction_following": 0.92,
        "visual_quality": 0.88,
    },
    participants=(
        (source_image_id, IMAGE, "source"),
        (instruction_id, TEXT, "instruction"),
        (target_image_id, IMAGE, "target"),
    )
)
```

### 场景4：从文件批量导入
从 CSV/Parquet/JSONL 文件批量导入标注结果。

```bash
# CSV 格式
python add_sample_annotations_tutorial.py import_from_file \
    --vault_path=/path/to/vault \
    --file_path=/path/to/scores.csv \
    --file_type=csv

# Parquet 格式（推荐大规模数据）
python add_sample_annotations_tutorial.py import_from_file \
    --vault_path=/path/to/vault \
    --file_path=/path/to/scores.parquet \
    --file_type=parquet
```

**CSV 文件格式**：
```csv
sequence_id,image_id,score
a1b2c3d4-...,e5f6g7h8-...,7.5
i9j0k1l2-...,m3n4o5p6-...,8.2
```

### 场景5：分布式处理（大规模数据）
多进程并行处理大规模数据集。

```bash
python add_sample_annotations_tutorial.py distributed \
    --vault_path=/path/to/vault \
    --num_workers=16  # 工作进程数
```

**优势**：
- 多进程并行处理，显著提升速度
- 每个进程独立的临时数据库，避免冲突
- 最后统一合并到主 Vault

## 📚 使用指南

### 🔍 获取样本级标注

#### 方法1：直接SQL查询
```python
# 查询所有美学评分
with storager.meta_handler as handler:
    scores = handler.query_batch("""
        SELECT i.id, i.uri, sa.value_float
        FROM images i
        JOIN sample_annotation_elements sae ON i.id = sae.element_id
        JOIN sample_annotations sa ON sae.sample_annotation_id = sa.id
        WHERE sa.name = 'aesthetic_score'
    """)

# 统计不同来源的平均评分
stats = handler.query_batch("""
    SELECT i.source, AVG(sa.value_float) as avg_score, COUNT(*) as count
    FROM images i
    JOIN sample_annotation_elements sae ON i.id = sae.element_id
    JOIN sample_annotations sa ON sae.sample_annotation_id = sa.id
    WHERE sa.name = 'aesthetic_score'
    GROUP BY i.source
""")
```

#### 方法2：使用工具集
```python
from annotation_utils import AnnotationAnalyzer

analyzer = AnnotationAnalyzer("/your/vault/path")
analyzer.print_annotation_summary("aesthetic_score")
```

### ➕ 添加样本级标注

#### 场景1：添加单个图像评分
```python
from annotation_utils import AnnotationPipeline
from vault.schema import ID

pipeline = AnnotationPipeline("/your/vault/path")

# 添加美学评分
pipeline.add_sample_annotation(
    name="aesthetic_score",
    element_id=ID.from_uuid("image-uuid"),
    element_type="image",
    value_float=7.8,
    creator_name="aesthetic_model"
)
```

#### 场景2：批量添加评分（推荐）
```bash
# 添加美学评分（支持多进程）
python add_sample_annotations_tutorial.py add_aesthetic_scores \
    --vault_path=/your/vault/path \
    --num_workers=16

# 添加CLIP相似度（图文对）
python add_sample_annotations_tutorial.py add_clip_scores \
    --vault_path=/your/vault/path \
    --num_workers=8
```

#### 场景3：从文件导入
```bash
# 从CSV导入
python add_sample_annotations_tutorial.py import_from_file \
    --vault_path=/your/vault/path \
    --file_type=csv \
    --input_file=/path/to/scores.csv \
    --annotation_name=custom_score

# 从JSONL导入
python add_sample_annotations_tutorial.py import_from_file \
    --vault_path=/your/vault/path \
    --file_type=jsonl \
    --input_file=/path/to/results.jsonl
```

## 🛠️ 可用工具

| 文件名 | 用途 | 适用场景 |
|--------|------|----------|
| `quickstart_5min.py` | 最简示例 | 快速验证、学习概念 |
| `add_sample_annotations_tutorial.py` | 完整示例 | 生产环境、多场景 |
| `annotation_utils.py` | 工具类库 | 工程集成、自定义开发 |
| `test_sample_annotations.py` | 自动测试 | 环境验证、功能测试 |

## 💡 常见使用模式

### 模式1：数据质量筛选
```python
# 筛选高质量图像（美学评分 > 7.5 AND 清晰度 > 0.8）
high_quality = handler.query_batch("""
    SELECT DISTINCT i.id, i.uri
    FROM images i
    WHERE i.id IN (
        SELECT sae.element_id
        FROM sample_annotation_elements sae
        JOIN sample_annotations sa ON sae.sample_annotation_id = sa.id
        WHERE sa.name = 'aesthetic_score' AND sa.value_float > 7.5
    ) AND i.id IN (
        SELECT sae.element_id
        FROM sample_annotation_elements sae
        JOIN sample_annotations sa ON sae.sample_annotation_id = sa.id
        WHERE sa.name = 'sharpness_score' AND sa.value_float > 0.8
    )
""")
```

### 模式2：模型性能分析
```python
# 分析CLIP分数分布
clip_stats = handler.query_batch("""
    SELECT
        CASE
            WHEN sa.value_float > 0.8 THEN 'High'
            WHEN sa.value_float > 0.6 THEN 'Medium'
            ELSE 'Low'
        END as score_range,
        COUNT(*) as count,
        AVG(sa.value_float) as avg_score
    FROM sample_annotations sa
    JOIN sample_annotation_elements sae ON sa.id = sae.sample_annotation_id
    WHERE sa.name = 'clip_score' AND sae.element_type = 'image'
    GROUP BY score_range
""")
```

### 模式3：多元素关系标注
```python
# 为图文对添加CLIP分数
pipeline.add_multi_element_annotation(
    name="clip_score",
    value_float=0.85,
    elements=[
        {"id": image_id, "type": "image", "role": "image"},
        {"id": text_id, "type": "text", "role": "caption"}
    ],
    sequence_id=sequence_id  # 可选，明确标注上下文
)
```


## ⚠️ 注意事项

1. **类型选择**：评分用 `value_float`，结构化数据用 `value_json`
2. **角色明确**：多元素标注时必须指定 `role` 字段
3. **效率优化**：批量操作时使用事务，大 dataset 时使用多进程
4. **索引利用**：常用查询字段已建索引，无需额外优化

## 📖 核心工作流程

所有场景都遵循相同的四步流程：

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: 查询 Vault 获取需要标注的数据                    │
│ ─────────────────────────────────────────────────────── │
│ storager = MultiModalStorager(vault_path, read_only=True)│
│ sequences = storager.meta_handler.query_batch(...)      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Step 2: 执行标注任务（模型推理/人工打分等）               │
│ ─────────────────────────────────────────────────────── │
│ for item in sequences:                                  │
│     score = model.predict(item)                         │
│     annotations.append(SampleAnnotation.create(...))    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Step 3: 写入临时 DuckDB 文件                            │
│ ─────────────────────────────────────────────────────── │
│ temp_handler = DuckDBHandler(...)                       │
│ storager.add_sample_annotations(                        │
│     annotations, duckdb_handler=temp_handler)           │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Step 4: 合并到主 Vault                                  │
│ ─────────────────────────────────────────────────────── │
│ storager.merge(duckdb_files=[temp_db_path])             │
└─────────────────────────────────────────────────────────┘
```

## ❓ 常见问题解答

### Q1: Annotation vs SampleAnnotation 如何选择？
**使用 Annotation（共享标注）**：
- 多个样本共享同一标签
- 分类标签、数据来源标记
- 如 `generated_by:gpt4o`、`dataset:coco`

**使用 SampleAnnotation（样本级标注）**：
- 每个样本有独特的数值（如美学评分、CLIP分数）
- 模型推理结果（每个样本不同）
- 需要支持数值范围查询

### Q2: 如何存储复杂的结构化数据？
将 Python dict/list 直接传给 `value` 参数，会自动序列化为 JSON：

```python
SampleAnnotation.create(
    value={
        "scores": [0.8, 0.9, 0.7],
        "labels": ["cat", "dog"],
        "metadata": {"model": "v1.0"}
    },
    ...
)
```

### Q3: participants 参数是什么？
`participants` 定义了这个标注涉及哪些元素以及它们的角色：

```python
participants=(
    (image_id, MultiModalType.IMAGE, "source"),   # 角色: source
    (text_id, MultiModalType.TEXT, "instruction"), # 角色: instruction
    (image_id2, MultiModalType.IMAGE, "target"),  # 角色: target
)
```

角色字段帮助区分同类型的多个元素。

### Q4: 如何避免重复标注？
在查询时排除已标注的数据：

```python
with storager.meta_handler as handler:
    sequences = handler.query_batch("""
        SELECT s.id, si.image_id
        FROM sequences s
        JOIN sequence_images si ON s.id = si.sequence_id
        WHERE NOT EXISTS (
            SELECT 1 FROM sample_annotations sa
            WHERE sa.sequence_id = s.id
              AND sa.name = 'my_score'
        )
    """)
```

### Q5: 标注值可以是什么类型？
- **数字** → `value_float`
- **其他类型**（dict/list/str）→ `value_json`（自动处理）

## 🎯 最佳实践

1. **使用有意义的标注名称**：
   - ✅ `aesthetic_score_v2`
   - ❌ `score1`

2. **为 Creator 添加元信息**：
   ```python
   creator = Creator.create(
       name="aesthetic_scorer_v2",
       meta={
           "model": "aesthetic-predictor-v2.5",
           "training_data": "LAION-Aesthetics",
           "date": "2024-01-15",
       }
   )
   ```

3. **批量处理以提高效率**：
   - 每次处理 1000-10000 个样本
   - 使用 `batch_size` 参数控制内存

4. **验证标注结果**：
   ```python
   # 写入后立即验证
   with storager.meta_handler as handler:
       count = handler.query_batch(
           "SELECT COUNT(*) as cnt FROM sample_annotations WHERE name = ?",
           ["my_score"]
       )[0]["cnt"]
       assert count == expected_count
   ```

5. **记录 sequence_id**：
   - 强烈建议始终填写 `sequence_id` 字段
   - 有助于后续查询和分析

## 💻 命令行工具总览

```bash
# 教程脚本提供的所有命令
python add_sample_annotations_tutorial.py --help

# 可用命令列表
add_aesthetic_scores    # 场景 1：美学评分
add_clip_scores        # 场景 2：CLIP相似度
import_from_file       # 场景 4：文件导入
distributed            # 场景 5：分布式处理
query                  # 查询统计
export                 # 导出 CSV

# 测试脚本提供的命令
python test_sample_annotations.py --help

# 可用命令列表
quick_test             # 一键测试所有功能
create_vault           # 只创建测试 Vault
test_annotations       # 只运行标注测试
```

---

**开始使用**：`python test_sample_annotations.py quick_test`
