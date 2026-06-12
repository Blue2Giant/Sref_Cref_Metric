# Vault 数据结构总结

生成时间: 2026-04-14

本文档基于以下两部分信息整理:

- 代码定义: `vault/src/vault/schema/multimodal.py`、`vault/src/vault/storage/lanceduck/multimodal.py`
- 实际数据运行结果: `/mnt/marmot/chengwei/vault/cref_sref_oneig_filter_part1`

目标是总结这份 vault 数据里每个 `index` 对应的字段、角色和它们之间的关系。

## 1. 总体结论

这份 vault 不是扁平的单图单文数据，而是以 `sequence` 为核心组织的多模态样本库。

每条 `sequence` 通常由两部分组成:

- 3 张带角色索引的图片
- 1 条或 11 条带角色索引的文本，极少数为 10 条或 13 条

图片和文本都通过关系表挂在 `sequence` 上，真正描述“这个元素在样本里扮演什么角色”的，不是表名，而是关系表中的 `index` 字段。

## 2. 实际数据规模

对 `/mnt/marmot/chengwei/vault/cref_sref_oneig_filter_part1` 实测得到:

| 对象 | 数量 |
| --- | ---: |
| sequences | 106075 |
| images | 313622 |
| texts | 952528 |
| sequence_images | 318225 |
| sequence_texts | 952528 |

由此可以看出:

- `sequence_images = 106075 * 3`，说明每条 sequence 固定挂 3 张图片
- `sequence_texts` 数量不固定，说明文本侧有模板差异

## 3. 物理存储和逻辑关系

### 3.1 DuckDB 负责什么

DuckDB 里保存的是结构化元信息和关系:

- `sequences`
- `images`
- `texts`
- `sequence_images`
- `sequence_texts`
- `annotations`
- `image_annotations`
- `text_annotations`
- `sample_annotations`
- `sample_annotation_elements`

和本次分析最相关的是前 5 张表。

### 3.2 Lance 负责什么

Lance 里保存的是大对象和高维特征:

- `images`: 原始图片 bytes 和图像特征
- `annotations`: 带 blob 的 annotation

### 3.3 主关系

核心关系可以理解为:

```text
sequences
  ├─< sequence_images >─ images
  └─< sequence_texts  >─ texts
```

也就是说:

- `sequence_images.index` 定义图片在该样本中的角色
- `sequence_texts.index` 定义文本在该样本中的角色

## 4. `get_sequence_metas()` 的返回结构

`sample_pics.py` 调用的是:

```python
sequence_meta = storager.get_sequence_metas([seq_id])
```

单条 sequence 的返回结构实测为:

```python
{
    "sequence_id": ID(...),
    "images": [
        {
            "id": ID(...),
            "uri": "...",
            "source": "...",
            "width": ...,
            "height": ...,
            "index": "..."
        },
        ...
    ],
    "texts": [
        {
            "id": ID(...),
            "content": "...",
            "uri": "...",
            "source": "...",
            "language": ...,
            "index": "..."
        },
        ...
    ]
}
```

注意:

- `get_sequence_metas()` 当前不会把 `sequences.meta` 带出来
- `sample_pics.py` 里先单独查 `SELECT id, meta FROM sequences`，就是为了补这部分 sequence 级元信息
- 因此 sequence 级字段和元素级字段是分开放的

## 5. sequence 级字段

`sequences` 表中的字段为:

| 字段 | 含义 |
| --- | --- |
| `id` | sequence 主键 |
| `uri` | sequence 的资源路径 |
| `source` | sequence 来源 |
| `meta` | sequence 级 JSON 元信息 |

对本次抽样运行结果，`meta` 为 `None`。

## 6. 图片侧 index 与字段关系

### 6.1 图片公共字段

在 `get_sequence_metas()` 的结果里，每个图片元素都有相同字段:

| 字段 | 含义 |
| --- | --- |
| `id` | 图片 ID |
| `uri` | 图片 URI |
| `source` | 图片来源 |
| `width` | 图片宽度 |
| `height` | 图片高度 |
| `index` | 图片在 sequence 中的角色 |

### 6.2 图片在 Lance 中还能读到的额外字段

如果继续走 `get_image_bytes_by_ids()` 或直接读 Lance `images` 表，还能拿到:

| 字段 | 含义 |
| --- | --- |
| `image` | 原始图片 bytes |
| `file_hash` | 文件 hash |
| `file_size` | 文件大小 |
| `aspect_ratio` | 宽高比 |
| `color_mode` | 图像模式 |
| `mean_saturation` | 平均饱和度 |
| `mean_lightness` | 平均亮度 |
| `clarity` | 清晰度特征 |
| `entropy` | 熵 |
| `edge_probability` | 边缘概率 |
| `edge_near_patch_min_std` | 边缘附近 patch 统计 |
| `pdq_hash` | 感知 hash |
| `pdq_quality` | PDQ 质量分数 |

### 6.3 实际出现的图片 index

实测 `sequence_images.index` 只有 3 个值:

| image index | 出现次数 | 每个元素字段 | 角色总结 |
| --- | ---: | --- | --- |
| `sref_0` | 106075 | `id, uri, source, width, height, index` | 风格参考图 |
| `cref_0` | 106075 | `id, uri, source, width, height, index` | 内容参考图 |
| `target_image` | 106075 | `id, uri, source, width, height, index` | 目标结果图 |

这里“风格参考图 / 内容参考图 / 目标结果图”是根据实际 caption 和 instruction 语义推断出来的角色名，不是 schema 层面显式写死的约束。

## 7. 文本侧 index 与字段关系

### 7.1 文本公共字段

在 `get_sequence_metas()` 的结果里，每个文本元素都有相同字段:

| 字段 | 含义 |
| --- | --- |
| `id` | 文本 ID |
| `content` | 文本内容 |
| `uri` | 文本 URI |
| `source` | 文本来源 |
| `language` | 语言字段 |
| `index` | 文本在 sequence 中的角色 |

### 7.2 实际出现的文本 index

实测 `sequence_texts.index` 的分布如下:

| text index | 出现次数 | 每个元素字段 | 角色总结 |
| --- | ---: | --- | --- |
| `style_caption` | 106075 | `id, content, uri, source, language, index` | 样本的风格标签或风格摘要 |
| `captions/scene_1` | 84645 | 同上 | `scene_1` 的中文 caption |
| `captions/scene_1_en` | 84645 | 同上 | `scene_1` 的英文 caption |
| `captions/scene_2` | 84645 | 同上 | `scene_2` 的中文 caption |
| `captions/scene_2_en` | 84645 | 同上 | `scene_2` 的英文 caption |
| `captions/scene_3` | 84645 | 同上 | `scene_3` 的中文 caption |
| `captions/scene_3_en` | 84644 | 同上 | `scene_3` 的英文 caption |
| `sample_instruction_cn_123` | 84645 | 同上 | 简短中文指令 |
| `sample_instruction_en_123` | 84645 | 同上 | 简短英文指令 |
| `primary_instruction_cn_123` | 84645 | 同上 | 详细中文指令 |
| `primary_instruction_en_123` | 84645 | 同上 | 详细英文指令 |
| `captions/scene_4` | 2 | 同上 | 极少数扩展样本的额外 caption |
| `captions/scene_4_en` | 2 | 同上 | 极少数扩展样本的额外英文 caption |

### 7.3 文本 index 的语义关系

从一条实际样本的文本内容可以推断出:

- `scene_1` 描述内容主体
- `scene_2` 描述风格参考
- `scene_3` 描述目标生成结果
- `sample_instruction_*` 是简短版本的“把 scene_2 的风格迁移到 scene_1”
- `primary_instruction_*` 是更详细、更完整的版本

这与图片侧的常见角色大致对应为:

| 文本角色 | 可能对应的图片角色 | 说明 |
| --- | --- | --- |
| `captions/scene_1*` | `cref_0` | 内容参考图的描述 |
| `captions/scene_2*` | `sref_0` | 风格参考图的描述 |
| `captions/scene_3*` | `target_image` | 目标图的描述 |
| `sample_instruction_*` | `cref_0 + sref_0 -> target_image` | 简短任务指令 |
| `primary_instruction_*` | `cref_0 + sref_0 -> target_image` | 详细任务指令 |
| `style_caption` | 整条 sequence | 风格总结，不一定只绑定某一张图 |

这里的对应关系同样是语义推断，不是数据库里显式的外键字段。数据库只保证这些文本属于同一条 sequence，并拥有对应的 `index` 名称。

## 8. 常见 sequence 模板

实测每条 sequence 的形状分布为:

| 图片数 | 文本数 | sequence 数量 | 说明 |
| ---: | ---: | ---: | --- |
| 3 | 11 | 84642 | 标准完整模板 |
| 3 | 1 | 21430 | 仅保留 `style_caption` 的精简模板 |
| 3 | 13 | 2 | 扩展模板，额外带 `scene_4` 与 `scene_4_en` |
| 3 | 10 | 1 | 接近完整模板，但缺少 1 个文本字段 |

因此可以把这份数据理解成 3 种主要模板:

### 8.1 完整模板

- 图片: `sref_0`, `cref_0`, `target_image`
- 文本:
  - `style_caption`
  - `captions/scene_1`
  - `captions/scene_1_en`
  - `captions/scene_2`
  - `captions/scene_2_en`
  - `captions/scene_3`
  - `captions/scene_3_en`
  - `sample_instruction_cn_123`
  - `sample_instruction_en_123`
  - `primary_instruction_cn_123`
  - `primary_instruction_en_123`

### 8.2 精简模板

- 图片: `sref_0`, `cref_0`, `target_image`
- 文本:
  - `style_caption`

### 8.3 扩展模板

完整模板基础上额外增加:

- `captions/scene_4`
- `captions/scene_4_en`

## 9. 一条标准 sequence 的推荐理解方式

可以把一条常见样本理解为:

```text
sequence
├── 图片
│   ├── cref_0        -> 内容参考图
│   ├── sref_0        -> 风格参考图
│   └── target_image  -> 目标结果图
└── 文本
    ├── captions/scene_1*          -> 对内容参考图的描述
    ├── captions/scene_2*          -> 对风格参考图的描述
    ├── captions/scene_3*          -> 对目标结果图的描述
    ├── sample_instruction_*       -> 简短的风格迁移指令
    ├── primary_instruction_*      -> 详细的风格迁移指令
    └── style_caption              -> 风格概括
```

## 10. `sample_pics.py` 这段脚本实际做了什么

`sample_pics.py` 的逻辑可以概括为:

1. 打开 `MultiModalStorager`
2. 从 `sequences` 表拿到所有 `id -> meta`
3. 逐条取 `get_sequence_metas([seq_id])`
4. 打印该 sequence 下的 `images` 和 `texts`
5. 取第一张图的 `id`
6. 用 `get_image_bytes_by_ids()` 去 Lance 中取原始图片 bytes
7. 打印这条 sequence 的 `meta`

要注意的点:

- 脚本中的注释 `dict_keys(['id', 'meta'])` 已经过时
- 当前真实字段如本文第 6 节和第 7 节所示

## 11. 实际使用建议

如果后续要导出训练样本或做分析，建议按下面方式理解这份数据:

- 不要把 `images` / `texts` 当成独立样本主表
- 应该以 `sequence` 为主键组织样本
- 图片和文本的角色需要靠 `index` 区分
- 图片原始 bytes 需要去 Lance 读
- sequence 级 `meta` 需要单独查 `sequences.meta`

最常用的结构化视角是:

```text
sequence_id
  + sequence.uri / sequence.source / sequence.meta
  + role=cref_0 的图片
  + role=sref_0 的图片
  + role=target_image 的图片
  + role=captions/scene_1* 的文本
  + role=captions/scene_2* 的文本
  + role=captions/scene_3* 的文本
  + role=sample_instruction_* 的文本
  + role=primary_instruction_* 的文本
  + role=style_caption 的文本
```

## 12. 代码定位

如果要继续深入看实现，优先看这些文件:

- `sample_pics.py`
- `src/vault/schema/multimodal.py`
- `src/vault/storage/lanceduck/multimodal.py`
- `src/vault/storage/lanceduck/sql/schema.sql`
- `src/vault/storage/lanceduck/sql/get_sequences.sql`

