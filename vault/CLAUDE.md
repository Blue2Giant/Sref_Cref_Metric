## 项目概述

**Vault** 是一个面向大规模机器学习工作流的 AI 优先多模态数据存储和管理库。它采用混合存储架构，结合了：
- **Lance**（列式格式）用于高效的二进制数据存储（图像、标注），支持随机访问
- **DuckDB**（关系型数据库）用于结构化元数据和 SQL 查询

## 架构概述

### 存储架构

Vault 使用**双层存储系统**：

1. **Lance 层**（二进制数据存储）
   - 位置：`{vault_path}/images/` 和 `{vault_path}/annotations/`
   - 存储：原始图像字节、标注数据、质量指标
   - 特性：列式格式、随机访问、版本控制、BTREE 索引

2. **DuckDB 层**（元数据存储）
   - 位置：`{vault_path}/metadata.duckdb`
   - 存储：关系、URI、来源、结构化元数据
   - 模式：9 个规范化表，包含用于多对多关系的连接表

### 核心数据模型

位于 `src/vault/schema/multimodal.py`：

- **`Image`**：不可变数据类，用于图像数据，支持 PIL
- **`Text`**：不可变数据类，用于文本内容，支持语言跟踪
- **`PackSequence`**：逻辑容器，用于相关图像/文本，支持排序
- **`Annotation`**：灵活的标注系统，支持创建者跟踪
- **`ID`**：128 位基于内容的哈希标识符（确定性，支持去重）
  - `ID.from_()` 可以方便地把文本/bytes/UUID等不同格式转为`ID`，`ID`包含很多method转为string/bytes/UUID。
  - `ID.hash()` 可以快速计算一个python对象的hash值

### DuckDB 模式（11 个表）

**核心表：**
1. `creators` - 用户/标注者信息
2. `images` - 图像元数据（尺寸、来源、URI）
3. `texts` - 文本内容，支持语言跟踪
4. `sequences` - 内容的逻辑分组
5. `annotations` - **共享标注系统**（可复用的标签）
6. `sample_annotations` - **样本级标注系统**（每个样本独有的标注）

**连接表**（多对多关系）：
7. `image_annotations` - 链接图像 ↔ 共享标注
8. `text_annotations` - 链接文本 ↔ 共享标注
9. `sequence_images` - 链接序列 ↔ 图像（通过 `index` 字段排序）
10. `sequence_texts` - 链接序列 ↔ 文本（通过 `index` 字段排序）
11. `sample_annotation_elements` - 链接样本标注 ↔ 元素（支持多元素标注）

**重要**：
- 连接表中的 `index` 字段可以是数字或语义字符串（例如 "caption"、"caption_cn"、"image_1"、"gallery"），以保留有意义的关系
- `sample_annotation_elements` 中的 `role` 字段用于区分多元素标注中每个元素的角色（例如 "source_image", "target_image"）

### 完整的 DuckDB Schema SQL

DuckDB Schema 由两部分组成：
1. `src/vault/storage/lanceduck/sql/schema.sql` - 基础多模态数据模式（9 个表）
2. `src/vault/storage/lanceduck/sql/sample_annotations_schema.sql` - 扩展的样本级标注系统（2 个表）

## 数据导出工作

## 重要实现细节

### ID 系统

`ID` 类（位于 `src/vault/schema/__init__.py`）支持：
- 基于内容的哈希（xxhash，确定性）
- UUID 表示
- 转换：`ID.from_uuid()`、`ID.from_hex()`、`ID.from_bytes()`
- 使用 `ID.from_()` 进行灵活的输入处理


## 开发实践

### 错误处理

- 使用 `loguru.logger` 进行日志记录
- 在提交前验证数据

### 性能考虑

- 使用 `megfile` 进行 S3/OSS 访问

## 常见陷阱

1. **忘记提交**：添加数据后始终调用 `storager.commit()`
2. **标注系统选择错误**：
   - 不要将每个样本独有的评分存入 `annotations`（会创建大量重复记录）
   - 不要将可复用的标签存入 `sample_annotations`（无法利用去重优势）
   - `annotations` 适合"标签"，`sample_annotations` 适合"评分/测量值"
3. **sample_annotations 的 sequence_id**：虽然是可选字段，但强烈建议填写以明确标注的上下文
4. **多元素标注的 role 字段**：必须明确指定每个元素的角色，避免歧义（如 "source_image" vs "target_image"）

## skills

下面列出来一些你做具体任务时，可以参考的文档：

- 当需要为 Vault 中的标注数据生成分析报告时，阅读[标注数据分析报告制作指南](/data/code/vault/docs/skills/标注数据分析报告制作指南.md)
