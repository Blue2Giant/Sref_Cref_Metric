# Vault

Vault 是一个数据的存储和管理库，支持多模态数据的高效写入和**随机**读取。

采用 AI-first 的设计理念，提供简单透明的底层文件结构，让 AI 能够轻松理解和操作数据。

## 🎯 核心功能

### ✅ 提供什么
- **多模态数据存储**: 支持图像、文本及其组合序列的高效存储，存储占用空间几乎与tar一致
- **数据的随机读取**: 提供API支持高效地完全随机读取特定图片or文本，加速训练
- **透明文件结构**: 基于 Lance (列式存储) + DuckDB (关系型元数据) 的混合架构
- **AI 友好的数据模型**: 不可变数据类设计，核心类提供清晰的上下文信息
- **基础操作 API**: 数据添加、查看、查询等核心功能
- **并发处理支持**: 内置分布式写入和并发处理能力

### ❌ 不提供什么
- 复杂的数据筛选分析工具 (通过底层 API 自行实现)
- 高级查询语言 (直接使用 SQL 操作 DuckDB)
- 不同来源、组织方式数据集的导入脚本（每个人自行撰写，不进入该仓库）

### 提供了哪些示例

- 数据可视化界面 (examples 中的 Gradio 应用)：展示如何**随机读数据，数据组织存储格式**
- 数据转换和预处理 (可扩展自定义 Ingestor 实现)：展示如何**批量多进程导入数据**，比如tar、oss散文件等，几分钟内导入100K数据
- 数据集分析（report整个数据集meta信息）：展示如何**秒加载1M数据集的元数据**并分析统计

## 🚀 快速开始

### 安装

推荐安装方式

```bash
# 安装最新版本vault
pip install -i http://mirrors.i.basemind.com/pypi/simple/ --trusted-host mirrors.i.basemind.com "$(megfile cat s3+b://ruiwang/pypi/vault/latest.txt | xargs -I {} sh -c 'megfile cp s3+b://ruiwang/pypi/vault/{} /tmp/{} >/dev/null && echo /tmp/{}')"
```

其它安装方式

```bash
# 指定版本, sync特定whl，然后安装
VER=0.2.0 && pip install -i http://mirrors.i.basemind.com/pypi/simple/ --trusted-host mirrors.i.basemind.com "$(megfile cp s3+b://ruiwang/pypi/vault/step_vault-${VER}-py3-none-any.whl /tmp/step_vault-${VER}-py3-none-any.whl >/dev/null && echo /tmp/step_vault-${VER}-py3-none-any.whl)"

# 也可以clone所有版本，然后装最新的
megfile sync -g s3+b://ruiwang/pypi/vault/ ./vault
# 把./vault当做一个pypi源，安装指定的包, 这里就可以指定版本，像是正常的pypi源
pip install step-vault --find-links ./vault

# 也可以clone项目后pip install -e安装
```


### 基本使用

```python
import vault
from vault.schema.multimodal import Image, Text, PackSequence
from vault.storage.lanceduck.multimodal import MultiModalStorager

# 一个创建pil imgae 的helper函数
from vault.utils.image import create_text_image 

# 1. 初始化存储
vault_path = "/tmp/test_vault_path"
MultiModalStorager.init(vault_path)
storager = MultiModalStorager(vault_path)

example_pil_image = create_text_image("hello, world!")

# 2. 创建数据
image = Image.create(
    example_pil_image,  # 图像字节数据 , 或者pil image
    uri="/path/to/image.jpg",
    source="my_dataset"
)

text = Text.create(
    content="这是一段文本",
    uri="/path/to/text.txt", 
    source="my_dataset"
)

# 3. 创建序列
sequence = PackSequence.from_text_to_image(
    caption=text,
    image=image,
    source="my_dataset",
    uri="/path/to/sequence"
)

# 4. 存储数据
storager.add_sequences([sequence])
storager.commit()
```

你将会看到类似于下面的输出:

```bash
[2025-09-17T07:00:57Z WARN  lance::dataset::write::insert] No existing dataset at /tmp/test_vault_path/images, it will be created
2025-09-17 15:00:57.947 | INFO     | vault.backend.duckdb:commit:285 - Starting merge and cleanup process...
2025-09-17 15:00:57.947 | INFO     | vault.backend.duckdb:commit:294 - Found 1 temporary DB files to merge.
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 66.31it/s]
2025-09-17 15:00:57.988 | INFO     | vault.backend.duckdb:_cleanup_temp_dir:331 - Temporary directory '/tmp/test_vault_path/_duckdb' has been removed.
2025-09-17 15:00:57.988 | INFO     | vault.backend.duckdb:commit:324 - Merge and cleanup process completed.
2025-09-17 15:00:57.989 | INFO     | vault.backend.lance:commit:222 - Found 1 fragments in /tmp/test_vault_path/images/_lance_fragments from 1 workers to commit. 
2025-09-17 15:00:57.989 | INFO     | vault.backend.lance:commit:242 - Coordinator: Dataset not found at '/tmp/test_vault_path/images'. Creating new dataset from fragments.
2025-09-17 15:00:57.990 | SUCCESS  | vault.backend.lance:commit:257 - Coordinator: Successfully committed 1 fragments.
2025-09-17 15:00:57.997 | INFO     | vault.backend.lance:_create_id_index_if_exists:317 - Successfully created BTREE index for 'id' column in dataset at '/tmp/test_vault_path/images'
2025-09-17 15:00:57.997 | DEBUG    | vault.backend.lance:commit:285 - Metadata cache in /tmp/test_vault_path/images/_lance_fragments cleared successfully.
2025-09-17 15:00:57.997 | INFO     | vault.backend.lance:commit:217 - No new fragments found in /tmp/test_vault_path/annotations/_lance_fragments. Nothing to commit.
```


`storager.schema_summary()` 详细说明了vault底层存储结果，其内容很适合交给AI模型，可以和靠谱的AI模型交流这个结构设计，帮你写SQL等。

``` python
print(storager.schema_summary())
```

将会输出下面的内容：

``` text
MultiModalStorager Schema Summary
Storage Path: /tmp/test_vault_path
================================================================================

DESIGN PHILOSOPHY
--------------------------------------------------
  Hybrid Storage Architecture:
     • Lance: Columnar format with random access for binary data (images, annotations)
     • DuckDB: Relational metadata with ACID compliance and SQL queries
     • Separation of concerns: Data vs Metadata

  Multi-Modal Data Model:
     • Sequences: Logical containers for related content
     • Images & Texts: Core content with rich metadata
     • Annotations: Flexible labeling system with creator tracking
     • Index-based ordering: Preserves content sequence and relationships

  Performance Optimizations:
     • Distributed writing: Concurrent processing with atomic commits
     • Image quality metrics: Clarity, entropy, edge detection for filtering
     • PDQ hashing: Advanced perceptual hashing for duplicate detection
     • Strategic indexing: Optimized for common query patterns

LANCE SCHEMA (PyArrow Tables)
--------------------------------------------------
  Purpose: High-performance columnar storage with random access for binary content
  Technology: Lance format with PyArrow for columnar efficiency
  Content: Raw images, annotation blobs, and computed features

  1. Table: images
     URI: /tmp/test_vault_path/images
     Design: Stores raw image bytes + computed quality metrics
     Features: PDQ hash, clarity, entropy, edge detection
     Processing: 512x512 RGB conversion for consistent analysis
     Schema:
        id: extension<arrow.uuid> not null
        image: binary not null
        uri: string
        source: string
        file_hash: fixed_size_binary[16]
        file_size: int64
        width: int64
        height: int64
        aspect_ratio: float
        color_mode: string
        mean_saturation: float
        mean_lightness: float
        clarity: float
        entropy: float
        edge_probability: float
        edge_near_patch_min_std: float
        pdq_hash: fixed_size_binary[32]
        pdq_quality: float

  2. Table: annotations
     URI: /tmp/test_vault_path/annotations
     Design: Flexible annotation system with creator tracking
     Content: Structured labels, binary blobs, metadata
     Relations: Links to images/texts via junction tables
     Schema:
        id: extension<arrow.uuid> not null
        name: string
        type_: string
        creator_name: string
        blob: binary
        meta: string


DUCKDB SCHEMA (SQL Tables)
--------------------------------------------------
  Purpose: Relational metadata with ACID compliance and SQL queries
  Technology: DuckDB for analytical workloads and complex joins
  Content: Structured metadata, relationships, and searchable attributes
  Design: Normalized schema with junction tables for many-to-many relationships

  Tables:
    1. creators - User/creator management with metadata
    2. annotations - Flexible labeling system with creator tracking
    3. images - Image metadata (dimensions, source, URI)
    4. texts - Text content with language and source tracking
    5. sequences - Logical containers for related content groups
    6. image_annotations - Many-to-many: Images ↔ Annotations
    7. text_annotations - Many-to-many: Texts ↔ Annotations
    8. sequence_images - Many-to-many: Sequences ↔ Images (with ordering)
    9. sequence_texts - Many-to-many: Sequences ↔ Texts (with ordering)

  SQL Schema:
    CREATE TABLE IF NOT EXISTS creators (
    id UUID PRIMARY KEY,
    name VARCHAR NOT NULL,
    meta JSON
    );

    CREATE TABLE IF NOT EXISTS annotations (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    type VARCHAR,
    creator_id UUID REFERENCES creators(id),
    meta JSON
    );

    CREATE TABLE IF NOT EXISTS images (
    id UUID PRIMARY KEY,
    uri VARCHAR NOT NULL,
    source VARCHAR,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS texts (
    id UUID PRIMARY KEY,
    content TEXT NOT NULL,
    uri VARCHAR,
    source VARCHAR,
    language VARCHAR(10),
    );

    CREATE TABLE IF NOT EXISTS sequences (
    id UUID PRIMARY KEY,
    uri VARCHAR,
    source VARCHAR,
    meta JSON
    );


    -- 图片与 annotation 的关联表
    CREATE TABLE IF NOT EXISTS image_annotations (
    image_id UUID NOT NULL REFERENCES images(id),
    annotation_id UUID NOT NULL REFERENCES annotations(id),
    PRIMARY KEY (image_id, annotation_id)
    );

    -- 文本与 annotation 的关联表
    CREATE TABLE IF NOT EXISTS text_annotations (
    text_id UUID NOT NULL REFERENCES texts(id),
    annotation_id UUID NOT NULL REFERENCES annotations(id),
    PRIMARY KEY (text_id, annotation_id)
    );

    -- 序列与图片的关联表
    -- 存储图片在序列中的位置
    CREATE TABLE IF NOT EXISTS sequence_images (
    sequence_id UUID NOT NULL REFERENCES sequences(id),
    image_id UUID NOT NULL REFERENCES images(id),
    "index" VARCHAR NOT NULL,  -- 图片在序列中的索引/顺序
    PRIMARY KEY (sequence_id, image_id, "index")
    );

    -- 序列与文本的关联表
    -- 存储文本在序列中的位置
    CREATE TABLE IF NOT EXISTS sequence_texts (
    sequence_id UUID NOT NULL REFERENCES sequences(id),
    text_id UUID NOT NULL REFERENCES texts(id),
    "index" VARCHAR NOT NULL,   -- 文本在序列中的索引/顺序
    PRIMARY KEY (sequence_id, text_id, "index")
    );


    -- 加速通过 annotation_id 查找 image
    CREATE INDEX IF NOT EXISTS idx_image_annotations_annotation_id ON image_annotations(annotation_id);

    -- 加速通过 annotation_id 查找 text
    CREATE INDEX IF NOT EXISTS idx_text_annotations_annotation_id ON text_annotations(annotation_id);

    -- 加速标注名称和类型查找
    CREATE INDEX IF NOT EXISTS idx_annotations_name ON annotations(name);
    CREATE INDEX IF NOT EXISTS idx_annotations_type ON annotations(type);

    -- 加速图片和文本的 URI/Source 查找 (如果这些列是唯一的，可使用 UNIQUE 索引)
    CREATE INDEX IF NOT EXISTS idx_images_uri ON images(uri);
    CREATE INDEX IF NOT EXISTS idx_texts_uri ON texts(uri);
    CREATE INDEX IF NOT EXISTS idx_images_source ON images(source);
    CREATE INDEX IF NOT EXISTS idx_texts_source ON texts(source);

    -- 加速从 creator 查找 annotations
    CREATE INDEX IF NOT EXISTS idx_annotations_creator_id ON annotations(creator_id);

    -- 加速通过 image_id 反向查找 sequence
    CREATE INDEX IF NOT EXISTS idx_sequence_images_image_id ON sequence_images(image_id);
    -- 加速通过 text_id 反向查找 sequence
    CREATE INDEX IF NOT EXISTS idx_sequence_texts_text_id ON sequence_texts(text_id);

================================================================================
Summary: 2 Lance tables, 9 DuckDB tables
Storage: Lance format for binary data, DuckDB for metadata
```

我把这个示例vault库copy到了`/mnt/marmot/examples/readme_vault`，你可以从下面的链接中，直接可视化看看其结构：

[vault:/mnt/marmot/examples/readme_vault](https://vault.iap.platform.shaipower.com/multi-modal/?vault_path=%2Fmnt%2Fmarmot%2Fexamples%2Freadme_vault&sources=my_dataset&sequence_index=1)

### vault ID

vault中序列、文本、图片都有自己的ID，可以用于查询、标识数据。

每个ID都是128bit，可以来源于内容的hash，也可以是随机UUID标识。

`ID` 类导入方式：

```
from vault.schema import ID
```

ID类详细的说明文档：[docs/ID类.md](docs/ID类.md)

### 查看数据

```python
# 查询序列
sequences = storager.get_sequence_metas([sequence_id])

# 直接使用 SQL 查询元数据
results = storager.meta_hanlder.query_batch(
    "SELECT * FROM sequences WHERE source = ?",
    ["my_dataset"]
)

# 从 Lance 表读取图像数据
from vault.backend.lance import LanceTaker
taker = LanceTaker()
images = taker.by_ids(lance_uri, [image_id])
```

### 添加样本级标注

为已有的 Vault 数据添加样本级标注（如模型评分、质量分数等）：

```python
from vault.schema import ID
from vault.schema.multimodal import Creator, MultiModalType, SampleAnnotation
from vault.backend.duckdb import DuckDBHandler

# 1. 查询需要标注的数据
storager = MultiModalStorager(vault_path, read_only=True)
with storager.meta_handler as handler:
    sequences = handler.query_batch(
        "SELECT s.id as sequence_id, si.image_id FROM sequences s "
        "JOIN sequence_images si ON s.id = si.sequence_id LIMIT 100"
    )

# 2. 创建标注
creator = Creator.create(name="aesthetic_scorer_v1", meta={"model": "v2.5"})
annotations = []

for item in sequences:
    # 这里应该是实际的模型推理
    score = 7.5  # 你的评分逻辑

    annotation = SampleAnnotation.create(
        name="aesthetic_score",
        sequence_id=ID.from_(item["sequence_id"]),
        creator=creator,
        value=score,  # 数字会存入 value_float，复杂对象存入 value_json
        participants=(
            (ID.from_(item["image_id"]), MultiModalType.IMAGE, "target"),
        ),
    )
    annotations.append(annotation)

# 3. 写入临时数据库
temp_handler = DuckDBHandler(
    schema=storager.DUCKDB_SCHEMA,
    read_only=False,
    db_path="/tmp/my_annotations.duckdb"
)
temp_handler.create()

storager_rw = MultiModalStorager(vault_path, read_only=False)
storager_rw.add_sample_annotations(annotations, duckdb_handler=temp_handler)

# 4. 合并到主 Vault
storager_rw.merge(duckdb_files=["/tmp/my_annotations.duckdb"])
```

**更多场景**：
- 图文对 CLIP 相似度标注
- 多元素标注（图像编辑三元组）
- 从 CSV/Parquet 批量导入
- 分布式多进程标注

详见 [样本级标注完整教程](examples/annotate/README.md)

## 📚 文档

### 使用文档
- [新增数据源导入指南](docs/新增数据源导入指南.md) - 如何基于 `BaseIngestor` 创建数据导入器
- [样本级标注完整教程](examples/annotate/README.md) - 为 Vault 添加模型评分、质量标注等
- [Examples 常用模式总结](docs/examples常用模式总结.md) - 常用代码块和设计模式
- [Examples 目录](examples/) - 包含导入、分析、查看数据库的完整示例

### 开发文档
- [设计哲学与架构](docs/design_philosophy.md) - 项目的核心设计理念
- [不可变数据类风格指南](docs/design_guides/immutable_dataclass_style.md) - 数据模型设计模式

## 🏗️ 项目结构

```
src/vault/           # 核心库代码
├── backend/         # 存储后端适配层 (DuckDB / Lance)
├── schema/          # 数据模型 & 不可变数据类定义  
├── storage/         # 高层封装的存取逻辑
├── utils/           # 工具函数 & 动态链接库
examples/            # 使用示例 (高频更新)
├── ingest/          # 数据导入示例
├── analyze/         # 数据分析示例  
└── apps/            # 应用示例 (如 Gradio 浏览器)
docs/                # 设计文档
├── design_guides/   # 设计模式和架构指南
└── 使用指南.md      # 具体使用说明
tests/               # 单元测试
```

## 🔧 如何开发


推荐使用 [uv](https://github.com/astral-sh/uv) 管理依赖：

```bash
# 快速安装`uv`:
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆仓库
git clone git@gitlab.basemind.com:step_aigc/vault.git
cd vault

# 安装依赖
uv sync
```


### 激活开发环境

```bash
# 激活虚拟环境
uv venv
source .venv/bin/activate   # Linux / macOS
# 或
.venv\Scripts\activate      # Windows

# 安装开发依赖
uv sync --group dev
```

### 运行示例

```bash
# 运行数据导入示例
uv run python examples/ingest/t2i_tar_to_vault.py --help

# 运行数据分析示例  
uv run python examples/analyze/report_vault.py --help

# 启动 Gradio 浏览器应用
uv run python examples/apps/vault_browser.py
```

### 测试

```bash
# 运行所有测试
uv run pytest

# 进入交互调试
uv run ipython
```

## 🤝 贡献指南

### 设计哲学

1. **稳定性优先**: 库本身保持相对稳定，核心功能不会频繁变更
2. **API 优先**: 通过暴露底层操作 API 来扩展功能，而非添加过多高级功能
3. **透明性**: 底层文件结构简单透明，AI 可以轻松理解和操作
4. **AI First**: 提供清晰的上下文信息，让 AI 能够写出合理有用的代码

### 如何贡献

- **功能需求**: 如有新功能需求，请先讨论设计，确认符合项目哲学
- **示例更新**: `examples/` 目录会高频更新，欢迎提交新的使用示例
- **文档改进**: 欢迎改进文档的清晰度和完整性
- **Bug 修复**: 发现 bug 请及时反馈

### 共建方式

有什么需要可以找我聊，然后我改。项目采用协作共建的方式，欢迎各种形式的贡献！

## 📦 构建与发布

```bash
# 构建包
uv build

# 安装构建产物测试
pip install dist/vault-0.1.0-py3-none-any.whl
```

