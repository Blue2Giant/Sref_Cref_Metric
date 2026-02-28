# Vault 设计哲学与架构

## 核心理念

Vault 的设计遵循以下核心理念，这些理念指导着项目的每一个技术决策：

### 1. AI-First 设计

**理念**: 让 AI 能够轻松理解和操作数据

**实现方式**:
- 采用不可变数据类 (Immutable Dataclass) 设计，提供清晰的类型信息和上下文
- 文件结构简单透明，AI 可以直接理解底层存储格式
- 提供丰富的元数据信息，帮助 AI 理解数据的语义和关系
- 避免复杂的抽象层，保持 API 的直观性

**示例**:
```python
# 清晰的数据模型，AI 可以轻松理解
@dataclass(frozen=True)
class Image:
    id: ID
    uri: str
    source: str
    pil_image: PIL.Image.Image
    blob: bytes
    annotations: list[Annotation] | None = None
```

### 2. 稳定性优先

**理念**: 库本身保持相对稳定，核心功能不会频繁变更

**实现方式**:
- 核心 API 设计简洁，避免过度抽象
- 版本兼容性优先，向后兼容性得到保证
- 新功能通过扩展而非修改现有接口实现
- 底层文件格式稳定，支持长期数据存储

**设计原则**:
- 一旦发布的核心 API，尽量不进行破坏性变更
- 新功能通过新的类或方法添加，而非修改现有接口
- 保持数据格式的向前兼容性

### 3. API 优先的扩展性

**理念**: 通过暴露底层操作 API 来扩展功能，而非添加过多高级功能

**实现方式**:
- 提供基础的 CRUD 操作
- 暴露底层存储系统的直接访问能力
- 支持自定义数据处理逻辑
- 通过组合而非继承来扩展功能

**架构层次**:
```
高级 API (MultiModalStorager)
    ↓
中层 API (DuckDBHandler, LanceTaker)  
    ↓
底层存储 (DuckDB, Lance)
```

### 4. 透明性

**理念**: 底层文件结构简单透明，用户可以完全理解和控制

**实现方式**:
- 使用标准的文件格式 (Lance, DuckDB)
- 文件结构清晰，可以直接用标准工具访问
- 提供完整的元数据信息
- 支持直接 SQL 查询和操作

**文件结构**:
```
vault/
├── images/              # Lance 格式的图像数据
├── metadata.duckdb      # DuckDB 格式的元数据
└── sequences/           # 序列数据 (可选)
```

## 技术架构

### 混合存储架构

Vault 采用 Lance + DuckDB 的混合存储架构：

- **Lance**: 用于存储大型二进制数据 (图像、文本内容)
- **DuckDB**: 用于存储结构化元数据和关系信息

这种设计的优势：
- 充分利用了两种存储系统的优势
- 支持高效的查询和分析
- 文件格式标准，工具生态丰富

### 数据模型设计

#### 不可变数据类

所有数据模型都采用不可变数据类设计：

```python
@dataclass(frozen=True)
class PackSequence:
    id: ID
    images: Sequence[tuple[Image, str | int]]
    texts: Sequence[tuple[Text, str | int]]
    source: str
    uri: str
    annotations: list[Annotation] | None = None
```

**优势**:
- 线程安全
- 避免意外的数据修改
- 提供清晰的类型信息
- 支持哈希和比较操作

#### ID 系统

使用基于内容的哈希 ID 系统：

```python
class ID:
    @classmethod
    def hash(cls, *args) -> "ID":
        # 基于内容生成确定性 ID
```

**优势**:
- 内容去重
- 确定性 ID 生成
- 支持分布式环境
- 避免 ID 冲突

### 并发处理

#### 分布式写入

支持多进程并发写入：

```python
class DistributedLanceWriter:
    def __init__(self, num_workers: int = 4):
        # 多进程写入支持
```

#### 任务分块

自动将大任务分解为小块：

```python
def run_concurrently(func, tasks, num_workers):
    # 自动任务分块和分发
```

## 扩展性设计

### 自定义 Ingestor

通过继承 `BaseIngestor` 来支持新的数据源：

```python
@dataclass
class CustomIngestor(BaseIngestor):
    def prepare_tasks(self) -> list[Any]:
        # 准备数据源任务
        
    def process_tasks(self, tasks_chunk, worker_id):
        # 处理数据并生成 PackSequence
```

### 直接底层访问

支持直接访问底层存储系统：

```python
# 直接使用 DuckDB
storager.meta_hanlder.conn.execute("SELECT * FROM sequences")

# 直接使用 Lance
taker = LanceTaker()
images = taker.by_ids(lance_uri, image_ids)
```

### 插件化架构

通过组合而非继承来扩展功能：

```python
# 可以轻松添加新的存储后端
class CustomBackend:
    def store(self, data):
        # 自定义存储逻辑
```

## 性能优化

### 批量操作

所有操作都支持批量处理：

```python
storager.add_sequences(sequences)  # 批量添加
storager.get_sequence_metas(ids)   # 批量查询
```

### 内存管理

- 流式处理大文件
- 延迟加载图像数据
- 智能缓存策略

### 并发优化

- 多进程写入
- 异步 I/O 支持
- 任务队列管理

## 未来发展方向

### 短期目标

1. 完善文档和示例
2. 优化性能瓶颈
3. 增加更多数据源支持

### 长期愿景

1. 支持更多存储后端
2. 提供更丰富的查询接口
3. 集成更多 AI 工具链

## 总结

Vault 的设计哲学可以概括为：**简单、透明、稳定、可扩展**。我们相信，通过保持核心设计的简洁性，同时提供足够的扩展能力，可以让用户和 AI 都能轻松地使用和扩展这个系统。

这种设计理念不仅体现在代码架构上，也体现在文档、示例和社区建设上。我们希望通过这种"共建"的方式，让 Vault 成为一个真正有用的多模态数据管理工具。
