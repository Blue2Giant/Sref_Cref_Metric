# Examples 常用代码块和设计模式总结

本文档总结了 `examples/` 目录下的常用代码块、设计模式和最佳实践，可以作为开发新功能或编写文档的参考。

## 一、核心设计模式

### 1.1 分布式数据写入模式

**使用场景**：需要将外部标注结果（如模型推理输出、人工标注）批量写入 vault

**核心组件**：
- `DistributedDuckDBWriter` - 多进程分布式写入
- `DuckDBHandler` - 单独的 DuckDB 文件处理
- `merge()` - 合并临时 DuckDB 文件到主库

**典型流程**（参见 `examples/annotate/add_sample_annotations.py`）：

```python
# 步骤1: 创建独立的临时 DuckDB 文件
storager = MultiModalStorager(vault_path)
handler = DuckDBHandler(
    schema=storager.DUCKDB_SCHEMA,
    read_only=False,
    db_path=temp_duckdb_path
)
handler.create()

# 步骤2: 将数据写入临时文件
sample_annotations = []
for result in results:
    sample_annotation = SampleAnnotation.create(...)
    sample_annotations.append(sample_annotation)

storager.add_sample_annotations(sample_annotations, duckdb_handler=handler)

# 步骤3: 合并到主库
storager.merge(duckdb_files=(temp_duckdb_path,))
```

**优势**：
- 避免并发写入冲突
- 可以分批处理大量数据
- 失败时只影响单个临时文件
- 支持增量更新

---

### 1.2 Vault 路径发现模式

**使用场景**：批量处理多个 vault 数据集

**代码块**（参见 `examples/annotate/list_vault.py:202-230`）：

```python
def find_vault_paths(root_folder: str = "/mnt/marmot") -> list[str]:
    """
    递归查找一个文件夹下的所有 vault。

    判断标准：
    1. 包含 'metadata.duckdb' 文件
    2. 包含 'images' 文件夹

    找到后不再递归其子文件夹。
    """
    vault_paths = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        has_metadata = "metadata.duckdb" in filenames
        has_images_dir = "images" in dirnames

        if has_metadata and has_images_dir:
            vault_paths.append(dirpath)
            dirnames[:] = []  # 停止递归
    return vault_paths
```

**扩展：Vault 有效性验证**（参见 `examples/annotate/list_vault.py:250-263`）：

```python
def is_valid_vault(vault_path: str) -> bool:
    """根据 vault.toml 配置判断是否有效"""
    if not (Path(vault_path) / "vault.toml").exists():
        return False

    v = load_toml(Path(vault_path) / "vault.toml")
    return (
        "弃用" not in v["tags"]
        and "examples" not in vault_path
        and "QT:算法检查过" in v["tags"]
    )
```

**配套工具：加载 vault.toml**（参见 `examples/annotate/list_vault.py:233-247`）：

```python
def load_toml(file_path: str | Path) -> dict:
    """加载并处理 vault.toml 配置文件"""
    data = toml.load(str(file_path))

    # 处理 datetime 类型
    data["created_at"] = datetime.fromisoformat(data["created_at"])
    data["updated_at"] = datetime.fromisoformat(data["updated_at"])

    # 处理 index_descriptions（如果存在）
    if "index_descriptions" in data:
        index_descriptions = {}
        for index_str, description in data["index_descriptions"].items():
            index_type, index_name = index_str.split(":")
            index_descriptions[(index_type, index_name)] = description
        data["index_descriptions"] = index_descriptions

    return data
```

---

### 1.4 JSON 文本提取与验证模式

**使用场景**：从 LLM 输出中提取并验证 JSON 内容

**代码块**（参见 `examples/annotate/list_vault.py:33-84`）：

```python
def _extract_and_validate_json(text: str) -> str:
    """
    从响应文本中提取并验证JSON内容

    支持三种格式：
    1. 纯 JSON 文本
    2. ```json ... ``` 代码块
    3. 文本中嵌入的 JSON 对象
    """
    if not text:
        raise ValueError("响应文本为空")

    # 尝试1: 直接解析整个文本
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # 尝试2: 提取 ```json``` 包裹的内容
    json_pattern = r"```json\s*(.*?)\s*```"
    match = re.search(json_pattern, text, re.DOTALL | re.IGNORECASE)

    if match:
        json_content = match.group(1).strip()
        try:
            json.loads(json_content)
            return json_content
        except json.JSONDecodeError as e:
            raise ValueError(f"提取的JSON内容无效: {e}")

    # 尝试3: 查找以{开头，以}结尾的内容
    json_pattern = r"\{.*\}"
    match = re.search(json_pattern, text, re.DOTALL)

    if match:
        json_content = match.group(0).strip()
        try:
            json.loads(json_content)
            return json_content
        except json.JSONDecodeError as e:
            raise ValueError(f"提取的JSON内容无效: {e}")

    raise ValueError(f"无法从响应文本中提取有效的JSON内容。原始文本: {text[:200]}...")
```

---

### 1.5 Sample Annotation 写入模式

**使用场景**：将样本级标注（如模型评分、推理结果）写入 vault

**代码块**（参见 `examples/annotate/add_sample_annotations.py:266-318`）：

```python
def parquet_to_duckdb(
    model_name="Qwen3-VL-235B",
    name="describe_differences_20251015",
    vault_path="/path/to/vault",
    parquet_path="/path/to/results.parquet",
    duckdb_path="/path/to/output.duckdb",
):
    """将 parquet 格式的标注结果转换为 DuckDB 并写入 vault"""

    # 1. 初始化
    storager = MultiModalStorager(vault_path)
    handler = DuckDBHandler(
        schema=storager.DUCKDB_SCHEMA,
        read_only=False,
        db_path=duckdb_path
    )
    handler.create()

    # 2. 读取结果
    df = pd.read_parquet(parquet_path)

    # 3. 创建 Creator
    creator = Creator.create(
        name=name,
        meta=dict(
            model_name=model_name,
            # 其他元信息...
        ),
    )

    # 4. 构建 SampleAnnotation 对象
    sample_annotations = []
    for _, row in df.iterrows():
        # 从 URL 或其他来源提取 IDs
        sequence_id, element_ids = get_sequence_and_element_ids(row["vault_url"])
        if element_ids is None or sequence_id is None:
            continue

        source_image_id, instruction_id, target_image_id = element_ids

        # 解析数据
        try:
            json_str = _extract_and_validate_json(str(row["api_result"]))
            data = json.loads(json_str)
        except Exception:
            continue

        # 创建标注对象
        sample_annotation = SampleAnnotation.create(
            name=name,
            creator=creator,
            value=data,  # 存储为 JSON
            sequence_id=sequence_id,  # 关联到序列
            participants=(  # 多元素标注
                (source_image_id, MultiModalType.IMAGE, "source"),
                (instruction_id, MultiModalType.TEXT, "instruction"),
                (target_image_id, MultiModalType.IMAGE, "target"),
            ),
        )
        sample_annotations.append(sample_annotation)

    # 5. 写入
    storager.add_sample_annotations(sample_annotations, duckdb_handler=handler)
```

---

### 1.6 Sample Annotation 查询与转换模式

**使用场景**：将 sample_annotations 中的结构化数据提取为独立的 Text 对象

**代码块**（参见 `examples/annotate/add_sample_annotations.py:127-199`）：

```python
def sample_annotation_to_elements(vault_path: str):
    """将 sample_annotations 中的 JSON 数据转换为 Text 元素"""
    storager = MultiModalStorager(vault_path, read_only=False)

    # 1. 查询特定名称的 sample_annotations
    results = storager.meta_handler.query_batch(
        """
        SELECT
            COALESCE(value_json, json_object('value_float', value_float)) AS value_json,
            id,
            sequence_id
        FROM
            sample_annotations
        WHERE
            name = ?
        """,
        ["describe_differences_20251015"],
    )

    creator_name = "describe_differences_20251015"
    texts = []

    # 2. 遍历结果，提取需要的字段
    for result in results:
        data = json.loads(result["value_json"])
        sequence_id = ID.from_(result["sequence_id"])
        sample_annotation_id = ID.from_(result["id"])

        # 3. 从 JSON 中提取文本并创建 Text 对象
        texts.append(
            (
                Text.create(
                    content=data["training_output"]["primary_instruction"],
                    source=f"{creator_name}.instruction",
                    uri=f"{creator_name}/{sample_annotation_id}/primary_instruction",
                ),
                PackSequenceIndex(
                    sequence_id=sequence_id,
                    index="primary_instruction"
                ),
            )
        )

        # 提取其他字段...
        if "analyze_edit_instruction" in data:
            texts.append(...)

        for name in ["source", "target"]:
            texts.append(...)

    # 4. 批量添加 Text
    storager.add_texts(texts)
```

---

## 二、常用工具函数

### 2.1 数据分析与统计

**完整的 Vault 分析报告**（参见 `examples/analyze/report_vault.py`）：

```python
class DataVaultCommander:
    """Vault 数据分析命令行工具"""

    def __init__(self, vault_path: str):
        self.vault_path = vault_path
        self.lance_path = os.path.join(vault_path, "images")
        self.duckdb_path = os.path.join(vault_path, "metadata.duckdb")

    def profile_lance(self):
        """分析 Lance 图像数据集"""
        # - ID 与哈希值分析（唯一性、重复检测）
        # - 图像维度与尺寸分析
        # - 数据来源与路径分析
        # - 色彩与质量指标分析

    def profile_duckdb(self):
        """分析 DuckDB 元数据"""
        # - 核心表总览
        # - 序列分析（每个序列的图片数量统计）
        # - 孤立图片分析

    def cross_validate(self):
        """交叉验证 Lance 和 DuckDB 的数据一致性"""
        # - ID 完整性校验

    def full_report(self):
        """生成完整报告"""
        self.profile_lance()
        self.profile_duckdb()
        self.cross_validate()
```

**关键模式**：
- 使用 `lance.dataset()` 读取 Lance 元数据（不包含图像字节）
- 使用 pandas 的 `describe()` 进行统计分析
- 使用 rich 库美化命令行输出

---

### 2.2 批量查询与进度显示

**Rich 进度条 + 批量查询**（参见 `examples/annotate/list_vault.py:404-523`）：

```python
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

def analyze_multiple_vaults():
    console = Console()
    table = Table(title="Vault数据源统计")
    table.add_column("数据源名称", style="cyan")
    table.add_column("数据量", style="yellow", justify="right")
    table.add_column("已完成项目", style="red", justify="right")

    vault_paths = find_vault_paths()
    data_info = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("查询中...", total=len(vault_paths))

        for vault_path in vault_paths:
            progress.update(task, description=f"查询 {vault_path}")

            try:
                storager = MultiModalStorager(vault_path, read_only=True)
                count = storager.meta_handler.query_batch(
                    "SELECT COUNT(id) as count FROM sequences"
                )[0]["count"]

                completed_count = storager.meta_handler.query_batch(
                    "SELECT COUNT(sequence_id) as count FROM sample_annotations WHERE name = ?",
                    (name,),
                )[0]["count"]

                data_info.append({
                    "name": folder_name,
                    "count": count,
                    "completed": completed_count,
                })
            except Exception as e:
                console.print(f"[red]错误: {e}[/red]")

            progress.advance(task)

    # 显示结果
    for info in data_info:
        table.add_row(info["name"], f"{info['count']:,}", f"{info['completed']:,}")

    console.print(table)
```

---

### 2.3 Lance 与 Parquet 互转

**Lance → Parquet**（参见 `examples/annotate/list_vault.py:321-360`）：

```python
import lance
import pyarrow.parquet as pq

def convert_lance_to_parquet(lance_dataset_path: str, output_parquet_path: str):
    """将 Lance 数据集转换为 Parquet 文件"""
    lance_path = Path(lance_dataset_path)
    parquet_path = Path(output_parquet_path)

    if not lance_path.is_dir():
        raise ValueError(f"Lance 路径不存在: {lance_path}")

    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # 读取 Lance 数据集
    dataset = lance.dataset(str(lance_path))
    table = dataset.to_table()

    # 写入 Parquet
    pq.write_table(table, str(parquet_path))
```

---

### 2.4 分组数据导入模式

**使用场景**：将散落的图像文件按某种关系分组后导入为序列

**代码块**（参见 `examples/ingest/group_images_to_vault.py`）：

```python
@dataclass
class ArtStationIngestor(BaseIngestor):
    name: str = "artstation"
    vault_path: str = "/path/to/vault"
    task_chunk_size: int = 2000
    num_workers: int = 16
    root: str = "s3+b://data/artstation/"

    def scan_root(self):
        """扫描根目录，将分组信息保存到 JSONL"""
        with megfile.smart_open(f"{self.name}_tasks.jsonl", "wt") as f:
            for entry in tqdm(megfile.smart_scandir(self.root)):
                if entry.is_dir():
                    image_dir = megfile.smart_path_join(self.root, entry.name, "image")
                    image_files = megfile.smart_listdir(image_dir)

                    if len(image_files) > 1:  # 至少2张图片才算一组
                        f.write(json.dumps({
                            "image_dir": image_dir,
                            "image_files": image_files,
                            "id": entry.name,
                        }) + "\n")

    def prepare_tasks(self) -> list[Any]:
        """从 JSONL 读取任务列表"""
        with megfile.smart_open(f"{self.name}_tasks.jsonl", "rt") as f:
            return [json.loads(line) for line in f]

    def process_tasks(self, tasks_chunk: list[Any], worker_id: int):
        """处理每组图像"""
        for group in tqdm(tasks_chunk, position=worker_id % 8 + 1):
            images: list[tuple[multimodal.Image, str]] = []

            for f in group["image_files"]:
                image_path = megfile.smart_path_join(group["image_dir"], f)
                try:
                    image = multimodal.Image.create(
                        megfile.smart_load_content(image_path),
                        uri=str(megfile.SmartPath(image_path).relative_to(self.root)),
                        source=self.name,
                    )
                    images.append((image, "image"))
                except Exception as e:
                    logger.warning(f"加载图像失败 {image_path}: {e}")
                    continue

            if len(images) < 2:
                continue

            yield multimodal.PackSequence.create(
                images=images,
                texts=[],
                source=self.name,
                uri=group["id"],
            )
```

**关键点**：
1. 先扫描并生成 JSONL 任务列表（可复用）
2. 按组处理，确保每组至少有最小数量的图像
3. 使用 `try-except` 处理单个图像加载失败

---
