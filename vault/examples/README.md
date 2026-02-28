# Vault Examples

这个目录包含了 Vault 的各种使用示例，展示了如何导入、分析和查看多模态数据。

## 目录结构

```
examples/
├── ingest/          # 数据导入示例
├── analyze/         # 数据分析示例  
└── apps/            # 应用示例
```

## 数据导入示例 (ingest/)

### t2i_tar_to_vault.py

**用途**: 将文本到图像 (Text-to-Image) 数据集从 tar 格式导入到 Vault

**功能**:
- 处理 WebDataset 格式的 tar 文件
- 支持多种数据源 (EchoGPT4o, QwenTextualImage, CIRR)
- 并发处理大量数据
- 自动生成图像和文本的关联

**使用方法**:
```bash
# 运行 EchoGPT4o 数据导入
uv run python examples/ingest/t2i_tar_to_vault.py echo_gpt_4o

# 运行 Qwen 数据导入  
uv run python examples/ingest/t2i_tar_to_vault.py xp_250901

# 运行 CIRR 数据导入
uv run python examples/ingest/t2i_tar_to_vault.py cirr
```

**自定义数据源**:
```python
# 继承 BaseIngestor 创建自定义导入器
@dataclass
class MyCustomIngestor(BaseIngestor):
    def prepare_tasks(self) -> list[Any]:
        # 准备数据源任务
        pass
    
    def process_tasks(self, tasks_chunk, worker_id):
        # 处理数据并生成 PackSequence
        pass
```

## 数据分析示例 (analyze/)

### report_vault.py

**用途**: 对 Vault 数据进行全面的画像分析和报告生成

**功能**:
- Lance 数据集分析 (图像维度、质量指标、来源分布)
- DuckDB 元数据分析 (表统计、序列分析、孤立数据检测)
- 交叉完整性校验 (Lance 和 DuckDB 数据一致性检查)
- 生成详细的分析报告

**使用方法**:
```bash
# 分析指定 vault 的所有数据
uv run python examples/analyze/report_vault.py full_report --vault_path /path/to/vault

# 只分析 Lance 数据
uv run python examples/analyze/report_vault.py profile_lance --vault_path /path/to/vault

# 只分析 DuckDB 数据
uv run python examples/analyze/report_vault.py profile_duckdb --vault_path /path/to/vault

# 执行交叉校验
uv run python examples/analyze/report_vault.py cross_validate --vault_path /path/to/vault
```

**输出示例**:
```
📊 Lance 数据集画像报告
====================================
--- ID 与哈希值分析 ---
  - 图片ID: 唯一值: 10000 / 10000 (唯一性比例: 100.00%)
  - 文件哈希: 唯一值: 10000 / 10000 (唯一性比例: 100.00%)
  - 感知哈希(PDQ): 唯一值: 10000 / 10000 (唯一性比例: 100.00%)

--- 图像维度与尺寸分析 ---
         width    height  aspect_ratio    file_size
计数     10000     10000        10000        10000
平均值    512.0     512.0          1.0       45.2
标准差    256.0     256.0          0.0       12.1
```

## 应用示例 (apps/)

### vault_browser.py

**用途**: 基于 Gradio 的交互式 Vault 数据浏览器

**功能**:
- 加载和浏览 Vault 数据
- 按数据源筛选序列
- 查看序列中的图像和文本内容
- 自定义序列生成和 HTML 导出
- 支持随机浏览和顺序浏览

**使用方法**:
```bash
# 启动浏览器应用
uv run python examples/apps/vault_browser.py
```

**界面功能**:
1. **数据加载**: 输入 vault 路径，加载可用的数据源
2. **序列浏览**: 选择数据源，加载序列列表
3. **内容查看**: 使用滑块或随机选择浏览序列内容
4. **自定义展示**: 选择元素生成自定义的 HTML 序列

**依赖安装**:
```bash
# 安装应用依赖
uv sync --group apps
```

## 运行环境要求

### 基础依赖
```bash
# 安装基础依赖
uv sync
```

### 应用依赖
```bash
# 安装 Gradio 应用依赖
uv sync --group apps
```

### 开发依赖
```bash
# 安装开发工具
uv sync --group dev
```

## 自定义示例

### 创建新的数据导入器

```python
# examples/ingest/my_custom_ingestor.py
from vault.utils.ingest import BaseIngestor
from vault.schema.multimodal import PackSequence, Image, Text

@dataclass
class MyCustomIngestor(BaseIngestor):
    name: str = "MyCustomDataset"
    vault_path: str = "/path/to/vault"
    task_chunk_size: int = 100
    num_workers: int = 4
    
    def prepare_tasks(self) -> list[Any]:
        # 返回需要处理的任务列表
        return ["task1", "task2", "task3"]
    
    def process_tasks(self, tasks_chunk: list[Any], worker_id: int):
        for task in tasks_chunk:
            # 处理单个任务，生成 PackSequence
            yield PackSequence.create(...)

if __name__ == "__main__":
    import fire
    fire.Fire(MyCustomIngestor)
```

### 创建新的分析工具

```python
# examples/analyze/my_analyzer.py
import duckdb
import pandas as pd
from vault.storage.lanceduck.multimodal import MultiModalStorager

class MyAnalyzer:
    def __init__(self, vault_path: str):
        self.storager = MultiModalStorager(vault_path)
    
    def analyze_custom_metrics(self):
        # 自定义分析逻辑
        results = self.storager.meta_hanlder.query_batch(
            "SELECT * FROM sequences WHERE ..."
        )
        return results

if __name__ == "__main__":
    import fire
    fire.Fire(MyAnalyzer)
```

## 最佳实践

### 1. 数据导入
- 使用合适的 `task_chunk_size` 和 `num_workers` 参数
- 定期调用 `storager.commit()` 保存进度
- 处理异常情况，确保数据完整性

### 2. 数据分析
- 使用 SQL 查询进行复杂分析
- 结合 Lance 和 DuckDB 的优势
- 生成可读性强的报告

### 3. 应用开发
- 使用 Gradio 创建交互式界面
- 提供清晰的用户指导
- 处理大数据的性能优化

## 贡献示例

欢迎提交新的示例！请确保：

1. **代码质量**: 遵循项目的代码风格
2. **文档完整**: 提供清晰的说明和使用方法
3. **功能实用**: 解决实际的数据处理需求
4. **易于理解**: 代码结构清晰，注释充分

## 获取帮助

如果遇到问题或需要帮助：

1. 查看 [使用指南](../docs/使用指南.md)
2. 阅读 [设计哲学](../docs/design_philosophy.md)
3. 参考现有示例的实现
4. 提交 Issue 或联系维护者

---

**注意**: 这些示例会高频更新，请定期查看最新版本。如有需要，可以找我聊，然后我改。
