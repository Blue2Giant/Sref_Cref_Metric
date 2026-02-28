# LLM 标注工具快速上手

用 VLM 大规模标注 Vault 数据的模块化工具包

## 架构概览

```
examples/llm_annotate/
├── llm_annotator/           # 稳定核心工具 (API 稳定)
│   ├── base.py              # 基类: AnnotateTool, VaultAnnotator
│   ├── utils.py             # 工具函数: VLM 调用、重试、JSON 解析
│   └── prompt_loader.py     # Jinja2 模板加载
└── tools/                   # 用户可扩展的标注工具
    ├── *.py                 # 你的工具实现
    └── templates/           # 提示词模板 (.j2)
```

## 5分钟快速创建新工具

### 1. 复制模板

```bash
# 复制现有工具作为模板
cp tools/stepflow_tag_tool.py tools/my_tool.py
```

### 2. 修改必须实现的3个方法

```python
from llm_annotator import AnnotateTool
from llm_annotator.utils import call_vlm_single
from llm_annotator.prompt_loader import load_prompt
from vault.schema.multimodal import MultiModalType
from openai import OpenAI
from pathlib import Path

class MyTool(AnnotateTool):
    basic_name: str = "my_tool"          # 工具标识
    sample_type: str = "image"           # "image", "text", or "sequence"

    def __init__(self, model_name: str):
        super().__init__(model_name)

        # 加载提示词模板
        template_dir = Path(__file__).parent / "templates"
        self.prompt = load_prompt("my_template.j2", template_dir=template_dir)
        self.creator_meta = dict(prompt=self.prompt)

        # 初始化 API 客户端
        self.client = OpenAI(api_key="EMPTY", base_url="YOUR_API_URL", timeout=3600)

    def _prepare_kwargs_and_participants(self, sample, storager):
        """准备数据 - 必须实现"""
        # 从 vault 加载数据
        image_bytes = storager.get_image_bytes_by_ids([sample["sample_id"]])

        # 定义参与者 (标注关联的元素)
        participants = ((sample["sample_id"], MultiModalType.IMAGE, "image"),)

        # 准备 API 调用参数
        kwargs = {
            "image": image_bytes[sample["sample_id"]],
            "text": self.prompt,
        }
        return participants, kwargs

    def __call__(self, image, text, **kwargs):
        """执行 VLM 调用 - 必须实现"""
        return call_vlm_single(
            image=image,
            text=text,
            model_name=self.model_name,
            client=self.client,
        )
```

### 3. 创建提示词模板

```bash
# 在 tools/templates/ 创建 my_template.j2
mkdir -p tools/templates
touch tools/templates/my_template.j2
```

```jinja2
{# tools/templates/my_template.j2 #}
分析这张图片并输出 JSON:

```json
{
  "category": "...",
  "confidence": 0.95,
  "description": "..."
}
```
```

### 4. 运行标注

```python
from llm_annotator import VaultAnnotator
from tools.my_tool import MyTool

tool = MyTool(model_name="qwen2-vl-7b")
annotator = VaultAnnotator(
    tool=tool,
    vault_path="/path/to/vault",
    output_path="/path/to/output.duckdb",
    batch_size=32,
    max_workers=64,
)
annotator.run()
```

## 关键实现细节

### AnnotateTool 基类要求

**必须重写的方法:**
1. `__init__(model_name)` - 初始化工具、加载提示词、创建客户端
2. `_prepare_kwargs_and_participants(sample, storager)` - 准备数据和参与者
3. `__call__(**kwargs)` - 执行 VLM API 调用

**关键属性:**
- `basic_name`: 工具唯一标识 (用于数据库查询)
- `sample_type`: `"image"`, `"text"`, 或 `"sequence"`
- `creator_meta`: 存储在标注中的元数据 (通常包含提示词)

### 数据准备模式

#### 单图像分析 (最常用)
```python
def _prepare_kwargs_and_participants(self, sample, storager):
    image_bytes = storager.get_image_bytes_by_ids([sample["sample_id"]])
    participants = ((sample["sample_id"], MultiModalType.IMAGE, "image"),)
    kwargs = {"image": image_bytes[sample["sample_id"]], "text": self.prompt}
    return participants, kwargs
```

#### 双图像对比
```python
def _prepare_kwargs_and_participants(self, sample, storager):
    # 假设 sample 包含两个图像 ID
    source_id, target_id = sample["source_id"], sample["target_id"]
    image_bytes = storager.get_image_bytes_by_ids([source_id, target_id])

    participants = (
        (source_id, MultiModalType.IMAGE, "source"),
        (target_id, MultiModalType.IMAGE, "target"),
    )

    kwargs = {
        "source_image": image_bytes[source_id],
        "target_image": image_bytes[target_id],
        "text": self.prompt,
    }
    return participants, kwargs

def __call__(self, source_image, target_image, text, **kwargs):
    from llm_annotator.utils import call_vlm_compare
    return call_vlm_compare(
        source_image=source_image,
        target_image=target_image,
        text=text,
        model_name=self.model_name,
        client=self.client,
    )
```

#### 序列级分析
```python
def _prepare_kwargs_and_participants(self, sample, storager):
    sequence_id = sample["sample_id"]
    sequence_meta = storager.get_sequence_metas([sequence_id])[0]

    # 获取序列中所有图像
    image_ids = [img.id for img in sequence_meta.images]
    image_bytes_dict = storager.get_image_bytes_by_ids(image_ids)

    # 所有图像参与，带编号角色
    participants = tuple(
        (img_id, MultiModalType.IMAGE, f"frame_{i}")
        for i, img_id in enumerate(image_ids)
    )

    kwargs = {
        "images": [image_bytes_dict[img_id] for img_id in image_ids],
        "text": self.prompt,
    }
    return participants, kwargs
```

### 提示词模板进阶用法

#### 动态变量渲染
```python
# 在 __init__ 中
from llm_annotator.prompt_loader import render_prompt

self.prompt = render_prompt("my_template.j2", {
    "categories": ["cat", "dog", "bird"],
    "format": "JSON",
    "instruction": "Focus on colors and composition"
}, template_dir=template_dir)
```

```jinja2
{# my_template.j2 #}
分析图片类别: {{ categories|join(', ') }}

{% if instruction %}
{{ instruction }}
{% endif %}

输出格式: {{ format }}
```

### 核心工具函数

在 `llm_annotator/utils.py` 中:

- `call_vlm_single()` - 单图像 VLM 调用
- `call_vlm_compare()` - 双图像对比 VLM 调用
- `execute_with_retry()` - 自动重试机制
- `extract_and_validate_json()` - JSON 提取和验证
- `image_to_base64()` - 图像编码转换

### VaultAnnotator 参数调优

```python
annotator = VaultAnnotator(
    tool=tool,
    vault_path="/path/to/vault",      # 输入 vault 路径
    output_path="/path/to/output.duckdb",  # 输出数据库
    batch_size=32,                    # 批量保存大小 (32-128)
    max_workers=64,                   # 并发线程数 (I/O 任务可设置较高)
    retry_count=3,                    # API 失败重试次数
)
```

**调优建议:**
- **CPU 密集型**: `max_workers` = CPU 核心数
- **I/O 密集型 (VLM 调用)**: `max_workers` = 64-512
- **大批量**: `batch_size` = 64-128 (减少写入开销)
- **小批量实验**: `batch_size` = 16-32 (更频繁进度更新)

## 最佳实践

1. **工具命名**: 使用描述性的 `basic_name` (如 `aesthetic_scorer`)
2. **角色语义**: 为参与者使用清晰的 role (如 `source_image`, `target_image`)
3. **提示词版本控制**: 用 git 跟踪模板变化
4. **错误处理**: 利用内置重试机制，避免手动 try-catch
5. **测试先行**: 先用小数据集 (1-10 样本) 测试
6. **JSON 格式**: 保持输出格式一致性，便于后续处理
7. **监控资源**: 大任务时注意内存和 API 限流

## 现有工具参考

- `stepflow_tag_tool.py` - 单图像场景标签 (模板)
- `stepflow_compare_tool.py` - 双图像对比分析

复制现有工具，修改3个核心方法，即可快速创建新的标注工具！