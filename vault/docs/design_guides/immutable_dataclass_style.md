## 数据类核心设计原则

### 设计哲学

我们的核心数据类设计旨在实现最高级别的代码清晰度与安全性。
其基石思想是严格分离一个对象的两种生命周期来源：**重建 (Reconstruction)** 与 **创建 (Creation)**。

-----

### 三大核心原则

#### 1\. 绝对的不可变性 (Absolute Immutability)

  * **做什么 (What):** 始终使用 `@dataclass(frozen=True)`。
  * **为什么 (Why):** 对象一旦创建即被“冻结”，杜绝意外修改，保证了数据的完整性、线程安全和行为的可预测性。

#### 2\. 严格分离“重建”与“创建” (Strict Separation)

  * **做什么 (What):**

      * **重建**：使用默认构造函数 `Tag(...)`。它唯一的职责是接收已存在的、完整的数据，不包含任何计算逻辑。
      * **创建**：使用 `@classmethod` 工厂方法，如 `Tag.create(...)`。它负责处理所有业务逻辑、数据校验、哈希计算和默认值设定。

  * **为什么 (Why):** 这使得代码的意图一目了然。看到 `Tag(...)` 就知道是在从数据源恢复对象；看到 `Tag.create(...)` 就知道是在生成一个全新的实体。

#### 3\. 职责清晰的工厂链 (Chain of Responsibility)

  * **做什么 (What):** 提供一个**主工厂** (`create`) 处理所有核心的、通用的计算逻辑。提供多个**快捷工厂** (`ai`, `user` 等) 来封装特定的业务场景，并由它们调用主工厂来完成最终创建。
  * **为什么 (Why):** 遵循 DRY (Don't Repeat Yourself) 原则，避免逻辑重复。当需要扩展新类型的标签时，只需添加一个新的快捷工厂即可，无需改动核心逻辑，使得系统扩展变得极为简单。

-----

### 最终设计范例：`Tag` 类

```python
from dataclasses import dataclass, field
from typing import Any

# (外部辅助函数，保持 Tag 类内部纯粹)
def jsonify_meta(meta: Any) -> str | None:
    # ... 实现细节 ...
    pass

def object_xxhash(text, type, source, json_meta) -> int:
    # ... 实现细节 ...
    pass


# 原则一：对象始终不可变
@dataclass(frozen=True)
class Tag:
    """
    一个不可变的数据对象，严格遵循分离设计原则。
    - 重建: Tag(id=1, text='..', json_meta='..')
    - 创建: Tag.create(text='..')
    - 快捷创建: Tag.ai(model='..')
    """
    # 原则二：“重建”入口的字段定义。
    # __init__ 只负责接收这些数据，不含任何逻辑。
    id: int
    text: str
    type: str | None
    source: str | None
    meta: Any
    json_meta: str | None = field(repr=False)

    @classmethod
    def create(cls, text: str, type: str | None, source: str | None, meta: Any):
        """原则二：“创建”的主入口，封装所有计算逻辑。"""
        # 1. 计算所有派生数据
        json_meta = jsonify_meta(meta)
        calculated_id = object_xxhash(text, type, source, json_meta)

        # 2. 调用构造函数完成“重建”
        return cls(
            id=calculated_id,
            text=text,
            type=type,
            source=source,
            meta=meta,
            json_meta=json_meta,
        )

    @classmethod
    def ai(cls, model: str, source: str | None, meta: Any):
        """原则三：快捷工厂，封装业务场景并调用主工厂。"""
        return cls.create(text=model, type="generated_by", source=source, meta=meta)

```

**总结:** 该设计模式确保了我们的数据对象不仅是**不可变的**，而且其**来源和意图**在代码中也是**清晰可辨的**，极大地提升了代码的可读性和长期可维护性。