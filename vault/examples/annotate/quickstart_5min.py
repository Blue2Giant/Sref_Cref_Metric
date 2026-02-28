"""
样本级标注 - 5 分钟快速上手

这是一个极简版教程，让你在 5 分钟内完成第一个标注任务。
详细文档请查看 README.md
"""

# ============================================================
# 步骤 1: 准备工作（30 秒）
# ============================================================

from vault.schema import ID
from vault.schema.multimodal import Creator, MultiModalType, SampleAnnotation
from vault.storage.lanceduck.multimodal import MultiModalStorager
from vault.backend.duckdb import DuckDBHandler

# 设置你的 Vault 路径
VAULT_PATH = "/path/to/your/vault"  # 修改这里！

# ============================================================
# 步骤 2: 查询需要标注的数据（1 分钟）
# ============================================================

storager = MultiModalStorager(VAULT_PATH, read_only=True)

# 方式 A: 查询所有图像（最简单）
with storager.meta_handler as handler:
    items = handler.query_batch(
        """
        SELECT s.id as sequence_id, si.image_id
        FROM sequences s
        JOIN sequence_images si ON s.id = si.sequence_id
        LIMIT 100  -- 先处理 100 个试试
        """
    )

# 方式 B: 只查询特定来源的数据
# with storager.meta_handler as handler:
#     items = handler.query_batch(
#         """
#         SELECT s.id as sequence_id, si.image_id
#         FROM sequences s
#         JOIN sequence_images si ON s.id = si.sequence_id
#         WHERE s.source = ?
#         LIMIT 100
#         """,
#         ["your_dataset_name"]  # 修改这里！
#     )

print(f"找到 {len(items)} 个需要标注的样本")

# ============================================================
# 步骤 3: 创建标注（2 分钟）
# ============================================================

# 创建标注创建者（描述标注来源）
creator = Creator.create(
    name="my_scorer_v1",  # 修改为你的标注器名称
    meta={
        "model": "your-model-name",  # 可选：模型信息
        "version": "1.0",
    },
)

# 批量创建标注
annotations = []

for item in items:
    sequence_id = ID.from_(item["sequence_id"])
    image_id = ID.from_(item["image_id"])

    # 🔥 在这里添加你的标注逻辑 🔥
    # 示例：调用你的模型
    # score = your_model.predict(image_id)

    # 这里用固定值演示（请替换为实际逻辑）
    score = 7.5

    # 创建标注对象
    annotation = SampleAnnotation.create(
        name="my_score",  # 修改为你的标注名称，如 "aesthetic_score"
        sequence_id=sequence_id,
        creator=creator,
        value=score,  # 你的标注值
        participants=(  # 这个标注涉及哪些元素
            (image_id, MultiModalType.IMAGE, "target"),
        ),
    )
    annotations.append(annotation)

print(f"创建了 {len(annotations)} 条标注")

# ============================================================
# 步骤 4: 写入 Vault（1.5 分钟）
# ============================================================

# 4.1 创建临时数据库
temp_db_path = "/tmp/my_annotations.duckdb"

storager_rw = MultiModalStorager(VAULT_PATH, read_only=False)
temp_handler = DuckDBHandler(
    schema=storager_rw.DUCKDB_SCHEMA, read_only=False, db_path=temp_db_path
)
temp_handler.create()

print(f"临时数据库创建于: {temp_db_path}")

# 4.2 写入标注到临时数据库
storager_rw.add_sample_annotations(annotations, duckdb_handler=temp_handler)
print("✅ 标注已写入临时数据库")

# 4.3 合并到主 Vault
storager_rw.merge(duckdb_files=[temp_db_path])
print("✅ 标注已合并到主 Vault")

# ============================================================
# 步骤 5: 验证结果（30 秒）
# ============================================================

# 查询刚才添加的标注
with storager.meta_handler as handler:
    count = handler.query_batch(
        "SELECT COUNT(*) as count FROM sample_annotations WHERE name = ?",
        ["my_score"],  # 改为你的标注名称
    )[0]["count"]

    print(f"\n🎉 成功！Vault 中现在有 {count} 条 'my_score' 标注")

    # 查看一些示例
    samples = handler.query_batch(
        """
        SELECT sa.sequence_id, sa.value_float
        FROM sample_annotations sa
        WHERE sa.name = ?
        LIMIT 5
        """,
        ["my_score"],
    )

    print("\n前 5 条标注示例：")
    for s in samples:
        print(f"  Sequence: {s['sequence_id']}, Score: {s['value_float']}")

# ============================================================
# 🎓 恭喜！你已经完成了第一个标注任务
# ============================================================

print("\n" + "=" * 60)
print("下一步建议：")
print("=" * 60)
print("""
1. 查看完整教程了解更多场景：
   examples/annotate/README.md

2. 学习如何查询和导出标注：
   python add_sample_annotations_tutorial.py query --vault_path={vault}
   python add_sample_annotations_tutorial.py export --vault_path={vault}

3. 了解如何从 CSV/Parquet 批量导入：
   查看 add_sample_annotations_tutorial.py 的场景 4

4. 学习分布式并行标注：
   查看 add_sample_annotations_tutorial.py 的场景 5
""".format(vault=VAULT_PATH))

# ============================================================
# 常见问题速查
# ============================================================

"""
Q: 如何存储复杂的标注结果（如 JSON 对象）？
A: 直接传 dict/list 给 value 参数：
   value={"quality": 8.5, "metrics": {...}}

Q: 如何标注图文对（需要同时引用图像和文本）？
A: 在 participants 中添加多个元素：
   participants=(
       (image_id, MultiModalType.IMAGE, "image"),
       (text_id, MultiModalType.TEXT, "caption"),
   )

Q: 如何避免重复标注？
A: 在查询时排除已标注的数据：
   WHERE NOT EXISTS (
       SELECT 1 FROM sample_annotations sa
       WHERE sa.sequence_id = s.id AND sa.name = 'my_score'
   )

Q: 如何更新已有的标注？
A: 重新写入相同 name + creator + participants 的标注会覆盖旧值

Q: 临时数据库文件需要删除吗？
A: 不需要，可以保留作为备份，或手动删除：rm /tmp/my_annotations.duckdb

更多问题请查看 README.md 的 FAQ 部分
"""
