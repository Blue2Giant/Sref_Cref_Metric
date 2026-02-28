"""
快速测试：样本级标注功能验证

本脚本创建一个小型测试 Vault 并添加各种类型的标注，用于验证功能。
"""

import tempfile

from loguru import logger

from vault.schema.multimodal import Image, PackSequence, Text
from vault.storage.lanceduck.multimodal import MultiModalStorager
from vault.utils.image import create_text_image


def create_test_vault(vault_path: str | None = None) -> str:
    """
    创建一个包含测试数据的 Vault

    Returns:
        vault_path: 创建的 Vault 路径
    """
    if vault_path is None:
        temp_dir = tempfile.mkdtemp(prefix="vault_test_")
        vault_path = temp_dir

    logger.info(f"创建测试 Vault: {vault_path}")

    # 初始化 Vault
    MultiModalStorager.init(vault_path)
    storager = MultiModalStorager(vault_path)

    # 创建测试数据：10 个图文对
    sequences = []
    for i in range(10):
        # 创建测试图像
        pil_image = create_text_image(f"Test Image {i}")
        image = Image.create(
            pil_image,
            uri=f"test/image_{i}.jpg",
            source="test_dataset",
        )

        # 创建测试文本
        text = Text.create(
            content=f"This is test caption {i}",
            uri=f"test/caption_{i}.txt",
            source="test_dataset",
        )

        # 创建序列
        sequence = PackSequence.from_text_to_image(
            caption=text,
            image=image,
            source="test_dataset",
            uri=f"test/sequence_{i}",
        )
        sequences.append(sequence)

    # 写入数据
    storager.add_sequences(sequences)
    storager.commit()

    logger.info(f"✅ 测试 Vault 创建完成，包含 {len(sequences)} 个序列")
    logger.info(f"路径: {vault_path}")

    return vault_path


def test_all_annotation_scenarios(vault_path: str):
    """
    测试所有标注场景

    Args:
        vault_path: Vault 路径
    """
    logger.info("\n" + "=" * 60)
    logger.info("开始测试所有标注场景")
    logger.info("=" * 60)

    # 导入场景函数
    from add_sample_annotations_tutorial import (
        add_aesthetic_scores_simple,
        add_clip_scores,
        query_annotations,
    )

    # 场景 1: 美学评分
    logger.info("\n>>> 测试场景 1: 美学评分")
    add_aesthetic_scores_simple(vault_path=vault_path)

    # 场景 2: CLIP 评分
    logger.info("\n>>> 测试场景 2: CLIP 评分")
    add_clip_scores(vault_path=vault_path)

    # 查询验证
    logger.info("\n>>> 查询所有标注")
    query_annotations(vault_path=vault_path)

    logger.info("\n" + "=" * 60)
    logger.info("✅ 所有测试完成！")
    logger.info("=" * 60)


def quick_test():
    """
    快速测试流程：创建 Vault -> 添加标注 -> 验证

    使用方式：
        python test_sample_annotations.py
    """
    logger.info("🚀 开始快速测试...")

    # 1. 创建测试 Vault
    vault_path = create_test_vault()

    # 2. 运行所有测试
    test_all_annotation_scenarios(vault_path)

    # 3. 打印查询示例
    logger.info("\n" + "=" * 60)
    logger.info("📝 后续操作建议")
    logger.info("=" * 60)

    print(f"""
测试 Vault 位置: {vault_path}

你可以运行以下命令继续测试：

1. 查看标注统计：
   python add_sample_annotations_tutorial.py query --vault_path={vault_path}

2. 导出标注到 CSV：
   python add_sample_annotations_tutorial.py export \\
       --vault_path={vault_path} \\
       --annotation_name=aesthetic_score \\
       --output_path=/tmp/test_scores.csv

3. 直接查询 DuckDB：
   python -c "
from vault.storage.lanceduck.multimodal import MultiModalStorager
s = MultiModalStorager('{vault_path}', read_only=True)
with s.meta_handler as h:
    results = h.query_batch('''
        SELECT sa.name, COUNT(*) as count
        FROM sample_annotations sa
        GROUP BY sa.name
    ''')
    for r in results:
        print(f'{{r[\"name\"]}}: {{r[\"count\"]}} 条标注')
"

4. 测试完成后删除测试数据：
   rm -rf {vault_path}
    """)


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "quick_test": quick_test,
            "create_vault": create_test_vault,
            "test_annotations": test_all_annotation_scenarios,
        }
    )
