"""
为手工挑选的图像对添加统一标注

本示例展示如何为一批手工挑选的图像添加统一的 Annotation 标注。
与 SampleAnnotation（每样本独有值）不同，Annotation 是共享标注，
适用于标记一组图像的共同属性（如：手工筛选标记、数据集版本、质量标签等）。

使用场景：
- 标记手工筛选的高质量样本
- 标记用于特定实验的数据子集
- 添加数据集版本或批次标签

============================================================
"""

import fire
from loguru import logger
from tqdm import tqdm

from vault.schema import ID
from vault.schema.multimodal import Annotation, Creator
from vault.storage.lanceduck.multimodal import MultiModalStorager

# 80个需要标注的序列ID（来自手工挑选）
SEQUENCE_IDS_TO_PROCESS = """
26d3ff8c771b354ce8b52d21143065db
    006574f7afc68803abb85282ad8d647d
    0cfb85d4dc0b22b3cd6996dd97813452
    0b4229aac0eb4a4b6c8325088cea79e0
    176e782d90e627bd891f25b9b15ee420
    07a2d957691d20afc85e43675354f100
    071d7bca097d2c54e8703bf64f915146
    19bbd34f16e24f61a768c0b93869f75a
    0a27a1ec187c3c05eda03efa44d8bc8b
    1f489f44f4e9bf7e2c1c22ccc1bf1b8a
    1019782e8489a62e14bf5f8733f96f1a
    0fc3a2f55c133a6b46c32b9863eb436f
    262c4d2124a9711ed54f079947f3f5b6
    16a637d8c9774a7eca849be084c45716
    2912a7d535a5c981f999d83c9796d4e4
    200bf0bcaa7bef6cabaad34ff4205168
    121b71bea4f3ef5f7be553891131c006
    278bf3e89a8c2a2177219cc338e35916
    061ba65470fe94154fe6c268c9322a55
    27aecf9767305e95e5f3527cf22bfce4
    14df54c729383da928ed1af2390af608
    18be3c5087b6f91ee7ee567336b40bc0
    1cd9313ba367c23f70795e73bd3007f1
    053e27d4f84f20f7bb988471c8ae5b0a
    0b6a90f100f08c4ef2c83a4ed1b109a4
    0a079971dcf6b3be675bf8e62249c6f4
    19b1c01535e3ce8101a9ef9f283ec465
    15ee77b246b57806f6dfa4b58deba2a5
    0e35cbdb8d0dfafaa80909d9cb10b42f
    08389ff644e23ea8b4cdedb818cdad3b
    1ead630019ec7482df0c5232b8fc82ff
    2912a7d535a5c981f999d83c9796d4e4
    03f4df43bc2224371c61f5c682838d23
    017768e265dab1ffd45f3a47f7fc7109
    1a15f6f14903b00bc09185535d1f96ef
    1eff1ea700b88e8fa8294e0c58d1f54d
    0834472d8570cfdae12017c164e2aaea
    0bbcc8aeccd074f16c723e33a1d8184f
    0323556dda60856a61f7a8011fad2d7c
    11be7ebce0c0b44a68f083727351db73
    0376bf072465a70fd06aa1877ce58446
    05da2388b3dc224a5db13d6b0536b58c
    19e93c01232d6b84e5c3d7162b593dd5
    202d140b6a174d5f9f17522fd688e273
    09c457e8e27a7e4d70fb536cf0032df8
    1f7d40790eccec9e606968d51533a86a
    1ca5045aea377893dcf49fac91b67f60
    0ea0456ce894e1547ebb88a356fa49d3
    00cc0e7bd2a608cbb9e7c9d368315a2b
    0a51863a694cc755c9f5935b77a22e78
    1b0f188f809727c20641472612133830
    0780c428b0c66ec3790f9928748393e8
    158d05265fbad1589586ced1c8b1e4a2
    070f7f46c3be272ec70b5bba629b699e
    1ec004b85be37cc8a917c092ac55cb82
    207450cc18bd1f72472d4712f00dc8c2
    1478a94870013826c2d19ce7a1e7fa80
    121277f43a0d4def71a13dbb6d39a547
    19bbd34f16e24f61a768c0b93869f75a
    07665bfd9bb9dbff0c09612461f2d644
    0d1e710ef87a0e0ed36d72018c5390ca
    1654bd46b41deb4989c7ab5ae2e77823
    1ce3eb152fa82abf236186d52956be21
    1dfdb68b2550dd146ebdab24a2c43a05
    0c2f193c19380cabfbcb708daeb4e59c
    05c9b97c4372cad830dd61e70de34190
    28b2003af9f410dbc8fa055cfc5852f7
    0d8507a19f1446500ab2417ea6a6ccb6
    19d60a77d208eb4fe8d84e6b5a5be2bf
    02d30c315777b693f2b4ef61b489a841
    1c1da744a091108b2f44fe2db395a6bf
    1c74e571449db67a6c0e43229c0adf1a
    09d6df0020b70f4233ae140c4d0df2d7
    010989e1b7c6573b9406f1d69a0644e7
    163e0039eb2a6bd4eba21044a4ae641e
    0418775e566b54e2929190d5db2b0db4
    1023e3a3ab2c4edb7fd54501cb615e65
    0f4d0b135503926a5736ef4a87c1b4de
    278bf3e89a8c2a2177219cc338e35916
    033f756b223f00938e21b6c6fc87194e
    254372d04595d9d3ae673d357a2e70a2
    271d8d3624f6731cb0ef3a5dc291110c
    217e0d9604135d8ad39965612ede1ac8
"""


def extract_source_target_image_ids(sample: dict) -> tuple[ID, ID] | None:
    """从序列元数据中提取 source 和 target 图片ID"""
    source_image_id = None
    target_image_id = None

    for img in sample["images"]:
        if "prev" in img["index"] or "source" in img["index"]:
            source_image_id = ID.from_(img["id"])
        if "current" in img["index"] or "target" in img["index"]:
            target_image_id = ID.from_(img["id"])

    if source_image_id is None or target_image_id is None:
        logger.warning(
            f"Cannot find source/target images in sequence: {sample.get('sequence_id', 'unknown')}"
        )
        return None

    return source_image_id, target_image_id


def add_manual_selection_annotation(
    vault_path: str = "/mnt/marmot/liaojie/ScreenMusings-251022",
    annotation_name: str = "manual_selection",
    annotation_type: str = "quality_control",
    creator_name: str = "wangrui",
    selection_date: str = "2025-11-20",
    selection_note: str = "手工挑选的高质量图像对样本",
    dry_run: bool = False,
):
    """
    为手工挑选的80个序列的 source 和 target 图片添加统一标注

    Args:
        vault_path: Vault 存储路径
        annotation_name: 标注名称
        annotation_type: 标注类型
        creator_name: 创建者名称
        selection_date: 筛选日期
        selection_note: 筛选说明
        dry_run: 如果为 True，只打印信息不实际写入
    """
    logger.info("=== 添加手工挑选标注 ===")
    logger.info(f"Vault路径: {vault_path}")
    logger.info(f"标注名称: {annotation_name}")
    logger.info(f"创建者: {creator_name}")
    logger.info(f"筛选日期: {selection_date}")

    # 1. 解析序列ID（去重）
    target_sequence_ids = list(
        set(
            ID.from_(seq_id.strip())
            for seq_id in SEQUENCE_IDS_TO_PROCESS.strip().split("\n")
            if seq_id.strip()
        )
    )
    logger.info(f"待处理序列数量: {len(target_sequence_ids)}")

    # 2. 初始化 Storager（读写模式，全程使用同一个实例避免连接冲突）
    storager = MultiModalStorager(vault_path, read_only=False)

    # 3. 获取序列元数据
    logger.info("获取序列元数据...")
    sequences = storager.get_sequence_metas(target_sequence_ids)
    logger.info(f"成功获取 {len(sequences)} 个序列的元数据")

    # 4. 提取所有 source 和 target 图片ID
    image_ids_to_annotate: list[ID] = []
    for sequence in tqdm(sequences, desc="提取图片ID"):
        result = extract_source_target_image_ids(sequence)
        if result is not None:
            source_id, target_id = result
            image_ids_to_annotate.append(source_id)
            image_ids_to_annotate.append(target_id)

    # 去重
    image_ids_to_annotate = list(set(image_ids_to_annotate))
    logger.info(f"共需标注 {len(image_ids_to_annotate)} 张图片")

    # 5. 创建标注对象
    creator = Creator.create(
        name=creator_name,
        meta={
            "selection_date": selection_date,
            "note": selection_note,
        },
    )
    annotation = Annotation.create(
        name=annotation_name,
        type_=annotation_type,
        creator=creator,
        meta={
            "selection_date": selection_date,
            "note": selection_note,
            "total_images": len(image_ids_to_annotate),
            "total_sequences": len(sequences),
        },
    )

    if dry_run:
        logger.info("[DRY RUN] 以下是将要创建的标注信息：")
        logger.info(f"  - 标注名称: {annotation_name}")
        logger.info(f"  - 标注类型: {annotation_type}")
        logger.info(f"  - 创建者: {creator_name}")
        logger.info(f"  - 图片数量: {len(image_ids_to_annotate)}")
        logger.info(
            f"  - 前5个图片ID: {[str(img_id) for img_id in image_ids_to_annotate[:5]]}"
        )
        logger.info(f"  - 创建者对象: {creator}")
        logger.info(f"  - 标注对象: {annotation}")
        return

    logger.info(f"创建标注对象: {annotation}")

    # 6. 添加标注（孤立添加，不绑定元素）
    logger.info("写入标注到数据库...")
    storager.add_annotations(annotations=[annotation])

    # 7. 建立标注与图片的关联
    logger.info("建立标注与图片的关联...")
    association_items = [
        (annotation.id, image_id, "image") for image_id in image_ids_to_annotate
    ]

    storager.associate_annotations(items=association_items)

    logger.info(f"✅ 已建立 {len(association_items)} 条关联")
    logger.info("✅ 标注添加完成！")

    # 8. 验证结果
    logger.info("验证结果...")
    with storager.meta_handler as handler:
        # 查询标注数量
        anno_count = handler.query_batch(
            "SELECT COUNT(*) as count FROM annotations WHERE name = ?",
            [annotation_name],
        )[0]["count"]

        # 查询关联数量
        assoc_count = handler.query_batch(
            """
            SELECT COUNT(*) as count
            FROM image_annotations ia
            JOIN annotations a ON ia.annotation_id = a.id
            WHERE a.name = ?
            """,
            [annotation_name],
        )[0]["count"]

        logger.info("验证结果:")
        logger.info(f"  - 标注记录数: {anno_count}")
        logger.info(f"  - 图片关联数: {assoc_count}")


def query_manual_selection(
    vault_path: str = "/mnt/marmot/liaojie/ScreenMusings-251022",
    annotation_name: str = "manual_selection",
    show_all: bool = False,
):
    """
    查询手工挑选标注的序列信息

    找出 source 和 target 图片都有指定标注的序列。

    Args:
        vault_path: Vault 存储路径
        annotation_name: 标注名称
        show_all: 是否显示所有序列ID（默认只显示前10个）
    """
    logger.info("=== 查询标注序列 ===")

    storager = MultiModalStorager(vault_path, read_only=True)

    with storager.meta_handler as handler:
        # 查询标注信息
        annotations = handler.query_batch(
            """
            SELECT
                a.id,
                a.name,
                a.type,
                a.meta,
                c.name as creator_name,
                c.meta as creator_meta
            FROM annotations a
            LEFT JOIN creators c ON a.creator_id = c.id
            WHERE a.name = ?
            """,
            [annotation_name],
        )

        if not annotations:
            logger.warning(f"未找到名为 '{annotation_name}' 的标注")
            return []

        for anno in annotations:
            logger.info("\n标注信息:")
            logger.info(f"  - ID: {anno['id']}")
            logger.info(f"  - 名称: {anno['name']}")
            logger.info(f"  - 类型: {anno['type']}")
            logger.info(f"  - 创建者: {anno['creator_name']}")
            logger.info(f"  - 元数据: {anno['meta']}")

        # 查询带有指定标注的序列（source 和 target 图片都有标注）
        # 逻辑：找到序列中至少有2张图片带有该标注的序列
        sequences = handler.query_batch(
            """
            SELECT
                s.id as sequence_id,
                s.uri as sequence_uri,
                s.source as sequence_source,
                COUNT(DISTINCT si.image_id) as annotated_image_count
            FROM sequences s
            JOIN sequence_images si ON s.id = si.sequence_id
            JOIN image_annotations ia ON si.image_id = ia.image_id
            JOIN annotations a ON ia.annotation_id = a.id
            WHERE a.name = ?
            GROUP BY s.id, s.uri, s.source
            HAVING COUNT(DISTINCT si.image_id) >= 2
            ORDER BY s.id
            """,
            [annotation_name],
        )

        logger.info(f"\n=== 带有标注的序列（共 {len(sequences)} 个）===")

        if show_all:
            display_sequences = sequences
        else:
            display_sequences = sequences[:10]

        for seq in display_sequences:
            logger.info(
                f"  - {seq['sequence_id']} (图片数: {seq['annotated_image_count']})"
            )

        if not show_all and len(sequences) > 10:
            logger.info(f"  ... 还有 {len(sequences) - 10} 个序列未显示")
            logger.info("  (使用 --show_all=True 查看全部)")

        # 统计总数
        total_images = handler.query_batch(
            """
            SELECT COUNT(*) as count
            FROM image_annotations ia
            JOIN annotations a ON ia.annotation_id = a.id
            WHERE a.name = ?
            """,
            [annotation_name],
        )[0]["count"]

        logger.info("\n=== 统计 ===")
        logger.info(f"  - 总序列数: {len(sequences)}")
        logger.info(f"  - 总图片数: {total_images}")

        # 返回序列ID列表，方便程序调用
        return [seq["sequence_id"] for seq in sequences]


def main():
    """
    使用示例：

    # 预览（不实际写入）
    python add_manual_selection_annotation.py add \\
        --vault_path=/mnt/marmot/liaojie/ScreenMusings-251022 \\
        --dry_run=True

    # 实际添加标注
    python add_manual_selection_annotation.py add \\
        --vault_path=/mnt/marmot/liaojie/ScreenMusings-251022 \\
        --annotation_name=manual_selection \\
        --creator_name=wangrui \\
        --selection_date=2025-11-20

    # 查询标注
    python add_manual_selection_annotation.py query \\
        --vault_path=/mnt/marmot/liaojie/ScreenMusings-251022 \\
        --annotation_name=manual_selection
    """
    fire.Fire(
        {
            "add": add_manual_selection_annotation,
            "query": query_manual_selection,
        }
    )


if __name__ == "__main__":
    main()
