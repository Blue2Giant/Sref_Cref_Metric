"""
样本级标注添加完整教程

本示例展示了如何为已存在的 Vault 添加样本级标注（Sample Annotations）。
样本级标注用于存储每个样本独有的标注值，如：模型评分、人工打分、推理结果等。

============================================================
核心概念
============================================================

1. **SampleAnnotation vs Annotation 的区别**：
   - Annotation（共享标注）：多个样本共享同一标注对象（如 "generated_by:gpt4o"）
   - SampleAnnotation（样本级标注）：每个样本独有的标注值（如每张图的美学评分）

2. **SampleAnnotation 的组成**：
   - name: 标注名称（如 "aesthetic_score"、"clip_score"）
   - creator: 标注创建者（模型名称、人工标注者等）
   - value: 标注值（可以是数字或复杂 JSON）
   - sequence_id: 关联的序列 ID
   - participants: 参与元素列表（image/text + role）

3. **标注流程**：
   Step 1: 查询 Vault，获取需要标注的数据
   Step 2: 执行标注任务（模型推理、人工打分等）
   Step 3: 创建临时 DuckDB 文件并写入标注
   Step 4: 合并到主 Vault

============================================================
使用场景
============================================================

场景 1: 单图像标注（如美学评分）
场景 2: 图文对标注（如 CLIP 相似度）
场景 3: 多元素标注（如图像编辑的三元组：source, instruction, target）
场景 4: 从外部文件批量导入（CSV/Parquet/JSONL）
场景 5: 分布式标注（多进程并行处理）

============================================================
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import fire
import pandas as pd
from loguru import logger
from tqdm import tqdm

from vault.backend.duckdb import DuckDBHandler
from vault.schema import ID
from vault.schema.multimodal import Creator, MultiModalType, SampleAnnotation
from vault.storage.lanceduck.multimodal import MultiModalStorager


# ============================================================
# 场景 1: 单图像标注 - 美学评分
# ============================================================


def add_aesthetic_scores_simple(
    vault_path: str = "/mnt/jfs/datasets/vault/example",
    source_filter: str | None = None,
):
    """
    为图像添加美学评分（简单示例）

    Args:
        vault_path: Vault 路径
        source_filter: 可选，只处理特定来源的数据
    """
    logger.info("场景 1: 添加美学评分（简单模式）")

    # 1. 初始化 Storager（只读模式查询）
    storager = MultiModalStorager(vault_path, read_only=True)

    # 2. 查询需要标注的序列
    with storager.meta_handler as handler:
        if source_filter:
            sequences = handler.query_batch(
                """
                SELECT DISTINCT
                    s.id as sequence_id,
                    si.image_id,
                    i.uri,
                    i.source
                FROM sequences s
                JOIN sequence_images si ON s.id = si.sequence_id
                JOIN images i ON si.image_id = i.id
                WHERE s.source = ?
                """,
                [source_filter],
            )
        else:
            sequences = handler.query_batch(
                """
                SELECT DISTINCT
                    s.id as sequence_id,
                    si.image_id,
                    i.uri,
                    i.source
                FROM sequences s
                JOIN sequence_images si ON s.id = si.sequence_id
                JOIN images i ON si.image_id = i.id
                LIMIT 100  -- 限制数量，避免一次处理太多
                """
            )

    logger.info(f"找到 {len(sequences)} 个需要标注的样本")

    # 3. 创建标注创建者
    creator = Creator.create(
        name="aesthetic_scorer_v1",
        meta={
            "model": "aesthetic-predictor-v2.5",
            "description": "基于 LAION 训练的美学评分模型",
        },
    )

    # 4. 模拟标注过程并创建 SampleAnnotation 对象
    sample_annotations = []

    for item in tqdm(sequences, desc="生成标注"):
        sequence_id = ID.from_(item["sequence_id"])
        image_id = ID.from_(item["image_id"])

        # 这里应该是实际的模型推理
        # score = aesthetic_model.predict(image)
        # 这里用随机数模拟
        import random

        score = random.uniform(4.0, 9.0)

        # 创建样本标注
        sample_annotation = SampleAnnotation.create(
            name="aesthetic_score",  # 标注名称
            sequence_id=sequence_id,  # 关联序列
            creator=creator,  # 创建者
            value=score,  # 标注值（数字会自动存入 value_float）
            participants=(  # 参与元素
                (image_id, MultiModalType.IMAGE, "target"),  # (ID, 类型, 角色)
            ),
        )
        sample_annotations.append(sample_annotation)

    # 5. 创建临时 DuckDB 文件
    temp_dir = Path("/tmp/vault_annotations")
    temp_dir.mkdir(exist_ok=True)
    temp_db_path = temp_dir / "aesthetic_scores.duckdb"

    # 使用 storager 的 schema 初始化临时数据库
    storager_rw = MultiModalStorager(vault_path, read_only=False)
    temp_handler = DuckDBHandler(
        schema=storager_rw.DUCKDB_SCHEMA, read_only=False, db_path=str(temp_db_path)
    )
    temp_handler.create()

    logger.info(f"临时数据库创建于: {temp_db_path}")

    # 6. 写入标注到临时数据库
    storager_rw.add_sample_annotations(sample_annotations, duckdb_handler=temp_handler)

    logger.info(f"✅ 已写入 {len(sample_annotations)} 条标注到临时数据库")

    # 7. 合并到主 Vault
    logger.info("开始合并到主 Vault...")
    storager_rw.merge(duckdb_files=[str(temp_db_path)])

    logger.info("✅ 标注添加完成！")

    # 8. 验证结果
    with storager.meta_handler as handler:
        count = handler.query_batch(
            "SELECT COUNT(*) as count FROM sample_annotations WHERE name = ?",
            ["aesthetic_score"],
        )[0]["count"]
        logger.info(f"验证: 共有 {count} 条 aesthetic_score 标注")


# ============================================================
# 场景 2: 图文对标注 - CLIP 相似度
# ============================================================


def add_clip_scores(
    vault_path: str = "/mnt/jfs/datasets/vault/example",
    source_filter: str | None = None,
):
    """
    为图文对添加 CLIP 相似度评分

    适用场景：文生图数据集，需要评估 caption 和生成图像的匹配度
    """
    logger.info("场景 2: 添加 CLIP 相似度评分")

    storager = MultiModalStorager(vault_path, read_only=True)

    # 查询图文对
    with storager.meta_handler as handler:
        query = """
        SELECT DISTINCT
            s.id as sequence_id,
            si.image_id,
            st.text_id,
            t.content as caption
        FROM sequences s
        JOIN sequence_images si ON s.id = si.sequence_id
        JOIN sequence_texts st ON s.id = st.sequence_id
        JOIN texts t ON st.text_id = t.id
        WHERE st.index = 'caption'  -- 只取 caption 类型的文本
        """

        if source_filter:
            query += " AND s.source = ?"
            params = [source_filter]
        else:
            query += " LIMIT 100"
            params = []

        pairs = handler.query_batch(query, params)

    logger.info(f"找到 {len(pairs)} 个图文对")

    creator = Creator.create(
        name="clip_scorer_v1",
        meta={"model": "openai/clip-vit-large-patch14"},
    )

    sample_annotations = []

    for item in tqdm(pairs, desc="计算 CLIP 相似度"):
        sequence_id = ID.from_(item["sequence_id"])
        image_id = ID.from_(item["image_id"])
        text_id = ID.from_(item["text_id"])

        # 这里应该是实际的 CLIP 计算
        # similarity = clip_model.compute_similarity(image, text)
        import random

        similarity = random.uniform(0.2, 0.95)

        sample_annotation = SampleAnnotation.create(
            name="clip_score",
            sequence_id=sequence_id,
            creator=creator,
            value=similarity,  # 数值类型
            participants=(
                # 多个参与者：图像和文本
                (image_id, MultiModalType.IMAGE, "image"),
                (text_id, MultiModalType.TEXT, "caption"),
            ),
        )
        sample_annotations.append(sample_annotation)

    # 写入流程同场景 1
    temp_db_path = Path("/tmp/vault_annotations/clip_scores.duckdb")
    temp_db_path.parent.mkdir(exist_ok=True)

    storager_rw = MultiModalStorager(vault_path, read_only=False)
    temp_handler = DuckDBHandler(
        schema=storager_rw.DUCKDB_SCHEMA, read_only=False, db_path=str(temp_db_path)
    )
    temp_handler.create()

    storager_rw.add_sample_annotations(sample_annotations, duckdb_handler=temp_handler)
    storager_rw.merge(duckdb_files=[str(temp_db_path)])

    logger.info(f"✅ 已添加 {len(sample_annotations)} 条 CLIP 评分")


# ============================================================
# 场景 3: 多元素标注 - 图像编辑三元组
# ============================================================


def add_edit_quality_scores(
    vault_path: str = "/mnt/jfs/datasets/vault/example",
):
    """
    为图像编辑数据添加质量评分

    适用场景：图像编辑数据集，包含 source_image + instruction + target_image
    """
    logger.info("场景 3: 添加编辑质量评分（多元素标注）")

    storager = MultiModalStorager(vault_path, read_only=True)

    # 查询编辑三元组
    with storager.meta_handler as handler:
        triples = handler.query_batch(
            """
            SELECT
                s.id as sequence_id,
                source_img.image_id as source_image_id,
                target_img.image_id as target_image_id,
                inst.text_id as instruction_id
            FROM sequences s
            JOIN sequence_images source_img ON s.id = source_img.sequence_id
                AND source_img.index = 'source'
            JOIN sequence_images target_img ON s.id = target_img.sequence_id
                AND target_img.index = 'target'
            JOIN sequence_texts inst ON s.id = inst.sequence_id
                AND inst.index = 'instruction'
            WHERE s.source LIKE '%edit%'
            LIMIT 100
            """
        )

    logger.info(f"找到 {len(triples)} 个编辑样本")

    creator = Creator.create(
        name="edit_quality_scorer_v1",
        meta={"model": "edit-quality-predictor", "version": "1.0"},
    )

    sample_annotations = []

    for item in tqdm(triples, desc="评估编辑质量"):
        sequence_id = ID.from_(item["sequence_id"])
        source_image_id = ID.from_(item["source_image_id"])
        target_image_id = ID.from_(item["target_image_id"])
        instruction_id = ID.from_(item["instruction_id"])

        # 模拟质量评估（应该是模型推理）
        import random

        quality_score = random.uniform(0.5, 1.0)

        # 也可以存储复杂的结构化结果
        detailed_result = {
            "overall_quality": quality_score,
            "instruction_following": random.uniform(0.6, 1.0),
            "visual_quality": random.uniform(0.5, 1.0),
            "consistency": random.uniform(0.7, 1.0),
        }

        sample_annotation = SampleAnnotation.create(
            name="edit_quality",
            sequence_id=sequence_id,
            creator=creator,
            value=detailed_result,  # JSON 对象会自动存入 value_json
            participants=(
                # 三个参与者，明确各自角色
                (source_image_id, MultiModalType.IMAGE, "source"),
                (instruction_id, MultiModalType.TEXT, "instruction"),
                (target_image_id, MultiModalType.IMAGE, "target"),
            ),
        )
        sample_annotations.append(sample_annotation)

    # 写入流程
    temp_db_path = Path("/tmp/vault_annotations/edit_quality.duckdb")
    temp_db_path.parent.mkdir(exist_ok=True)

    storager_rw = MultiModalStorager(vault_path, read_only=False)
    temp_handler = DuckDBHandler(
        schema=storager_rw.DUCKDB_SCHEMA, read_only=False, db_path=str(temp_db_path)
    )
    temp_handler.create()

    storager_rw.add_sample_annotations(sample_annotations, duckdb_handler=temp_handler)
    storager_rw.merge(duckdb_files=[str(temp_db_path)])

    logger.info(f"✅ 已添加 {len(sample_annotations)} 条编辑质量评分")


# ============================================================
# 场景 4: 从外部文件批量导入
# ============================================================


@dataclass
class AnnotationImporter:
    """
    从外部文件批量导入标注的工具类

    支持的文件格式：
    - CSV/Parquet: 包含 sequence_id, element_ids, score 等列
    - JSONL: 每行一个标注记录
    """

    vault_path: str
    annotation_name: str
    creator_name: str
    creator_meta: dict | None = None

    def from_csv(
        self,
        csv_path: str,
        sequence_id_col: str = "sequence_id",
        element_id_cols: list[str] = None,
        value_col: str = "score",
    ):
        """
        从 CSV 文件导入标注

        CSV 格式示例：
        sequence_id,image_id,score
        uuid1,uuid2,7.5
        uuid3,uuid4,8.2
        """
        df = pd.read_csv(csv_path)
        logger.info(f"从 CSV 读取 {len(df)} 条记录")

        return self._import_from_dataframe(
            df, sequence_id_col, element_id_cols or ["image_id"], value_col
        )

    def from_parquet(
        self,
        parquet_path: str,
        sequence_id_col: str = "sequence_id",
        element_id_cols: list[str] = None,
        value_col: str = "score",
    ):
        """
        从 Parquet 文件导入标注

        适合大规模数据（如模型批量推理结果）
        """
        df = pd.read_parquet(parquet_path)
        logger.info(f"从 Parquet 读取 {len(df)} 条记录")

        return self._import_from_dataframe(
            df, sequence_id_col, element_id_cols or ["image_id"], value_col
        )

    def from_jsonl(
        self,
        jsonl_path: str,
        sequence_id_key: str = "sequence_id",
        element_ids_key: str = "element_ids",
        value_key: str = "value",
    ):
        """
        从 JSONL 文件导入标注

        JSONL 格式示例：
        {"sequence_id": "uuid1", "element_ids": ["uuid2"], "value": 7.5}
        {"sequence_id": "uuid3", "element_ids": ["uuid4", "uuid5"], "value": {"score": 8.2}}
        """
        records = []
        with open(jsonl_path, "r") as f:
            for line in f:
                records.append(json.loads(line))

        logger.info(f"从 JSONL 读取 {len(records)} 条记录")

        creator = Creator.create(name=self.creator_name, meta=self.creator_meta)
        sample_annotations = []

        for record in tqdm(records, desc="解析 JSONL"):
            try:
                sequence_id = ID.from_(record[sequence_id_key])
                element_ids = [ID.from_(eid) for eid in record[element_ids_key]]
                value = record[value_key]

                # 默认所有元素都是 image 类型，role 为 target
                participants = tuple(
                    (eid, MultiModalType.IMAGE, f"element_{i}")
                    for i, eid in enumerate(element_ids)
                )

                sample_annotation = SampleAnnotation.create(
                    name=self.annotation_name,
                    sequence_id=sequence_id,
                    creator=creator,
                    value=value,
                    participants=participants,
                )
                sample_annotations.append(sample_annotation)
            except Exception as e:
                logger.warning(f"跳过无效记录: {e}")
                continue

        return self._write_annotations(sample_annotations)

    def _import_from_dataframe(
        self,
        df: pd.DataFrame,
        sequence_id_col: str,
        element_id_cols: list[str],
        value_col: str,
    ):
        """从 DataFrame 导入标注的通用逻辑"""
        creator = Creator.create(name=self.creator_name, meta=self.creator_meta)
        sample_annotations = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="处理记录"):
            try:
                sequence_id = ID.from_(row[sequence_id_col])

                # 收集所有元素 ID
                participants = []
                for i, col in enumerate(element_id_cols):
                    element_id = ID.from_(row[col])
                    # 默认类型为 IMAGE，角色根据列名推断
                    role = col.replace("_id", "")  # image_id -> image
                    participants.append((element_id, MultiModalType.IMAGE, role))

                value = row[value_col]
                # 如果 value 是字符串且看起来像 JSON，尝试解析
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except:
                        pass

                sample_annotation = SampleAnnotation.create(
                    name=self.annotation_name,
                    sequence_id=sequence_id,
                    creator=creator,
                    value=value,
                    participants=tuple(participants),
                )
                sample_annotations.append(sample_annotation)
            except Exception as e:
                logger.warning(f"跳过无效记录: {e}")
                continue

        return self._write_annotations(sample_annotations)

    def _write_annotations(self, sample_annotations: list[SampleAnnotation]):
        """写入标注的通用逻辑"""
        if not sample_annotations:
            logger.warning("没有有效的标注数据")
            return

        # 创建临时数据库
        temp_db_path = (
            Path("/tmp/vault_annotations") / f"{self.annotation_name}_import.duckdb"
        )
        temp_db_path.parent.mkdir(exist_ok=True)

        storager = MultiModalStorager(self.vault_path, read_only=False)
        temp_handler = DuckDBHandler(
            schema=storager.DUCKDB_SCHEMA,
            read_only=False,
            db_path=str(temp_db_path),
        )
        temp_handler.create()

        logger.info(f"写入 {len(sample_annotations)} 条标注到临时数据库...")
        storager.add_sample_annotations(sample_annotations, duckdb_handler=temp_handler)

        logger.info("合并到主 Vault...")
        storager.merge(duckdb_files=[str(temp_db_path)])

        logger.info(f"✅ 成功导入 {len(sample_annotations)} 条标注")
        return len(sample_annotations)


def import_from_file_example(
    vault_path: str = "/mnt/jfs/datasets/vault/example",
    file_path: str = "/path/to/scores.csv",
    file_type: Literal["csv", "parquet", "jsonl"] = "csv",
):
    """
    场景 4：从外部文件批量导入标注

    Args:
        vault_path: Vault 路径
        file_path: 标注文件路径
        file_type: 文件类型
    """
    logger.info(f"场景 4: 从 {file_type.upper()} 文件批量导入标注")

    importer = AnnotationImporter(
        vault_path=vault_path,
        annotation_name="external_scores",
        creator_name="external_annotator",
        creator_meta={"source": "human_annotation", "batch": "2024Q1"},
    )

    if file_type == "csv":
        importer.from_csv(file_path)
    elif file_type == "parquet":
        importer.from_parquet(file_path)
    elif file_type == "jsonl":
        importer.from_jsonl(file_path)


# ============================================================
# 场景 5: 分布式标注（多进程处理）
# ============================================================


def distributed_annotation_example(
    vault_path: str = "/mnt/jfs/datasets/vault/example",
    num_workers: int = 8,
):
    """
    场景 5：分布式标注处理

    适用场景：大规模数据集，需要并行处理
    """
    logger.info(f"场景 5: 分布式标注（{num_workers} 个工作进程）")

    from vault.utils.ingest import run_concurrently

    storager = MultiModalStorager(vault_path, read_only=True)

    # 查询所有需要标注的序列
    with storager.meta_handler as handler:
        all_sequences = handler.query_batch(
            """
            SELECT s.id as sequence_id, si.image_id
            FROM sequences s
            JOIN sequence_images si ON s.id = si.sequence_id
            """
        )

    logger.info(f"共 {len(all_sequences)} 个样本需要标注")

    # 分块
    chunk_size = 1000
    chunks = [
        all_sequences[i : i + chunk_size]
        for i in range(0, len(all_sequences), chunk_size)
    ]

    logger.info(f"分为 {len(chunks)} 个任务块")

    def worker_process(chunk: list, worker_id: int):
        """每个工作进程执行的函数"""
        # 每个进程创建自己的临时数据库
        temp_db_path = (
            Path("/tmp/vault_annotations") / f"worker_{worker_id}_annotations.duckdb"
        )
        temp_db_path.parent.mkdir(exist_ok=True)

        storager_worker = MultiModalStorager(vault_path, read_only=False)
        temp_handler = DuckDBHandler(
            schema=storager_worker.DUCKDB_SCHEMA,
            read_only=False,
            db_path=str(temp_db_path),
        )
        temp_handler.create()

        creator = Creator.create(
            name=f"distributed_scorer_worker_{worker_id}",
            meta={"worker_id": worker_id},
        )

        sample_annotations = []
        for item in chunk:
            sequence_id = ID.from_(item["sequence_id"])
            image_id = ID.from_(item["image_id"])

            # 模拟标注
            import random

            score = random.uniform(0, 10)

            sample_annotation = SampleAnnotation.create(
                name="distributed_score",
                sequence_id=sequence_id,
                creator=creator,
                value=score,
                participants=((image_id, MultiModalType.IMAGE, "target"),),
            )
            sample_annotations.append(sample_annotation)

        # 写入工作进程的临时数据库
        storager_worker.add_sample_annotations(
            sample_annotations, duckdb_handler=temp_handler
        )
        logger.info(f"Worker {worker_id} 完成 {len(sample_annotations)} 条标注")

    # 并行执行
    run_concurrently(worker_process, chunks, num_workers)

    # 收集所有工作进程的临时数据库
    temp_dir = Path("/tmp/vault_annotations")
    worker_dbs = list(temp_dir.glob("worker_*_annotations.duckdb"))

    logger.info(f"收集到 {len(worker_dbs)} 个工作进程的数据库")

    # 合并所有临时数据库
    storager_final = MultiModalStorager(vault_path, read_only=False)
    storager_final.merge(duckdb_files=[str(db) for db in worker_dbs])

    logger.info(f"✅ 分布式标注完成，共处理 {len(all_sequences)} 个样本")


# ============================================================
# 工具函数：查询和验证标注
# ============================================================


def query_annotations(
    vault_path: str = "/mnt/jfs/datasets/vault/example",
    annotation_name: str | None = None,
):
    """
    查询并展示标注统计信息

    Args:
        vault_path: Vault 路径
        annotation_name: 可选，只查询特定名称的标注
    """
    logger.info("查询标注统计信息...")

    storager = MultiModalStorager(vault_path, read_only=True)

    with storager.meta_handler as handler:
        # 总体统计
        if annotation_name:
            stats = handler.query_batch(
                """
                SELECT
                    name,
                    COUNT(*) as count,
                    COUNT(DISTINCT creator_id) as num_creators,
                    COUNT(DISTINCT sequence_id) as num_sequences
                FROM sample_annotations
                WHERE name = ?
                GROUP BY name
                """,
                [annotation_name],
            )
        else:
            stats = handler.query_batch(
                """
                SELECT
                    name,
                    COUNT(*) as count,
                    COUNT(DISTINCT creator_id) as num_creators,
                    COUNT(DISTINCT sequence_id) as num_sequences
                FROM sample_annotations
                GROUP BY name
                ORDER BY count DESC
                """
            )

        if stats:
            logger.info("\n=== 标注统计 ===")
            for stat in stats:
                logger.info(
                    f"  {stat['name']}: {stat['count']} 条标注, "
                    f"{stat['num_creators']} 个创建者, "
                    f"{stat['num_sequences']} 个序列"
                )

            # 数值型标注的分布
            if annotation_name:
                value_stats = handler.query_batch(
                    """
                    SELECT
                        MIN(value_float) as min_value,
                        MAX(value_float) as max_value,
                        AVG(value_float) as avg_value,
                        STDDEV(value_float) as std_value
                    FROM sample_annotations
                    WHERE name = ? AND value_float IS NOT NULL
                    """,
                    [annotation_name],
                )
                if value_stats and value_stats[0]["avg_value"] is not None:
                    v = value_stats[0]
                    logger.info(
                        f"\n  数值分布: min={v['min_value']:.2f}, "
                        f"max={v['max_value']:.2f}, "
                        f"avg={v['avg_value']:.2f}, "
                        f"std={v['std_value']:.2f}"
                    )
        else:
            logger.warning("没有找到标注数据")


def export_annotations_to_csv(
    vault_path: str = "/mnt/jfs/datasets/vault/example",
    annotation_name: str = "aesthetic_score",
    output_path: str = "/tmp/annotations.csv",
):
    """
    将标注导出为 CSV 文件

    Args:
        vault_path: Vault 路径
        annotation_name: 标注名称
        output_path: 输出文件路径
    """
    logger.info(f"导出标注: {annotation_name}")

    storager = MultiModalStorager(vault_path, read_only=True)

    with storager.meta_handler as handler:
        results = handler.query_batch(
            """
            SELECT
                sa.id as annotation_id,
                sa.name,
                sa.sequence_id,
                sa.value_float,
                sa.value_json,
                c.name as creator_name,
                sae.element_id,
                sae.element_type,
                sae.role
            FROM sample_annotations sa
            JOIN creators c ON sa.creator_id = c.id
            JOIN sample_annotation_elements sae ON sa.id = sae.sample_annotation_id
            WHERE sa.name = ?
            ORDER BY sa.sequence_id
            """,
            [annotation_name],
        )

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        logger.info(f"✅ 已导出 {len(results)} 条记录到 {output_path}")
    else:
        logger.warning(f"没有找到名为 '{annotation_name}' 的标注")


# ============================================================
# 命令行接口
# ============================================================


def main():
    """
    使用示例：

    # 场景 1: 添加美学评分
    python add_sample_annotations_tutorial.py add_aesthetic_scores_simple \\
        --vault_path=/path/to/vault \\
        --source_filter=my_dataset

    # 场景 2: 添加 CLIP 评分
    python add_sample_annotations_tutorial.py add_clip_scores \\
        --vault_path=/path/to/vault

    # 场景 3: 添加编辑质量评分
    python add_sample_annotations_tutorial.py add_edit_quality_scores \\
        --vault_path=/path/to/vault

    # 场景 4: 从文件导入
    python add_sample_annotations_tutorial.py import_from_file_example \\
        --vault_path=/path/to/vault \\
        --file_path=/path/to/scores.csv \\
        --file_type=csv

    # 场景 5: 分布式标注
    python add_sample_annotations_tutorial.py distributed_annotation_example \\
        --vault_path=/path/to/vault \\
        --num_workers=16

    # 查询标注
    python add_sample_annotations_tutorial.py query_annotations \\
        --vault_path=/path/to/vault \\
        --annotation_name=aesthetic_score

    # 导出标注
    python add_sample_annotations_tutorial.py export_annotations_to_csv \\
        --vault_path=/path/to/vault \\
        --annotation_name=aesthetic_score \\
        --output_path=/tmp/scores.csv
    """
    fire.Fire(
        {
            # 基础场景
            "add_aesthetic_scores": add_aesthetic_scores_simple,
            "add_clip_scores": add_clip_scores,
            "add_edit_quality": add_edit_quality_scores,
            # 批量导入
            "import_from_file": import_from_file_example,
            # 分布式处理
            "distributed": distributed_annotation_example,
            # 查询和验证
            "query": query_annotations,
            "export": export_annotations_to_csv,
        }
    )


if __name__ == "__main__":
    main()
