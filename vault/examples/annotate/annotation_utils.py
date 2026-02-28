"""
样本级标注工具集 - 实用辅助函数

提供一组开箱即用的工具函数，简化常见的标注任务。
"""

from pathlib import Path
from typing import Callable, Literal

import pandas as pd
from loguru import logger
from tqdm import tqdm

from vault.backend.duckdb import DuckDBHandler
from vault.schema import ID
from vault.schema.multimodal import Creator, MultiModalType, SampleAnnotation
from vault.storage.lanceduck.multimodal import MultiModalStorager


# ============================================================
# 工具 1: 通用标注流水线
# ============================================================


class AnnotationPipeline:
    """
    简化标注流程的通用流水线

    示例：
        pipeline = AnnotationPipeline(
            vault_path="/path/to/vault",
            annotation_name="aesthetic_score",
            creator_name="aesthetic_v1",
        )

        # 定义标注函数
        def score_image(sequence_id, image_id):
            return 7.5  # 你的逻辑

        # 运行流水线
        pipeline.run(
            query="SELECT s.id, si.image_id FROM sequences s ...",
            annotator_func=score_image,
        )
    """

    def __init__(
        self,
        vault_path: str,
        annotation_name: str,
        creator_name: str,
        creator_meta: dict | None = None,
        temp_db_dir: str = "/tmp/vault_annotations",
    ):
        self.vault_path = vault_path
        self.annotation_name = annotation_name
        self.creator_name = creator_name
        self.creator_meta = creator_meta or {}
        self.temp_db_dir = Path(temp_db_dir)
        self.temp_db_dir.mkdir(exist_ok=True)

        self.creator = Creator.create(name=creator_name, meta=creator_meta)

    def run(
        self,
        query: str,
        annotator_func: Callable,
        query_params: list | None = None,
        participant_type: Literal["image", "text", "both"] = "image",
        batch_size: int = 1000,
        show_progress: bool = True,
    ):
        """
        运行标注流水线

        Args:
            query: SQL 查询，必须返回 sequence_id 和元素 ID
            annotator_func: 标注函数，接收查询结果行，返回标注值
            query_params: SQL 查询参数
            participant_type: 参与元素类型
            batch_size: 批处理大小
            show_progress: 是否显示进度条
        """
        # 1. 查询数据
        storager = MultiModalStorager(self.vault_path, read_only=True)
        with storager.meta_handler as handler:
            items = handler.query_batch(query, query_params or [])

        if not items:
            logger.warning("查询未返回任何数据")
            return

        logger.info(f"找到 {len(items)} 个需要标注的样本")

        # 2. 批量标注
        annotations = []
        iterator = tqdm(items, desc="创建标注") if show_progress else items

        for item in iterator:
            try:
                # 调用用户的标注函数
                value = annotator_func(item)

                # 确定参与者
                sequence_id = ID.from_(item["sequence_id"])

                if participant_type == "image":
                    participants = ((ID.from_(item["image_id"]), MultiModalType.IMAGE, "target"),)
                elif participant_type == "text":
                    participants = ((ID.from_(item["text_id"]), MultiModalType.TEXT, "target"),)
                elif participant_type == "both":
                    participants = (
                        (ID.from_(item["image_id"]), MultiModalType.IMAGE, "image"),
                        (ID.from_(item["text_id"]), MultiModalType.TEXT, "text"),
                    )
                else:
                    raise ValueError(f"Unsupported participant_type: {participant_type}")

                annotation = SampleAnnotation.create(
                    name=self.annotation_name,
                    sequence_id=sequence_id,
                    creator=self.creator,
                    value=value,
                    participants=participants,
                )
                annotations.append(annotation)

            except Exception as e:
                logger.warning(f"标注失败，跳过: {e}")
                continue

        logger.info(f"成功创建 {len(annotations)} 条标注")

        # 3. 写入
        self._write_annotations(annotations, batch_size)

        # 4. 验证
        self._verify_annotations()

    def _write_annotations(self, annotations: list[SampleAnnotation], batch_size: int):
        """写入标注"""
        temp_db_path = self.temp_db_dir / f"{self.annotation_name}.duckdb"

        storager = MultiModalStorager(self.vault_path, read_only=False)
        temp_handler = DuckDBHandler(
            schema=storager.DUCKDB_SCHEMA,
            read_only=False,
            db_path=str(temp_db_path),
        )
        temp_handler.create()

        logger.info("写入标注到临时数据库...")
        storager.add_sample_annotations(
            annotations, duckdb_handler=temp_handler, batch_size=batch_size
        )

        logger.info("合并到主 Vault...")
        storager.merge(duckdb_files=[str(temp_db_path)])

        logger.info(f"✅ 完成！临时文件: {temp_db_path}")

    def _verify_annotations(self):
        """验证标注结果"""
        storager = MultiModalStorager(self.vault_path, read_only=True)
        with storager.meta_handler as handler:
            result = handler.query_batch(
                """
                SELECT
                    COUNT(*) as count,
                    MIN(value_float) as min_val,
                    MAX(value_float) as max_val,
                    AVG(value_float) as avg_val
                FROM sample_annotations
                WHERE name = ?
                """,
                [self.annotation_name],
            )[0]

            logger.info(f"验证: {result['count']} 条标注")
            if result["avg_val"] is not None:
                logger.info(
                    f"数值范围: {result['min_val']:.2f} - {result['max_val']:.2f}, "
                    f"平均: {result['avg_val']:.2f}"
                )


# ============================================================
# 工具 2: 标注统计和可视化
# ============================================================


class AnnotationAnalyzer:
    """
    标注数据分析器

    示例：
        analyzer = AnnotationAnalyzer("/path/to/vault")
        analyzer.summary()  # 总览
        analyzer.distribution("aesthetic_score")  # 分布
        analyzer.compare_creators("aesthetic_score")  # 对比创建者
    """

    def __init__(self, vault_path: str):
        self.vault_path = vault_path
        self.storager = MultiModalStorager(vault_path, read_only=True)

    def summary(self):
        """显示所有标注的总览"""
        with self.storager.meta_handler as handler:
            stats = handler.query_batch(
                """
                SELECT
                    sa.name,
                    COUNT(*) as count,
                    COUNT(DISTINCT sa.creator_id) as num_creators,
                    COUNT(DISTINCT sa.sequence_id) as num_sequences,
                    MIN(sa.value_float) as min_value,
                    MAX(sa.value_float) as max_value,
                    AVG(sa.value_float) as avg_value
                FROM sample_annotations sa
                GROUP BY sa.name
                ORDER BY count DESC
                """
            )

        if not stats:
            logger.info("没有找到任何标注")
            return

        # 使用 pandas 美化输出
        df = pd.DataFrame(stats)
        print("\n=== 标注总览 ===")
        print(df.to_string(index=False))

        return df

    def distribution(self, annotation_name: str, bins: int = 10):
        """
        分析标注值的分布

        Args:
            annotation_name: 标注名称
            bins: 分箱数量
        """
        with self.storager.meta_handler as handler:
            values = handler.query_batch(
                "SELECT value_float FROM sample_annotations WHERE name = ? AND value_float IS NOT NULL",
                [annotation_name],
            )

        if not values:
            logger.warning(f"未找到 '{annotation_name}' 的数值型标注")
            return

        values_list = [v["value_float"] for v in values]
        df = pd.DataFrame({"value": values_list})

        print(f"\n=== {annotation_name} 分布统计 ===")
        print(df.describe())

        # 绘制直方图（文本版）
        counts, bin_edges = pd.cut(df["value"], bins=bins, retbins=True)
        hist = counts.value_counts().sort_index()

        print(f"\n=== 分布直方图 ===")
        max_count = hist.max()
        for interval, count in hist.items():
            bar_length = int(50 * count / max_count)
            bar = "█" * bar_length
            print(f"{str(interval):20s} | {bar} {count}")

        return df

    def compare_creators(self, annotation_name: str):
        """对比不同创建者的标注"""
        with self.storager.meta_handler as handler:
            stats = handler.query_batch(
                """
                SELECT
                    c.name as creator,
                    COUNT(*) as count,
                    MIN(sa.value_float) as min_value,
                    MAX(sa.value_float) as max_value,
                    AVG(sa.value_float) as avg_value,
                    STDDEV(sa.value_float) as std_value
                FROM sample_annotations sa
                JOIN creators c ON sa.creator_id = c.id
                WHERE sa.name = ?
                GROUP BY c.name
                ORDER BY count DESC
                """,
                [annotation_name],
            )

        if not stats:
            logger.warning(f"未找到 '{annotation_name}' 的标注")
            return

        df = pd.DataFrame(stats)
        print(f"\n=== {annotation_name} - 创建者对比 ===")
        print(df.to_string(index=False))

        return df

    def find_outliers(self, annotation_name: str, threshold: float = 2.0):
        """
        找出异常值（基于 Z-score）

        Args:
            annotation_name: 标注名称
            threshold: Z-score 阈值（默认 2.0）
        """
        with self.storager.meta_handler as handler:
            data = handler.query_batch(
                """
                SELECT
                    sa.id,
                    sa.sequence_id,
                    sa.value_float,
                    c.name as creator
                FROM sample_annotations sa
                JOIN creators c ON sa.creator_id = c.id
                WHERE sa.name = ? AND sa.value_float IS NOT NULL
                """,
                [annotation_name],
            )

        if not data:
            return

        df = pd.DataFrame(data)
        mean = df["value_float"].mean()
        std = df["value_float"].std()

        df["z_score"] = (df["value_float"] - mean) / std
        outliers = df[abs(df["z_score"]) > threshold]

        print(f"\n=== {annotation_name} - 异常值检测 ===")
        print(f"均值: {mean:.2f}, 标准差: {std:.2f}")
        print(f"发现 {len(outliers)} 个异常值（|Z-score| > {threshold}）")

        if len(outliers) > 0:
            print("\n前 10 个异常值：")
            print(
                outliers.nlargest(10, "z_score")[
                    ["sequence_id", "value_float", "z_score", "creator"]
                ].to_string(index=False)
            )

        return outliers


# ============================================================
# 工具 3: 标注清理和管理
# ============================================================


class AnnotationManager:
    """
    标注管理器 - 删除、重命名、导出等操作

    示例：
        manager = AnnotationManager("/path/to/vault")
        manager.export_to_csv("aesthetic_score", "/tmp/export.csv")
        manager.delete_by_creator("aesthetic_score", "old_scorer_v1")
    """

    def __init__(self, vault_path: str):
        self.vault_path = vault_path
        self.storager = MultiModalStorager(vault_path, read_only=False)

    def export_to_csv(
        self, annotation_name: str, output_path: str, include_elements: bool = True
    ):
        """
        导出标注到 CSV

        Args:
            annotation_name: 标注名称
            output_path: 输出文件路径
            include_elements: 是否包含元素 ID
        """
        with self.storager.meta_handler as handler:
            if include_elements:
                query = """
                    SELECT
                        sa.id as annotation_id,
                        sa.sequence_id,
                        sa.value_float,
                        sa.value_json,
                        c.name as creator_name,
                        sae.element_id,
                        sae.element_type,
                        sae.role
                    FROM sample_annotations sa
                    JOIN creators c ON sa.creator_id = c.id
                    LEFT JOIN sample_annotation_elements sae ON sa.id = sae.sample_annotation_id
                    WHERE sa.name = ?
                    ORDER BY sa.sequence_id
                """
            else:
                query = """
                    SELECT
                        sa.id as annotation_id,
                        sa.sequence_id,
                        sa.value_float,
                        sa.value_json,
                        c.name as creator_name
                    FROM sample_annotations sa
                    JOIN creators c ON sa.creator_id = c.id
                    WHERE sa.name = ?
                    ORDER BY sa.sequence_id
                """

            results = handler.query_batch(query, [annotation_name])

        if not results:
            logger.warning(f"未找到 '{annotation_name}' 的标注")
            return

        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        logger.info(f"✅ 已导出 {len(results)} 条记录到 {output_path}")

        return df

    def export_to_parquet(self, annotation_name: str, output_path: str):
        """导出标注到 Parquet（推荐大数据量）"""
        with self.storager.meta_handler as handler:
            results = handler.query_batch(
                """
                SELECT
                    sa.id as annotation_id,
                    sa.sequence_id,
                    sa.value_float,
                    sa.value_json,
                    c.name as creator_name
                FROM sample_annotations sa
                JOIN creators c ON sa.creator_id = c.id
                WHERE sa.name = ?
                """,
                [annotation_name],
            )

        if not results:
            logger.warning(f"未找到 '{annotation_name}' 的标注")
            return

        df = pd.DataFrame(results)
        df.to_parquet(output_path, index=False)
        logger.info(f"✅ 已导出 {len(results)} 条记录到 {output_path}")

        return df

    def list_creators(self, annotation_name: str | None = None):
        """列出所有创建者"""
        with self.storager.meta_handler as handler:
            if annotation_name:
                creators = handler.query_batch(
                    """
                    SELECT DISTINCT c.id, c.name, c.meta
                    FROM creators c
                    JOIN sample_annotations sa ON c.id = sa.creator_id
                    WHERE sa.name = ?
                    """,
                    [annotation_name],
                )
            else:
                creators = handler.query_batch("SELECT id, name, meta FROM creators")

        df = pd.DataFrame(creators)
        print("\n=== 创建者列表 ===")
        print(df.to_string(index=False))

        return df


# ============================================================
# 使用示例
# ============================================================


def example_pipeline():
    """示例：使用流水线快速添加标注"""

    # 创建流水线
    pipeline = AnnotationPipeline(
        vault_path="/path/to/vault",
        annotation_name="aesthetic_score",
        creator_name="aesthetic_v1",
        creator_meta={"model": "aesthetic-predictor-v2.5"},
    )

    # 定义标注函数
    def score_function(item):
        # 这里应该是你的模型推理逻辑
        # image_bytes = get_image(item['image_id'])
        # score = model.predict(image_bytes)
        import random

        return random.uniform(5.0, 9.0)

    # 运行流水线
    pipeline.run(
        query="""
            SELECT s.id as sequence_id, si.image_id
            FROM sequences s
            JOIN sequence_images si ON s.id = si.sequence_id
            WHERE s.source = ?
            LIMIT 1000
        """,
        query_params=["my_dataset"],
        annotator_func=score_function,
        participant_type="image",
    )


def example_analysis():
    """示例：分析标注"""
    analyzer = AnnotationAnalyzer("/path/to/vault")

    # 总览
    analyzer.summary()

    # 分布
    analyzer.distribution("aesthetic_score", bins=20)

    # 对比创建者
    analyzer.compare_creators("aesthetic_score")

    # 找异常值
    outliers = analyzer.find_outliers("aesthetic_score", threshold=2.5)


def example_management():
    """示例：管理标注"""
    manager = AnnotationManager("/path/to/vault")

    # 导出
    manager.export_to_csv("aesthetic_score", "/tmp/scores.csv")

    # 列出创建者
    manager.list_creators("aesthetic_score")


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "example_pipeline": example_pipeline,
            "example_analysis": example_analysis,
            "example_management": example_management,
        }
    )
