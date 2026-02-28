import binascii
import os
import uuid
from typing import Optional

import duckdb
import fire
import lance
import pandas as pd

# --- 通用辅助函数 ---


def print_header(title: str, char: str = "="):
    """打印格式化的章节标题"""
    print("\n" + char * 70)
    print(f"📊 {title}")
    print(char * 70)


def format_bytes_as_hex(byte_data: Optional[bytes]) -> Optional[str]:
    """安全地将字节转换为十六进制字符串"""
    if byte_data is None:
        return None
    return binascii.hexlify(byte_data).decode("ascii")


def format_uuid_bytes_as_hex(byte_data: Optional[bytes]) -> Optional[str]:
    """安全地将UUID字节转换为十六进制字符串"""
    if byte_data is None:
        return None
    try:
        return uuid.UUID(bytes=byte_data).hex
    except ValueError:
        return f"无效的UUID字节: {format_bytes_as_hex(byte_data)}"


# --- 模块1: Lance 画像分析器 ---


class LanceProfiler:
    """详细分析和画像Lance数据集。"""

    def __init__(self, lance_path: str):
        self.lance_path = lance_path
        self._lance_df = None
        # 定义 pandas.describe() 输出的中文映射
        self.describe_index_map = {
            "count": "计数",
            "mean": "平均值",
            "std": "标准差",
            "min": "最小值",
            "25%": "25分位",
            "50%": "中位数",
            "75%": "75分位",
            "max": "最大值",
        }

    def _load_metadata(self) -> pd.DataFrame:
        """加载Lance数据集元数据到Pandas DataFrame并缓存。"""
        if self._lance_df is None:
            print("⏳ 正在加载 Lance 元数据...")
            ds = lance.dataset(self.lance_path)
            columns_to_load = [
                field.name for field in ds.schema if field.name != "image"
            ]
            table = ds.to_table(columns=columns_to_load)
            df = table.to_pandas()
            df["id_hex"] = df["id"].apply(format_uuid_bytes_as_hex)
            df["file_hash_hex"] = df["file_hash"].apply(format_bytes_as_hex)
            df["pdq_hash_hex"] = df["pdq_hash"].apply(format_bytes_as_hex)
            df["file_size"] = df["file_size"].apply(lambda x: int(x) / 1024)
            self._lance_df = df
            print(f"✅ Lance 元数据加载完成 (共 {len(df)} 条记录)。")
        return self._lance_df

    def analyze_ids_and_hashes(self, df: pd.DataFrame):
        print_header("ID 与哈希值分析", "-")
        total_rows = len(df)
        for col, name in [
            ("id_hex", "图片ID"),
            ("file_hash_hex", "文件哈希"),
            ("pdq_hash_hex", "感知哈希(PDQ)"),
        ]:
            print(f"\n--- {name} ---")
            unique_count = df[col].nunique()
            uniqueness_ratio = unique_count / total_rows
            counts = df[col].value_counts()
            duplicates = counts[counts > 1]
            print(
                f"  - 唯一值: {unique_count} / {total_rows} (唯一性比例: {uniqueness_ratio:.2%})"
            )
            if duplicates.empty:  # pyright: ignore[reportAttributeAccessIssue]
                print("  - 重复项: ✅ 未发现重复值。")
            else:
                print(
                    f"  - 重复项: ⚠️ 发现 {len(duplicates)} 个值出现超过一次 (涉及 {duplicates.sum()} 行数据)。"
                )

    def analyze_dimensions_and_size(self, df: pd.DataFrame):
        print_header("图像维度与尺寸分析", "-")
        dim_cols = ["width", "height", "aspect_ratio", "file_size"]
        print("--- 描述性统计 ---")
        desc_df = df[dim_cols].describe().rename(index=self.describe_index_map)
        print(desc_df.to_string())
        print("\n--- Top 5 最常见分辨率 ---")
        common_dims = (
            df.groupby(["width", "height"]).size().nlargest(5).reset_index(name="数量")  # pyright: ignore[reportCallIssue]
        )
        print(common_dims.to_string(index=False))
        print("\n--- 数据质量检查 ---")
        invalid_size = df[
            (df["width"] <= 0) | (df["height"] <= 0) | (df["file_size"] <= 0)
        ]
        if invalid_size.empty:
            print("  ✅ 未发现尺寸或文件大小为0的无效数据。")
        else:
            print(f"  ⚠️ 发现 {len(invalid_size)} 条尺寸或文件大小无效的记录。")

    def analyze_source_and_uri(self, df: pd.DataFrame):
        print_header("数据来源与路径(URI)分析", "-")
        print("--- 数据来源 (`source`) 分布 ---")
        print(df["source"].value_counts().head(10).to_string())
        print("\n--- URI 顶层目录分布 ---")
        df["uri_prefix"] = df["uri"].str.split("/", n=1, expand=True)[0]
        print(df["uri_prefix"].value_counts().head(10).to_string())

    def analyze_color_and_quality(self, df: pd.DataFrame):
        print_header("色彩与质量指标分析", "-")
        print("--- 色彩模式 (`color_mode`) 分布 ---")
        print(df["color_mode"].value_counts().to_string())

        # 使用预定义的、可靠的质量指标列表进行分析
        EXPECTED_QUALITY_METRICS = [
            "mean_saturation",
            "mean_lightness",
            "clarity",
            "entropy",
            "edge_probability",
            "edge_near_patch_min_std",
            "pdq_quality",
        ]
        quality_cols_found = [
            col for col in EXPECTED_QUALITY_METRICS if col in df.columns
        ]

        print("\n--- 图像质量指标统计 ---")
        if not quality_cols_found:
            print("  - 在数据中未找到预期的质量指标列。")
        else:
            desc_df = (
                df[quality_cols_found].describe().rename(index=self.describe_index_map)
            )
            print(desc_df.to_string())

    def run(self):
        """执行针对Lance数据集的完整画像分析流程。"""
        print_header("Lance 数据集画像报告")
        df = self._load_metadata()
        self.analyze_ids_and_hashes(df)
        self.analyze_dimensions_and_size(df)
        self.analyze_source_and_uri(df)
        self.analyze_color_and_quality(df)
        print("\n✅ Lance 数据集画像分析完成。")


# --- 模块2: DuckDB 画像分析器 ---


class DuckDBProfiler:
    """分析和画像DuckDB数据库。"""

    def __init__(self, duckdb_path: str):
        self.duckdb_path = duckdb_path
        self._image_ids = None

    def _query(self, query: str) -> pd.DataFrame:
        """执行DuckDB查询并返回Pandas DataFrame。"""
        with duckdb.connect(self.duckdb_path, read_only=True) as con:
            return con.execute(query).fetchdf()

    def get_image_ids(self) -> set:
        """获取并缓存DuckDB中所有的图片ID。"""
        if self._image_ids is None:
            df = self._query("SELECT id FROM images")
            self._image_ids = set(
                df["id"].apply(lambda x: x.hex if isinstance(x, uuid.UUID) else str(x))
            )
        return self._image_ids

    def analyze_table_counts(self):
        print_header("核心表总览", "-")
        tables = [
            "sequences",
            "images",
            "texts",
            "annotations",
            "creators",
            "sequence_images",
        ]
        for table in tables:
            count = self._query(f"SELECT COUNT(*) FROM {table}").iloc[0, 0]
            print(f"  - {table.capitalize():<16}: {count} 行")

    def analyze_sequences(self):
        print_header("序列(Sequence)分析", "-")
        seq_counts_df = self._query(
            "SELECT sequence_id, COUNT(image_id) as image_count FROM sequence_images GROUP BY sequence_id"
        )
        if seq_counts_df.empty:
            print("数据库中未发现包含图片的序列。")
            return
        stats = seq_counts_df["image_count"].describe()
        print("  - 每个序列包含的图片数量统计:")
        print(f"    - 平均值: {stats['mean']:.2f}, 标准差: {stats['std']:.2f}")
        print(
            f"    - 最小值: {int(stats['min'])}, 中位数: {int(stats['50%'])}, 最大值: {int(stats['max'])}"  # type: ignore[reportCallIssue]
        )
        bins = [0, 1, 5, 10, 20, 50, float("inf")]
        labels = ["1张图片", "2-5张", "6-10张", "11-20张", "21-50张", "50+张"]
        dist = (
            pd.cut(seq_counts_df["image_count"], bins=bins, labels=labels, right=True)
            .value_counts()  # pyright: ignore[reportAttributeAccessIssue]
            .sort_index()
        )
        print("\n  - 序列长度分布:\n", dist.to_string())

    def analyze_orphan_images(self):
        print_header("孤立图片(Orphan Image)分析", "-")
        orphan_count = self._query(
            "SELECT COUNT(id) FROM images WHERE id NOT IN (SELECT DISTINCT image_id FROM sequence_images)"
        ).iloc[0, 0]
        total_images = len(self.get_image_ids())
        print(f"  - 发现 {orphan_count} 张孤立图片 (未被任何序列引用)。")
        if total_images > 0:
            print(f"  - 孤立图片占比: {orphan_count / total_images:.2%}")

    def run(self):
        """执行针对DuckDB数据库的完整画像分析流程。"""
        print_header("DuckDB 数据库画像报告")
        self.analyze_table_counts()
        self.analyze_sequences()
        self.analyze_orphan_images()
        print("\n✅ DuckDB 数据库画像分析完成。")


# --- 模块3: 交叉校验器 ---


class CrossValidator:
    """在Lance和DuckDB之间执行完整性校验。"""

    def __init__(self, lance_profiler: LanceProfiler, duckdb_profiler: DuckDBProfiler):
        self.lance_profiler = lance_profiler
        self.duckdb_profiler = duckdb_profiler

    def check_id_integrity(self):
        print_header("ID 完整性交叉校验", "-")
        lance_df = self.lance_profiler._load_metadata()
        lance_ids = set(lance_df["id_hex"])
        duckdb_ids = self.duckdb_profiler.get_image_ids()

        print(f"  - Lance 中的唯一图片 ID 数量: {len(lance_ids)}")
        print(f"  - DuckDB 中的唯一图片 ID 数量: {len(duckdb_ids)}")

        if lance_ids == duckdb_ids:
            print("  - ✅ 完美匹配！Lance 和 DuckDB 中的图片 ID 完全一致。")
        else:
            print("  - ⚠️ 发现不匹配！")
            if diff := lance_ids - duckdb_ids:
                print(f"    - {len(diff)} 个 ID 存在于 Lance 但不存在于 DuckDB。")
            if diff := duckdb_ids - lance_ids:
                print(f"    - {len(diff)} 个 ID 存在于 DuckDB 但不存在于 Lance。")

    def run(self):
        """执行所有交叉校验分析。"""
        print_header("Lance & DuckDB 完整性交叉校验")
        self.check_id_integrity()
        print("\n✅ 交叉校验完成。")


# --- 主程序命令行接口 ---


class DataVaultCommander:
    """一个用于分析和画像 Lance + DuckDB 混合数据仓库的命令行工具。"""

    def __init__(
        self,
        vault_path: str = "/mnt/jfs/datasets/vault/composed_image_retrieval/20250905-cirr",
    ):
        self.vault_path = vault_path
        self.lance_path = os.path.join(vault_path, "images")
        self.duckdb_path = os.path.join(vault_path, "metadata.duckdb")

        if not os.path.exists(self.lance_path):
            raise FileNotFoundError(f"Lance 路径不存在: {self.lance_path}")
        if not os.path.exists(self.duckdb_path):
            raise FileNotFoundError(f"DuckDB 路径不存在: {self.duckdb_path}")

    def profile_lance(self):
        """[命令] 对 Lance 图片数据集进行完整的、详细的画像分析。"""
        profiler = LanceProfiler(self.lance_path)
        profiler.run()

    def profile_duckdb(self):
        """[命令] 对 DuckDB 关系型元数据进行完整的画像分析。"""
        profiler = DuckDBProfiler(self.duckdb_path)
        profiler.run()

    def cross_validate(self):
        """[命令] 对 Lance 和 DuckDB 进行交叉完整性校验。"""
        lance_profiler = LanceProfiler(self.lance_path)
        duckdb_profiler = DuckDBProfiler(self.duckdb_path)
        validator = CrossValidator(lance_profiler, duckdb_profiler)
        validator.run()

    def full_report(self):
        """[命令] 运行所有分析模块，生成一份全面的综合报告。"""
        print_header("开始生成数据仓库完整画像综合报告", "*")

        lance_profiler = LanceProfiler(self.lance_path)
        duckdb_profiler = DuckDBProfiler(self.duckdb_path)

        lance_profiler.run()
        duckdb_profiler.run()

        validator = CrossValidator(lance_profiler, duckdb_profiler)
        validator.run()

        print_header("综合报告生成完毕", "*")


if __name__ == "__main__":
    fire.Fire(DataVaultCommander)
