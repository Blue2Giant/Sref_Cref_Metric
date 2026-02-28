"""
Stepflow Tag Tool - Single image scene tagging for movie analysis.

This tool analyzes individual movie frames and generates structured tags for:
- Content assessment (meaningful-content, credits, solid-color)
- Quality flags (watermarks, subtitles, blur)
- Spatial domain (indoor, outdoor, etc.)
- Scene atmosphere (lighting, weather)
- Camera framing and angles
- Character composition
- Visual style

Usage:
    from llm_annotate.core import VaultAnnotator
    from llm_annotate.tools.stepflow_tag_tool import StepflowTagTool

    tool = StepflowTagTool(model_name="qwen3-vl-30ba3b")
    annotator = VaultAnnotator(
        tool=tool,
        vault_path="/path/to/vault",
        output_path="/path/to/output.duckdb",
        batch_size=32,
        max_workers=256,
    )
    annotator.run()
"""

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import fire
from llm_annotator import AnnotateTool
from llm_annotator.prompt_loader import load_prompt
from llm_annotator.utils import call_vlm_single
from openai import OpenAI
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from vault.schema.multimodal import MultiModalType
from vault.storage.lanceduck.multimodal import MultiModalStorager

# Configuration
OPENAI_BASE_URL = os.environ.get(
    "OPENAI_BASE_URL", "http://stepcast-router.shai-core:9200/v1"
)


SINGLE_CHOICE_FIELDS = [
    "content-assessment",
    "spatial-domain",
    "shot-scale",
    "camera-angle",
    "character-composition",
]

MULTI_CHOICE_FIELDS = ["quality-flags", "scene-atmosphere", "visual-style"]

ALL_FIELDS = SINGLE_CHOICE_FIELDS + MULTI_CHOICE_FIELDS


class StepflowTagTool(AnnotateTool):
    """
    Stepflow movie scene tagging tool.

    Analyzes single images to generate structured scene metadata.
    """

    basic_name: str = "stepflow_movie_tag"
    sample_type: str = "image"

    def __init__(self, model_name: str):
        super().__init__(model_name)

        # Get template directory (templates/ next to this file)
        template_dir = Path(__file__).parent / "templates"

        # Load prompt template
        self.prompt = load_prompt("stepflow_tag.j2", template_dir=template_dir)

        # Store prompt in creator metadata
        self.creator_meta = dict(prompt=self.prompt)

        # Initialize OpenAI client (shared across all calls)
        self.client = OpenAI(api_key="EMPTY", base_url=OPENAI_BASE_URL, timeout=3600)

    def _prepare_kwargs_and_participants(
        self, sample: Dict, storager: MultiModalStorager
    ):
        """
        Prepare data for single image tagging.

        Args:
            sample: Dict with 'sample_id' (image ID) and 'sequence_id'
            storager: Vault storage interface

        Returns:
            Tuple of (participants, kwargs)
        """
        # Get image bytes from vault
        image_bytes = storager.get_image_bytes_by_ids([sample["sample_id"]])

        # Define participants (what elements this annotation relates to)
        participants = ((sample["sample_id"], MultiModalType.IMAGE, "image"),)

        # Prepare API call arguments
        kwargs = {
            "image": image_bytes[sample["sample_id"]],
            "text": self.prompt,
        }

        return participants, kwargs

    def __call__(self, image, text, **kwargs):
        """
        Execute VLM API call for image tagging.

        Args:
            image: Image bytes
            text: Prompt text
            **kwargs: Additional arguments (unused)

        Returns:
            JSON string with tagging results
        """
        return call_vlm_single(
            image=image,
            text=text,
            model_name=self.model_name,
            client=self.client,
        )


class AnalysisTool:
    """一个用于分析 DuckDB 中 VLM 标注数据的 CLI 工具。"""

    def _load_data_from_duckdb(self, db_path: str) -> List[Dict[str, Any]]:
        """从 DuckDB 数据库加载并解析标注数据。"""
        annotations = []
        console.print(f"\n[cyan]正在连接到数据库: [bold]{db_path}[/bold]...[/cyan]")
        try:
            con = duckdb.connect(database=db_path, read_only=True)
            # 首先获取总行数以用于进度条
            total_rows = con.execute(
                "SELECT COUNT(*) FROM sample_annotations WHERE value_json IS NOT NULL"
            ).fetchone()[0]  # type: ignore

            if total_rows == 0:
                console.print(
                    "[yellow]警告: 表 'sample_annotations' 中没有找到有效的 JSON 数据。[/yellow]"
                )
                con.close()
                return []

            # 获取所有相关的 JSON 字符串
            results = con.execute(
                "SELECT value_json FROM sample_annotations WHERE value_json IS NOT NULL"
            ).fetchall()
            con.close()
            console.print(
                f"[green]数据库连接成功，发现 [bold]{total_rows}[/bold] 条记录。[/green]"
            )

        except duckdb.Error as e:
            console.print(f"[bold red]数据库错误: {e}[/bold red]")
            return []

        # 使用进度条解析 JSON
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]正在解析 JSON 数据...", total=len(results))
            malformed_count = 0
            for row in results:
                json_string = row[0]
                try:
                    data = json.loads(json_string)

                    # 基础校验
                    if "content-assessment" in data:
                        annotations.append(data)
                except json.JSONDecodeError:
                    malformed_count += 1
                progress.update(task, advance=1)

        if malformed_count > 0:
            console.print(
                f"[yellow]警告: 跳过了 {malformed_count} 条格式错误的 JSON 记录。[/yellow]"
            )

        return annotations

    def _analyze_data(self, data: List[Dict[str, Any]]):
        """对加载的数据进行全面的统计分析。 (逻辑与之前脚本相同)"""
        total_files = len(data)
        if total_files == 0:
            return None

        content_assessment_counts = Counter(
            item.get("content-assessment", "未知") for item in data
        )

        meaningful_count = content_assessment_counts.get("meaningful-content", 0)
        meaningful_data = [
            item
            for item in data
            if item.get("content-assessment") == "meaningful-content"
        ]

        stats = {}
        for field in ALL_FIELDS:
            if field == "content-assessment":
                stats[field] = content_assessment_counts
                continue

            counter = Counter()
            if not meaningful_data:
                stats[field] = counter
                continue

            if field in SINGLE_CHOICE_FIELDS:
                tags = [item.get(field) or "未标注" for item in meaningful_data]
                counter.update(tags)
            elif field in MULTI_CHOICE_FIELDS:
                for item in meaningful_data:
                    tags = item.get(field, [])
                    if tags:
                        counter.update(tags)
            stats[field] = counter

        return total_files, meaningful_count, stats

    def _display_results(self, total_files, meaningful_count, stats):
        """使用 Rich 库美观地展示统计结果 (修复布局卡顿版)。"""
        console.clear()

        # --- 标题 ---
        console.print(
            Panel(
                Align.center(
                    "[bold cyan]VLM 图像标注统计分析报告 (来源: DuckDB)[/bold cyan]"
                ),
                border_style="bold blue",
                padding=(1, 2),
            )
        )

        # --- 总体概览 ---
        overview_table = Table(
            title="[bold]📊 总体概览[/bold]",
            show_header=True,
            header_style="bold magenta",
        )
        overview_table.add_column("评估类别", style="dim", width=25)
        overview_table.add_column("数量", justify="right")
        overview_table.add_column("占比", justify="right")

        for category, count in stats["content-assessment"].items():
            percentage = (count / total_files) * 100 if total_files > 0 else 0
            overview_table.add_row(category, str(count), f"{percentage:.2f}%")

        console.print(overview_table)
        console.print(
            f"[dim]总共分析了 [bold]{total_files}[/bold] 条记录，其中 [bold green]{meaningful_count}[/bold green] 条为 'meaningful-content'。[/dim]\n"
        )

        if meaningful_count == 0:
            console.print(
                "[yellow]没有 'meaningful-content' 的数据，无法进行后续分析。[/yellow]"
            )
            return

        console.print(
            Panel(
                Align.center("[bold]🏷️ 详细标签分布 ('meaningful-content' Only)[/bold]"),
                border_style="blue",
            )
        )

        # --- 详细标签分布 (使用 Grid + Group 优化性能) ---

        # 准备左右两列的内容列表
        left_panels = []
        right_panels = []

        fields_to_display = [f for f in ALL_FIELDS if f != "content-assessment"]

        for i, field in enumerate(fields_to_display):
            # 获取统计数据
            field_stats = stats.get(field)
            if not field_stats:
                continue

            is_multi_choice = field in MULTI_CHOICE_FIELDS
            # 多选的总分母是标签总数，单选的分母是有效图片数
            total_tags = (
                sum(field_stats.values()) if is_multi_choice else meaningful_count
            )

            # 创建单个表格
            table = Table(
                title=f"[bold]{field.replace('-', ' ').title()}[/bold]",
                header_style="bold green",
                expand=True,
            )
            table.add_column("标签", style="cyan")
            table.add_column("数量", justify="right", style="magenta")
            table.add_column("占比", justify="right", style="yellow")

            sorted_tags = sorted(
                field_stats.items(), key=lambda item: item[1], reverse=True
            )

            for tag, count in sorted_tags:
                percentage = (count / total_tags) * 100 if total_tags > 0 else 0
                table.add_row(tag, str(count), f"{percentage:.2f}%")

            # 将表格包裹在 Panel 中
            panel = Panel(table, border_style="green")

            # 分配到左列或右列
            if i % 2 == 0:
                left_panels.append(panel)
            else:
                right_panels.append(panel)

        # 创建一个无边框的 Grid 表格来作为布局容器
        grid = Table.grid(expand=True, padding=1)
        grid.add_column(ratio=1)  # 左列
        grid.add_column(ratio=1)  # 右列

        # 将 Panel 列表转换为 Group 对象放入 Grid 中
        # 这是一次性渲染，不会造成递归计算卡顿
        grid.add_row(Group(*left_panels), Group(*right_panels))

        console.print(grid)

    def run(self, db_path: str):
        """
        运行完整的分析流程。

        :param db_path: DuckDB 数据库文件的路径。
        """
        annotations = self._load_data_from_duckdb(db_path)
        if not annotations:
            console.print("[bold red]未能加载任何数据，程序退出。[/bold red]")
            return

        analysis_result = self._analyze_data(annotations)
        if analysis_result:
            total_files, meaningful_count, stats = analysis_result
            self._display_results(total_files, meaningful_count, stats)


if __name__ == "__main__":
    console = Console()

    fire.Fire(AnalysisTool)
