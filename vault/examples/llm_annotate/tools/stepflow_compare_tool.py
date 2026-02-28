import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import fire
from llm_annotator import AnnotateTool
from llm_annotator.prompt_loader import PromptLoader
from llm_annotator.utils import call_vlm_compare
from loguru import logger
from openai import OpenAI
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from vault.schema import ID
from vault.schema.multimodal import MultiModalType
from vault.storage.lanceduck.multimodal import MultiModalStorager

# Configuration
OPENAI_BASE_URL = os.environ.get(
    "OPENAI_BASE_URL", "http://stepcast-router.shai-core:9200/v1"
)


class StepflowCompareTool(AnnotateTool):
    basic_name: str = "stepflow_movie_compare"
    sample_type: str = "sequence"

    template_name: str = "stepflow_compare.j2"

    def __init__(self, model_name: str):
        super().__init__(model_name)

        # Get template directory (templates/ next to this file)
        template_dir = Path(__file__).parent / "templates"

        # Load prompt template
        self.prompt_loader = PromptLoader(template_dir=template_dir)
        self.prompt = self.prompt_loader.load(self.template_name)

        # Store prompt in creator metadata
        self.creator_meta = dict(prompt=self.prompt)

        # Initialize OpenAI client (shared across all calls)
        self.client = OpenAI(api_key="EMPTY", base_url=OPENAI_BASE_URL, timeout=3600)

    def _get_all_samples(self, storager: MultiModalStorager) -> List[Dict]:
        _sequence_ids = storager.meta_handler.query_batch(
            """
            SELECT sequence_id as sample_id
            FROM sample_annotations
            GROUP BY sequence_id
            HAVING COUNT(CASE WHEN json_extract_string(value_json, '$.content-assessment') <> 'meaningful-content' THEN 1 END) = 0;
            """
        )
        sequence_metas = storager.get_sequence_metas(
            [ID.from_(s["sample_id"]) for s in _sequence_ids]
        )
        samples = []
        for _meta in sequence_metas:
            _meta["sample_id"] = _meta["sequence_id"]
            samples.append(_meta)

        samples.sort(key=lambda x: x["sample_id"].to_int())
        return samples

    def _prepare_kwargs_and_participants(
        self, sample: Dict, storager: MultiModalStorager
    ):
        prev_image_id = None
        current_image_id = None
        for img in sample["images"]:
            if "prev" in img["index"]:
                prev_image_id = ID.from_(img["id"])
            if "current" in img["index"]:
                current_image_id = ID.from_(img["id"])

        if prev_image_id is None or current_image_id is None:
            logger.warning(
                f"can not found valid sample in sequence: {sample['sample_id']}"
            )
            return None, None

        # Get image bytes from vault
        image_bytes = storager.get_image_bytes_by_ids([prev_image_id, current_image_id])

        # Define participants (what elements this annotation relates to)
        participants = (
            (prev_image_id, MultiModalType.IMAGE, "source"),
            (current_image_id, MultiModalType.IMAGE, "target"),
        )

        frame_tags = storager.meta_handler.query_batch(
            """
            SELECT
            sae.element_id AS image_id,
            sa.value_json AS value
            FROM
            sample_annotation_elements AS sae
            JOIN
            sample_annotations AS sa ON sae.sample_annotation_id = sa.id
            WHERE
            sae.element_id IN ?;
            """,
            [tuple(id_.to_uuid() for id_ in [prev_image_id, current_image_id])],
        )
        frame_tags = {ID.from_(item["image_id"]): item["value"] for item in frame_tags}

        # Prepare API call arguments
        kwargs = {
            "source_image": image_bytes[prev_image_id],
            "target_image": image_bytes[current_image_id],
            "text": self.prompt_loader.render(
                self.template_name,
                context={
                    "frame_a_tags": frame_tags.get(prev_image_id),
                    "frame_b_tags": frame_tags.get(current_image_id),
                },
            ),
        }
        return participants, kwargs

    def __call__(self, source_image, target_image, text: str, **kwargs):
        o = call_vlm_compare(
            source_image=source_image,
            target_image=target_image,
            text=text,
            model_name=self.model_name,
            client=self.client,
        )
        return o


class CompareAnalysisTool:
    """
    A CLI tool to analyze VLM image pair comparison data stored in DuckDB.
    """

    def _load_data_from_duckdb(self, db_path: str, name: str) -> List[Dict[str, Any]]:
        """Loads and parses JSON data from a specific annotation name in DuckDB."""
        annotations = []
        console.print(
            f"\n[cyan]Connecting to [bold]{db_path}[/bold] and querying for name: [bold]'{name}'[/bold]...[/cyan]"
        )
        try:
            with duckdb.connect(database=db_path, read_only=True) as con:
                # Use a prepared statement for safety and clarity
                query = "SELECT value_json FROM sample_annotations WHERE name = ? AND value_json IS NOT NULL"
                results = con.execute(query, [name]).fetchall()
        except duckdb.Error as e:
            console.print(f"[bold red]Database Error: {e}[/bold red]")
            return []

        if not results:
            console.print(
                f"[yellow]Warning: No records found with the name '{name}'.[/yellow]"
            )
            return []

        console.print(f"[green]✓ Found {len(results)} records. Parsing JSON...[/green]")

        malformed_count = 0
        for row in results:
            try:
                data = json.loads(row[0])
                # Basic validation to ensure the structure is as expected
                if (
                    "analysis_report" in data
                    and "transition_type" in data["analysis_report"]
                ):
                    annotations.append(data)
                else:
                    malformed_count += 1
            except (json.JSONDecodeError, TypeError):
                malformed_count += 1

        if malformed_count > 0:
            console.print(
                f"[yellow]Warning: Skipped {malformed_count} malformed or incomplete JSON records.[/yellow]"
            )

        return annotations

    def _analyze_data(self, data: List[Dict[str, Any]]):
        """Performs a comprehensive statistical analysis of the annotation data."""
        total_pairs = len(data)
        if total_pairs == 0:
            return None

        # --- Primary Counters ---
        transition_counter = Counter(
            item["analysis_report"]["transition_type"] for item in data
        )
        event_category_counter = Counter(
            item["analysis_report"]["event_analysis"]["category_tag"] for item in data
        )

        # Flatten the list of lists for camera tags
        all_camera_tags = [
            tag
            for item in data
            for tag in item["analysis_report"]["camera_analysis"].get("tags", [])
        ]
        camera_tags_counter = Counter(all_camera_tags)

        # --- Cross-Analysis ---
        cross_analysis_stats = {
            transition: {
                "count": count,
                "camera_tags": Counter(),
                "event_category": Counter(),
            }
            for transition, count in transition_counter.items()
        }

        for item in data:
            report = item["analysis_report"]
            transition_type = report["transition_type"]

            # Update camera tags for this transition type
            camera_tags = report["camera_analysis"].get("tags", [])
            cross_analysis_stats[transition_type]["camera_tags"].update(camera_tags)

            # Update event category for this transition type
            event_category = report["event_analysis"]["category_tag"]
            cross_analysis_stats[transition_type]["event_category"].update(
                [event_category]
            )

        return (
            total_pairs,
            transition_counter,
            event_category_counter,
            camera_tags_counter,
            cross_analysis_stats,
        )

    def _display_results(
        self,
        total_pairs,
        transition_counter,
        event_category_counter,
        camera_tags_counter,
        cross_analysis_stats,
    ):
        """Displays the analysis results using rich components."""
        console.clear()

        # --- Title and Summary ---
        console.print(
            Panel(
                Align.center(
                    "[bold cyan]VLM Image Pair Transition Analysis Report[/bold cyan]"
                ),
                border_style="bold blue",
            )
        )
        console.print(
            Panel(
                f"Analyzed a total of [bold green]{total_pairs}[/bold green] image pairs.",
                title="[bold]📊 Overall Summary[/bold]",
                title_align="left",
            )
        )

        # --- Primary Classification Table ---
        transition_table = Table(
            title="[bold]▶️ Core Transition Relationship[/bold]",
            header_style="bold magenta",
        )
        transition_table.add_column("Transition Type", style="cyan", no_wrap=True)
        transition_table.add_column("Count", justify="right", style="magenta")
        transition_table.add_column("Percentage", justify="right", style="yellow")
        for t_type, count in transition_counter.most_common():
            percentage = (count / total_pairs) * 100
            transition_table.add_row(t_type, str(count), f"{percentage:.2f}%")
        console.print(transition_table)

        # --- Detailed Breakdowns using a Layout ---
        layout = Layout()
        layout.split_row(Layout(name="left"), Layout(name="right"))

        # Left Panel: Event Analysis
        event_table = Table(
            title="[bold]🎬 Scene Dynamics Category[/bold]", header_style="bold green"
        )
        event_table.add_column("Category", style="dim")
        event_table.add_column("Count", justify="right")
        event_table.add_column("Percentage", justify="right")
        for cat, count in event_category_counter.most_common():
            event_table.add_row(cat, str(count), f"{(count / total_pairs) * 100:.2f}%")
        layout["left"].update(Panel(event_table, border_style="green"))

        # Right Panel: Camera Dynamics
        camera_table = Table(
            title="[bold]📷 Camera Dynamics (Overall)[/bold]", header_style="bold green"
        )
        camera_table.add_column("Tag")
        camera_table.add_column("Occurrences", justify="right")
        # Note: A single pair can have multiple camera tags.
        for tag, count in camera_tags_counter.most_common(10):  # Show top 10
            camera_table.add_row(tag, str(count))
        layout["right"].update(
            Panel(
                camera_table, border_style="green", subtitle="Top 10 most common tags"
            )
        )

        console.print(layout)

        # --- Cross-Analysis Tree ---
        tree = Tree(
            "[bold gold1]🔎 Cross-Analysis: Tag Distribution per Transition Type[/bold gold1]",
            guide_style="cyan",
        )

        sorted_transitions = sorted(
            cross_analysis_stats.items(),
            key=lambda item: item[1]["count"],
            reverse=True,
        )

        for transition, stats in sorted_transitions:
            transition_count = stats["count"]
            branch = tree.add(f"[bold]{transition}[/bold] ({transition_count} pairs)")

            # Camera Tags Sub-branch
            cam_branch = branch.add("📷 [dim]Camera Dynamics[/dim]")
            for tag, count in stats["camera_tags"].most_common(5):
                percentage = (count / transition_count) * 100
                cam_branch.add(f"{tag}: {count} ({percentage:.1f}%)")

            # Event Category Sub-branch
            event_branch = branch.add("🎬 [dim]Scene Dynamics[/dim]")
            for cat, count in stats["event_category"].most_common():
                percentage = (count / transition_count) * 100
                event_branch.add(f"{cat}: {count} ({percentage:.1f}%)")

        console.print(
            Panel(
                tree,
                border_style="cyan",
                title="[bold]🔬 Sanity Check[/bold]",
                title_align="left",
            )
        )

    def run(
        self,
        db_path: str,
        name: str = "stepflow_movie_compare_Qwen3VL30BA3B-Image-Edit",
    ):
        """
        Executes the full analysis pipeline.

        :param db_path: Path to the DuckDB database file.
        :param name: The 'name' of the annotation set to analyze.
        """
        annotations = self._load_data_from_duckdb(db_path, name)
        if not annotations:
            console.print("[bold red]Could not load any data. Exiting.[/bold red]")
            return

        analysis_results = self._analyze_data(annotations)
        if analysis_results:
            self._display_results(*analysis_results)


if __name__ == "__main__":
    # --- Global Console Instance ---
    console = Console()

    fire.Fire(CompareAnalysisTool)
