import fire
import megfile
from loguru import logger
from openai import OpenAI
from tqdm import tqdm
from utils.annotate import annotate_single_turn, load_prompt_template
from utils.executor import Task, run_batch_concurrent

from vault.backend.duckdb import DuckDBHandler
from vault.schema import ID
from vault.schema.multimodal import Creator, MultiModalType
from vault.storage.lanceduck.multimodal import (
    MultiModalStorager,
    SampleAnnotation,
    convert_as_id,
)

OPENAI_BASE_URL: str = "http://stepcast-router.shai-core:9200/v1"
OPENAI_API_KEY: str = "EMPTY"


def get_processed_sample_ids(
    annotation_name: str, duckdb_handler: DuckDBHandler, sample_type: str
) -> set:
    if sample_type == "sequence":
        sql = "select sequence_id as id from sample_annotations where name = ?;"
    else:
        sql = f"""
        SELECT
            sae.element_id AS id
        FROM
            sample_annotation_elements AS sae
        JOIN
            sample_annotations AS sa ON sae.sample_annotation_id = sa.id
        WHERE
            sa.name = ? AND sae.element_type = '{sample_type}';
        """

    return {
        ID.from_(val["id"])
        for val in duckdb_handler.query_batch(sql, (annotation_name,))
    }


def fetch_image_bytes(image_ids: list[ID], storager: MultiModalStorager) -> list[bytes]:
    """Fetch image bytes by IDs in order."""
    image_bytes_dict = storager.get_image_bytes_by_ids(image_ids)
    return [image_bytes_dict[image_id] for image_id in image_ids]


def sample_to_sequence(_samples):
    from collections import defaultdict

    samples = defaultdict(list)
    for sample in _samples:
        samples[sample["sample_id"]].append(sample["sequence_id"])
    return samples


def tag_frame(
    vault_path: str = "/mnt/marmot/wangrui/stepflow/graph_video_wujingwei_world_model_data_1117/",
    annotation_name: str = "stepflow_v2_frame_analysis",
    model: str = "Qwen3-VL-235B-A22B-Instruct-image-edit-only",
    max_workers: int = 1024,
    batch_size: int = 32,
):
    full_annotation_name = f"{annotation_name}-{model}"

    # Initialize storager and client
    logger.info(f"Initializing Vault: {vault_path}")
    logger.info(f"Model: {model}")
    logger.info(f"Annotation name: {full_annotation_name}")
    logger.info(f"Max workers: {max_workers}")
    logger.info(f"Batch size: {batch_size}")

    storager = MultiModalStorager(vault_path, read_only=False)
    vlm_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    logger.info("Querying samples...")

    all_samples: dict[ID, list[ID]] = sample_to_sequence(
        convert_as_id(
            storager.meta_handler.query_batch(
                "SELECT image_id as sample_id, sequence_id FROM sequence_images;"
            )
        )
    )

    logger.info(f"Total samples: {len(all_samples)}")

    # Filter out already processed samples
    logger.info(
        f"Querying processed samples (annotation_name={full_annotation_name})..."
    )
    processed_sample_ids = get_processed_sample_ids(
        full_annotation_name, storager.meta_handler, "image"
    )
    sample_ids_to_process = [
        sample_id for sample_id in all_samples if sample_id not in processed_sample_ids
    ]
    logger.info(f"To process: {len(sample_ids_to_process)}")

    # Load prompt template
    prompt_template_path = megfile.smart_path_join(
        megfile.SmartPath(__file__).parent, "prompts/00_frame_tag.j2.md"
    )
    prompt = load_prompt_template(prompt_template_path).render()

    # Prepare tasks
    logger.info("Preparing tasks...")
    tasks = []
    for sample_id in tqdm(sample_ids_to_process, desc="Preparing tasks"):
        tasks.append(
            Task(
                input_data={
                    "image_id": sample_id,
                    "prompt": prompt,
                },
                context={
                    "participants": [
                        ((sample_id, MultiModalType.IMAGE, "image"),)
                        for _ in all_samples[sample_id]
                    ],
                    "sequence_ids": all_samples[sample_id],
                },
            )
        )

    logger.info(f"Successfully prepared {len(tasks)} tasks")

    # Define processing function
    def call_vlm_api(task_data: dict, max_retries: int = 2):
        """
        Call VLM API with schema validation and retry logic.

        Args:
            task_data: Task data containing image_id and prompt
            max_retries: Maximum number of retry attempts if validation fails

        Returns:
            Validated JSON response
        """
        image_bytes = fetch_image_bytes([task_data["image_id"]], storager)

        return annotate_single_turn(
            images=image_bytes,
            model=model,
            client=vlm_client,
            prompt=task_data["prompt"],
            is_json=True,
        )

    # Create annotation creator
    annotation_creator = Creator.create(
        name=full_annotation_name,
        meta={
            "model_name": model,
            "prompt": prompt,
        },
    )

    # Define batch save function
    def save_annotation_batch(results):
        """Save batch of annotation results to vault."""
        annotations = []
        for result in results:
            if result.success:
                for seq_id, participants in zip(
                    result.context["sequence_ids"], result.context["participants"]
                ):
                    annotations.append(
                        SampleAnnotation.create(
                            name=full_annotation_name,
                            creator=annotation_creator,
                            value=result.result,
                            sequence_id=seq_id,
                            participants=participants,
                        )
                    )

        if annotations:
            storager.add_sample_annotations(annotations)
            logger.debug(f"Saved {len(annotations)} annotations to {vault_path}")

    # Execute concurrent batch processing
    stats = run_batch_concurrent(
        tasks=tasks,
        process_func=call_vlm_api,
        save_func=save_annotation_batch,
        max_workers=max_workers,
        batch_size=batch_size,
    )

    logger.success(
        f"Annotation complete! Success rate: {stats.success / stats.total * 100:.1f}%"
    )


if __name__ == "__main__":
    fire.Fire(tag_frame)
