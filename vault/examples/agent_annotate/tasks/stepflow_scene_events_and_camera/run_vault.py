import json
from typing import Any

import fire
import megfile
import xxhash
from loguru import logger
from openai import OpenAI
from tqdm import tqdm
from utils.annotate import annotate_single_turn, load_prompt_template
from utils.conversation import Conversation
from utils.executor import Task, run_batch_concurrent
from utils.openai_client import call_openai, retry_on_error

from vault.schema import ID
from vault.schema.multimodal import Creator, MultiModalType
from vault.storage.lanceduck.multimodal import MultiModalStorager, SampleAnnotation

OPENAI_BASE_URL: str = "http://stepcast-router.shai-core:9200/v1"
OPENAI_API_KEY: str = "EMPTY"


def extract_source_target_participants(sample: dict):
    """Extract source and target image participants from sample."""
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

    participants = (
        (source_image_id, MultiModalType.IMAGE, "source"),
        (target_image_id, MultiModalType.IMAGE, "target"),
    )

    return participants


def fetch_image_bytes(image_ids: list[ID], storager: MultiModalStorager) -> list[bytes]:
    """Fetch image bytes by IDs in order."""
    image_bytes_dict = storager.get_image_bytes_by_ids(image_ids)
    return [image_bytes_dict[image_id] for image_id in image_ids]


def annotate_vault(
    vault_path: str = "/mnt/marmot/liaojie/ScreenMusings-251022",
    annotation_name: str = "stepflow_v2_scene_events_and_camera",
    model: str = "Qwen3VL235BBA22B-Image-Edit",
    max_workers: int = 128,
    batch_size: int = 8,
):
    """
    Annotate vault sequences with VLM comparison.

    Args:
        vault_path: Path to vault storage
        annotation_name: Base annotation name (model will be appended)
        model: VLM model name
        max_workers: Maximum concurrent workers
        batch_size: Batch size for saving annotations
    """
    full_annotation_name = f"{annotation_name}-{model}"

    # Initialize storager and client
    logger.info(f"Initializing Vault: {vault_path}")
    logger.info(f"Model: {model}")
    logger.info(f"Annotation name: {full_annotation_name}")
    logger.info(f"Max workers: {max_workers}")
    logger.info(f"Batch size: {batch_size}")

    storager = MultiModalStorager(vault_path, read_only=False)
    vlm_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    # Query all sequences
    logger.info("Querying sequences...")
    all_sequence_ids = [
        ID.from_(row["sequence_id"])
        for row in storager.meta_handler.query_batch(
            "SELECT id as sequence_id FROM sequences"
        )
    ]
    all_sequences = storager.get_sequence_metas(all_sequence_ids)
    logger.info(f"Total sequences: {len(all_sequences)}")

    # Filter out already processed sequences
    logger.info(
        f"Querying processed sequences (annotation_name={full_annotation_name})..."
    )
    processed_sequence_ids = {
        ID.from_(row["sequence_id"])
        for row in storager.meta_handler.query_batch(
            "SELECT sequence_id FROM sample_annotations WHERE name = ?",
            (full_annotation_name,),
        )
    }
    sequences_to_process = [
        seq for seq in all_sequences if seq["sequence_id"] not in processed_sequence_ids
    ]
    logger.info(f"To process: {len(sequences_to_process)}")

    if not sequences_to_process:
        logger.info("No sequences to process, exiting")
        return

    # Load prompt template
    prompt_template_path = megfile.smart_path_join(
        megfile.SmartPath(__file__).parent, "prompts/01_compare.j2"
    )
    prompt = load_prompt_template(prompt_template_path).render()

    # Prepare tasks
    logger.info("Preparing tasks...")
    tasks = []
    for sequence in tqdm(sequences_to_process, desc="Preparing tasks"):
        participants = extract_source_target_participants(sequence)
        if participants is None:
            continue

        source_image_id, target_image_id = participants[0][0], participants[1][0]

        tasks.append(
            Task(
                input_data={
                    "source_image_id": source_image_id,
                    "target_image_id": target_image_id,
                    "prompt": prompt,
                },
                context={
                    "participants": participants,
                    "sequence_id": sequence["sequence_id"],
                },
            )
        )

    logger.info(f"Successfully prepared {len(tasks)} tasks")

    # Define processing function
    def call_vlm_api(task_data: dict):
        """Call VLM API with image pair."""
        image_bytes_list = fetch_image_bytes(
            [task_data["source_image_id"], task_data["target_image_id"]], storager
        )
        return annotate_single_turn(
            images=image_bytes_list,
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
                annotations.append(
                    SampleAnnotation.create(
                        name=full_annotation_name,
                        creator=annotation_creator,
                        value=result.result,
                        sequence_id=result.context["sequence_id"],
                        participants=result.context["participants"],
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


def annotate_manual_selection_pairs(
    vault_path: str = "/mnt/marmot/liaojie/ScreenMusings-251022",
    filter_annotation_name: str = "manual_selection",
    annotation_name: str = "stepflow_v2_scene_events_and_camera",
    model: str = "Qwen3-VL-235B-A22B-Instruct-image-edit-only",
    output_dir: str = "s3://ruiwang/tmp/screen_musings_test/",
    max_workers: int = 128,
):
    # Initialize storager
    storager = MultiModalStorager(vault_path)

    # Query sequences with the specified annotation (both source and target have it)
    logger.info(f"Querying sequences with annotation: {filter_annotation_name}")
    with storager.meta_handler as handler:
        sequence_rows = handler.query_batch(
            """
            SELECT DISTINCT s.id as sequence_id
            FROM sequences s
            JOIN sequence_images si ON s.id = si.sequence_id
            JOIN image_annotations ia ON si.image_id = ia.image_id
            JOIN annotations a ON ia.annotation_id = a.id
            WHERE a.name = ?
            GROUP BY s.id
            HAVING COUNT(DISTINCT si.image_id) >= 2
            ORDER BY s.id
            """,
            [filter_annotation_name],
        )

    if not sequence_rows:
        logger.warning(f"No sequences found with annotation: {filter_annotation_name}")
        return

    target_sequence_ids = [ID.from_(row["sequence_id"]) for row in sequence_rows]
    logger.info(f"Found {len(target_sequence_ids)} sequences with annotation")

    # Get sequence metadata
    sequences = storager.get_sequence_metas(target_sequence_ids)
    vlm_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    logger.info(f"To process: {len(sequences)} sequences")

    if not sequences:
        logger.info("No sequences to process, exiting")
        return

    # Load prompt
    prompt_template_path = megfile.smart_path_join(
        megfile.SmartPath(__file__).parent, "prompts/01_compare.j2"
    )
    prompt = load_prompt_template(prompt_template_path).render()

    # Generate output folder name with prompt hash
    output_folder = (
        f"{annotation_name}-{xxhash.xxh32_hexdigest(prompt.encode())}-{model}"
    )

    # Prepare tasks
    logger.info("Preparing tasks...")
    tasks = []
    for sequence in tqdm(sequences, desc="Preparing tasks"):
        participants = extract_source_target_participants(sequence)
        if participants is None:
            continue

        source_image_id, target_image_id = participants[0][0], participants[1][0]

        tasks.append(
            Task(
                input_data={
                    "source_image_id": source_image_id,
                    "target_image_id": target_image_id,
                    "prompt": prompt,
                },
                context={
                    "sequence_id": sequence["sequence_id"],
                },
            )
        )

    logger.info(f"Successfully prepared {len(tasks)} tasks")

    # Define processing function
    def call_vlm_api(task_data: dict):
        """Call VLM API with image pair."""
        image_bytes_list = fetch_image_bytes(
            [task_data["source_image_id"], task_data["target_image_id"]], storager
        )
        return annotate_single_turn(
            images=image_bytes_list,
            model=model,
            client=vlm_client,
            prompt=task_data["prompt"],
            is_json=True,
            retry_count=100,
        )

    # Define save function (export to JSON files)
    def save_results_to_json(results):
        """Save results to individual JSON files."""
        for result in results:
            if result.success:
                output_path = megfile.smart_path_join(
                    output_dir, output_folder, f"{result.context['sequence_id']}.json"
                )
                with megfile.smart_open(output_path, "w") as f:
                    f.write(json.dumps(result.result, ensure_ascii=False, indent=2))

    # Execute concurrent batch processing
    stats = run_batch_concurrent(
        tasks=tasks,
        process_func=call_vlm_api,
        save_func=save_results_to_json,
        max_workers=max_workers,
        batch_size=1,  # Save immediately for JSON export
    )

    logger.success(
        f"Export complete! Success rate: {stats.success / stats.total * 100:.1f}%"
    )


def call_vlm_multi_turn(
    images: list[bytes],
    model: str,
    client: OpenAI,
    prompt_templates: dict[str, Any],
    max_tokens: int = 2048,
    max_pixels: int = 512 * 32 * 32,
    retry_count: int = 3,
) -> dict[str, Any]:
    """
    多轮对话调用 VLM API。

    Args:
        images: 图片字节列表 [source_image, target_image]
        model: 模型名称
        client: OpenAI 客户端
        prompt_templates: 各轮次的 prompt 模板字典，包含 step1-step5
        max_tokens: 最大 tokens
        max_pixels: 最大像素数
        retry_count: 重试次数

    Returns:
        包含所有轮次结果的字典
    """
    extra_body = {"chat_template_kwargs": {"add_vision_id": True}}
    results: dict[str, Any] = {}

    # === Round 1: 独立图像描述 ===
    conv = Conversation()
    conv.add_system("You are a helpful assistant.")
    prompt_step1 = prompt_templates["step1"].render()
    conv.add_user(text=prompt_step1, images=images, max_pixels=max_pixels)

    step1_result = retry_on_error(
        call_openai,
        messages=conv.get_messages(),
        model=model,
        client=client,
        max_tokens=max_tokens,
        extra_body=extra_body,
        is_json=True,
        retry_count=retry_count,
    )
    results["step1_captions_and_differences"] = step1_result
    conv.add_assistant(json.dumps(step1_result, ensure_ascii=False))

    # === Round 2: 场景一致性侦查 ===
    prompt_step2 = prompt_templates["step2"].render()
    conv.add_user(text=prompt_step2)

    step2_result = retry_on_error(
        call_openai,
        messages=conv.get_messages(),
        model=model,
        client=client,
        max_tokens=max_tokens,
        extra_body=extra_body,
        is_json=True,
        retry_count=retry_count,
    )
    results["step2_scene_consistency"] = step2_result
    conv.add_assistant(json.dumps(step2_result, ensure_ascii=False))

    # === Round 3: 并行差异分析 ===
    prompt_step3 = prompt_templates["step3"].render()
    conv.add_user(text=prompt_step3)

    step3_result = retry_on_error(
        call_openai,
        messages=conv.get_messages(),
        model=model,
        client=client,
        max_tokens=max_tokens,
        extra_body=extra_body,
        is_json=True,
        retry_count=retry_count,
    )
    results["step3_parallel_difference_analysis"] = step3_result
    conv.add_assistant(json.dumps(step3_result, ensure_ascii=False))

    # === Round 4: 变化归因与分类 ===
    prompt_step4 = prompt_templates["step4"].render()
    conv.add_user(text=prompt_step4)

    step4_result = retry_on_error(
        call_openai,
        messages=conv.get_messages(),
        model=model,
        client=client,
        max_tokens=max_tokens,
        extra_body=extra_body,
        is_json=True,
        retry_count=retry_count,
    )
    results["step4_change_attribution_and_classification"] = step4_result
    conv.add_assistant(json.dumps(step4_result, ensure_ascii=False))

    # === Round 5: 元评估 ===
    prompt_step5 = prompt_templates["step5"].render()
    conv.add_user(text=prompt_step5)

    step5_result = retry_on_error(
        call_openai,
        messages=conv.get_messages(),
        model=model,
        client=client,
        max_tokens=max_tokens,
        extra_body=extra_body,
        is_json=True,
        retry_count=retry_count,
    )
    results["step5_analysis_meta_evaluation"] = step5_result

    return results


def annotate_manual_selection_pairs_multi_turn(
    vault_path: str = "/mnt/marmot/liaojie/ScreenMusings-251022",
    filter_annotation_name: str = "manual_selection",
    annotation_name: str = "stepflow_v2_scene_events_and_camera_multi_turn",
    model: str = "Qwen3-VL-235B-A22B-Instruct-image-edit-only",
    output_dir: str = "s3://ruiwang/tmp/screen_musings_test/",
    max_workers: int = 128,
):
    """
    使用多轮对话标注手工挑选的图像对。

    多轮对话将分析过程拆分为5个步骤：
    1. 独立图像描述
    2. 场景一致性侦查
    3. 并行差异分析
    4. 变化归因与分类
    5. 元评估

    Args:
        vault_path: Vault 存储路径
        filter_annotation_name: 过滤用的标注名称
        annotation_name: 输出标注名称
        model: VLM 模型名称
        output_dir: 输出目录
        max_workers: 最大并发数
    """
    # Initialize storager
    storager = MultiModalStorager(vault_path)

    # Query sequences with the specified annotation
    logger.info(f"Querying sequences with annotation: {filter_annotation_name}")
    with storager.meta_handler as handler:
        sequence_rows = handler.query_batch(
            """
            SELECT DISTINCT s.id as sequence_id
            FROM sequences s
            JOIN sequence_images si ON s.id = si.sequence_id
            JOIN image_annotations ia ON si.image_id = ia.image_id
            JOIN annotations a ON ia.annotation_id = a.id
            WHERE a.name = ?
            GROUP BY s.id
            HAVING COUNT(DISTINCT si.image_id) >= 2
            ORDER BY s.id
            """,
            [filter_annotation_name],
        )

    if not sequence_rows:
        logger.warning(f"No sequences found with annotation: {filter_annotation_name}")
        return

    target_sequence_ids = [ID.from_(row["sequence_id"]) for row in sequence_rows]
    logger.info(f"Found {len(target_sequence_ids)} sequences with annotation")

    # Get sequence metadata
    sequences = storager.get_sequence_metas(target_sequence_ids)
    vlm_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    logger.info(f"To process: {len(sequences)} sequences")

    if not sequences:
        logger.info("No sequences to process, exiting")
        return

    # Load prompt templates for multi-turn
    prompts_dir = megfile.smart_path_join(
        megfile.SmartPath(__file__).parent, "prompts/multi_turn"
    )
    prompt_templates = {
        "step1": load_prompt_template(
            megfile.smart_path_join(prompts_dir, "step1_captions.j2.md")
        ),
        "step2": load_prompt_template(
            megfile.smart_path_join(prompts_dir, "step2_scene_consistency.j2.md")
        ),
        "step3": load_prompt_template(
            megfile.smart_path_join(prompts_dir, "step3_parallel_analysis.j2.md")
        ),
        "step4": load_prompt_template(
            megfile.smart_path_join(prompts_dir, "step4_classification.j2.md")
        ),
        "step5": load_prompt_template(
            megfile.smart_path_join(prompts_dir, "step5_meta_evaluation.j2.md")
        ),
    }

    # Generate output folder name with combined prompt hash
    combined_prompts = "".join(
        t.render()
        for t in [prompt_templates["step1"]]  # Use step1 for hash
    )
    output_folder = (
        f"{annotation_name}-{xxhash.xxh32_hexdigest(combined_prompts.encode())}-{model}"
    )

    # Prepare tasks
    logger.info("Preparing tasks...")
    tasks = []
    for sequence in tqdm(sequences, desc="Preparing tasks"):
        participants = extract_source_target_participants(sequence)
        if participants is None:
            continue

        source_image_id, target_image_id = participants[0][0], participants[1][0]

        tasks.append(
            Task(
                input_data={
                    "source_image_id": source_image_id,
                    "target_image_id": target_image_id,
                    "prompt_templates": prompt_templates,
                },
                context={
                    "sequence_id": sequence["sequence_id"],
                },
            )
        )

    logger.info(f"Successfully prepared {len(tasks)} tasks")

    # Define processing function for multi-turn
    def call_vlm_api_multi_turn(task_data: dict):
        """Call VLM API with multi-turn conversation."""
        image_bytes_list = fetch_image_bytes(
            [task_data["source_image_id"], task_data["target_image_id"]], storager
        )
        return call_vlm_multi_turn(
            images=image_bytes_list,
            model=model,
            client=vlm_client,
            prompt_templates=task_data["prompt_templates"],
            retry_count=100,
        )

    # Define save function (export to JSON files)
    def save_results_to_json(results):
        """Save results to individual JSON files."""
        for result in results:
            if result.success:
                output_path = megfile.smart_path_join(
                    output_dir, output_folder, f"{result.context['sequence_id']}.json"
                )
                with megfile.smart_open(output_path, "w") as f:
                    f.write(json.dumps(result.result, ensure_ascii=False, indent=2))

    # Execute concurrent batch processing
    stats = run_batch_concurrent(
        tasks=tasks,
        process_func=call_vlm_api_multi_turn,
        save_func=save_results_to_json,
        max_workers=max_workers,
        batch_size=1,
    )

    logger.success(
        f"Multi-turn export complete! Success rate: {stats.success / stats.total * 100:.1f}%"
    )


if __name__ == "__main__":
    fire.Fire(
        {
            "full": annotate_vault,
            "manual": annotate_manual_selection_pairs,
            "manual_multi_turn": annotate_manual_selection_pairs_multi_turn,
        }
    )
