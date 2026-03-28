"""
python /data/benchmark_metrics/lora_pipeline/tools/find_unfinished_pairs.py \
  --pair-model-id-txt /data/benchmark_metrics/lora_pipeline/meta/model_ids/illustrious_style_and_content.txt \
  --output-root /mnt/jfs/loras_combine/illustrious_0323_dual_lora_diverse_unique \
  --num-prompts 10 \
  --out-txt /data/benchmark_metrics/lora_pipeline/meta/model_ids/illustrious_unfinished.txt \
  --stats-json /data/benchmark_metrics/lora_pipeline/meta/model_ids/illustrious_unfinished_stats.json \
  --workers 256
python /data/benchmark_metrics/lora_pipeline/tools/find_unfinished_pairs.py \
  --pair-model-id-txt /data/benchmark_metrics/logs/triplet_style_firsthit_judge_0325_0.5_2/style_firsthit_non_empty_keys.txt \
  --output-root /mnt/jfs/loras_combine/flux_0323_dual_lora_diverse_save_prompt \
  --num-prompts 10 \
  --out-txt /data/benchmark_metrics/lora_pipeline/meta/model_ids/flux_unfinished.txt \
  --workers 256
"""
import argparse
import concurrent.futures
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_PIPELINE_DIR = os.path.dirname(CUR_DIR)
if LORA_PIPELINE_DIR not in sys.path:
    sys.path.insert(0, LORA_PIPELINE_DIR)

import illustrious_one_lora_diverse as base

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_pair_model_ids(path: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    seen = set()
    with base.mopen(path, "r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s or s.startswith("#"):
                continue
            m = re.match(r"(\d+)\s*__\s*(\d+)", s)
            if not m:
                continue
            pair = (m.group(1), m.group(2))
            key = f"{pair[0]}__{pair[1]}"
            if key in seen:
                continue
            seen.add(key)
            out.append(pair)
    return out


def _is_valid_image(path: str) -> bool:
    try:
        size = os.path.getsize(path)
        if size <= 0:
            return False
        ext = os.path.splitext(path)[1].lower()
        with open(path, "rb") as f:
            head = f.read(64)
            if size >= 64:
                f.seek(max(0, size - 64))
                tail = f.read(64)
            else:
                tail = b""
        if ext == ".png":
            return head.startswith(b"\x89PNG\r\n\x1a\n") and (b"IEND" in tail)
        if ext in {".jpg", ".jpeg"}:
            if len(head) < 2 or len(tail) < 2:
                return False
            return head[0:2] == b"\xff\xd8" and tail[-2:] == b"\xff\xd9"
        if ext == ".webp":
            if len(head) < 12:
                return False
            return head[0:4] == b"RIFF" and head[8:12] == b"WEBP"
        if ext == ".bmp":
            return len(head) >= 2 and head[0:2] == b"BM" and size >= 54
        return True
    except Exception:
        return False


def _cleanup_bad_images(eval_dir: str) -> int:
    if base.is_remote_path(eval_dir):
        return 0
    if not os.path.isdir(eval_dir):
        return 0
    removed = 0
    try:
        names = os.listdir(eval_dir)
    except Exception:
        return 0
    for name in names:
        path = os.path.join(eval_dir, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in IMAGE_EXTS:
            continue
        if _is_valid_image(path):
            continue
        try:
            os.remove(path)
            removed += 1
        except Exception:
            pass
    return removed


def _process_pair(
    pair_id: str,
    target: int,
    output_root: str,
    output_subdir: Optional[str],
    eval_dir_name: str,
    delete_bad_images: bool,
) -> Dict[str, object]:
    pair_dir = base.join_path(output_root.rstrip("/"), pair_id)
    if output_subdir:
        pair_dir = base.join_path(pair_dir, output_subdir)
    eval_dir = base.join_path(pair_dir, eval_dir_name)
    bad_removed = _cleanup_bad_images(eval_dir) if delete_bad_images else 0
    done = len(base.scan_done_prompt_indices(eval_dir))
    remaining = max(0, int(target) - int(done))
    return {
        "pair_id": pair_id,
        "target_num_prompts": int(target),
        "done_num_prompts": int(done),
        "remaining_num_prompts": int(remaining),
        "eval_dir": eval_dir,
        "removed_bad_images": int(bad_removed),
    }


def main():
    parser = argparse.ArgumentParser(description="查找未完成的 pair model_id 并输出 txt")
    parser.add_argument("--pair-model-id-txt", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--output-subdir", default=None)
    parser.add_argument("--eval-dir-name", default="eval_images_with_negative_new")
    parser.add_argument("--out-txt", default="")
    parser.add_argument("--stats-json", default="")
    parser.add_argument("--sort-by", choices=["none", "remaining_desc", "remaining_asc"], default="remaining_desc")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1)))
    parser.add_argument("--delete-bad-images", action="store_true")
    parser.add_argument("--stream-output", action="store_true")
    args = parser.parse_args()

    pairs = parse_pair_model_ids(args.pair_model_id_txt)

    pair_targets: List[Tuple[str, int]] = []
    for cid, sid in pairs:
        pair_id = f"{cid}__{sid}"
        target = max(0, int(args.num_prompts))
        pair_targets.append((pair_id, int(target)))

    rows: List[Dict[str, object]] = []
    out_fp = None
    if args.out_txt:
        out_dir = os.path.dirname(os.path.abspath(args.out_txt))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_fp = open(args.out_txt, "w", encoding="utf-8")
    workers = max(1, int(args.workers))
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _process_pair,
                    pair_id,
                    target,
                    args.output_root,
                    args.output_subdir,
                    args.eval_dir_name,
                    bool(args.delete_bad_images),
                )
                for pair_id, target in pair_targets
            ]
            for fut in concurrent.futures.as_completed(futures):
                row = fut.result()
                rows.append(row)
                if int(row["remaining_num_prompts"]) <= 0:
                    continue
                if args.stream_output:
                    print(f"{row['pair_id']}\tremaining={row['remaining_num_prompts']}\tdone={row['done_num_prompts']}\ttarget={row['target_num_prompts']}")
                if out_fp is not None:
                    out_fp.write(str(row["pair_id"]) + "\n")
    finally:
        if out_fp is not None:
            out_fp.close()

    unfinished = [x for x in rows if int(x["remaining_num_prompts"]) > 0]
    if args.sort_by == "remaining_desc":
        unfinished.sort(key=lambda x: (int(x["remaining_num_prompts"]), x["pair_id"]), reverse=True)
    elif args.sort_by == "remaining_asc":
        unfinished.sort(key=lambda x: (int(x["remaining_num_prompts"]), x["pair_id"]))

    if args.out_txt and args.sort_by != "none":
        with open(args.out_txt, "w", encoding="utf-8") as f:
            for row in unfinished:
                f.write(str(row["pair_id"]) + "\n")

    if args.stats_json:
        stats_dir = os.path.dirname(os.path.abspath(args.stats_json))
        if stats_dir:
            os.makedirs(stats_dir, exist_ok=True)
        with open(args.stats_json, "w", encoding="utf-8") as f:
            json.dump(unfinished, f, ensure_ascii=False, indent=2)

    print(f"total_pairs={len(rows)}")
    print(f"unfinished_pairs={len(unfinished)}")
    print(f"finished_pairs={max(0, len(rows) - len(unfinished))}")
    print(f"removed_bad_images={sum(int(x.get('removed_bad_images', 0)) for x in rows)}")
    if args.out_txt:
        print(f"out_txt={args.out_txt}")
    if args.stats_json:
        print(f"stats_json={args.stats_json}")


if __name__ == "__main__":
    main()
