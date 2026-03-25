"""
flux style 
python /data/benchmark_metrics/lora_pipeline/tools/remap_selection_paths.py \
  --input-jsonl /data/benchmark_metrics/logs/flux_content.jsonl \
  --output-jsonl /data/benchmark_metrics/logs/selections_with_origin_content_flux.jsonl \
  --eval-root s3://lanjinghong-data/loras_eval_flux_debug_1226 \
  --one-lora-root /mnt/jfs/loras_combine/flux_0321_one_lora \
  --eval-subfolder eval_images_with_negative \
  --one-subfolder eval_images_with_negative_new

python /data/benchmark_metrics/lora_pipeline/tools/remap_selection_paths.py \
  --input-jsonl /data/benchmark_metrics/logs/selections.jsonl \
  --output-jsonl /data/benchmark_metrics/logs/selections_with_origin_style_flux0325.jsonl \
  --eval-root s3://lanjinghong-data/loras_eval_flux_debug_1226 \
  --one-lora-root /mnt/jfs/loras_combine/flux_0321_one_lora \
  --eval-subfolder eval_images_with_negative \
  --one-subfolder eval_images_with_negative_new
"""
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from megfile import smart_exists, smart_listdir


IMG_RE = re.compile(r"^(\d{5})_(\d+)\.(png|jpg|jpeg|webp|bmp)$", re.IGNORECASE)
SLOT_RE = re.compile(r"^(eval|one_lora)_(\d+)$")


def smart_join(*parts: str) -> str:
    cleaned = [str(x).strip("/") for x in parts if str(x)]
    if not cleaned:
        return ""
    head = str(parts[0])
    if head.startswith("s3://"):
        base = "s3://" + cleaned[0].replace("s3://", "").strip("/")
        tail = cleaned[1:]
        return "/".join([base] + tail)
    if str(parts[0]).startswith("/"):
        return "/" + "/".join(cleaned)
    return "/".join(cleaned)


def list_indexed_images(folder: str) -> List[str]:
    if not smart_exists(folder):
        return []
    files: List[Tuple[int, int, str]] = []
    for name in smart_listdir(folder):
        bn = str(name).split("/")[-1]
        m = IMG_RE.match(bn)
        if not m:
            continue
        p = smart_join(folder, bn)
        files.append((int(m.group(1)), int(m.group(2)), p))
    files.sort(key=lambda x: (x[0], x[1], x[2]))
    return [x[2] for x in files]


def resolve_origin_path(
    copied_path: str,
    model_id: str,
    eval_root: str,
    one_lora_root: str,
    eval_subfolder: str,
    one_subfolder: str,
    cache: Dict[Tuple[str, str], List[str]],
) -> str:
    cp = str(copied_path)
    slot = cp.rstrip("/").split("/")[-2]
    m = SLOT_RE.match(slot)
    if not m:
        raise RuntimeError(f"无法识别目标槽位目录: {slot}")
    kind = m.group(1)
    idx = int(m.group(2))
    if idx <= 0:
        raise RuntimeError(f"槽位编号非法: {slot}")
    src_root = eval_root if kind == "eval" else one_lora_root
    sub = eval_subfolder if kind == "eval" else one_subfolder
    key = (kind, model_id)
    if key not in cache:
        src_dir = smart_join(src_root, model_id, sub)
        cache[key] = list_indexed_images(src_dir)
    imgs = cache[key]
    pos = idx - 1
    if pos >= len(imgs):
        raise RuntimeError(f"源目录图片不足: slot={slot}, model_id={model_id}, available={len(imgs)}")
    origin = imgs[pos]
    if not smart_exists(origin):
        raise RuntimeError(f"映射到的源图不存在: {origin}")
    return origin


def process_jsonl(
    input_jsonl: Path,
    output_jsonl: Path,
    eval_root: str,
    one_lora_root: str,
    eval_subfolder: str,
    one_subfolder: str,
    max_lines: int,
) -> Dict[str, int]:
    cache: Dict[Tuple[str, str], List[str]] = {}
    line_count = 0
    path_count = 0
    fail_count = 0

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with input_jsonl.open("r", encoding="utf-8") as fin, output_jsonl.open("w", encoding="utf-8") as fout:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            if max_lines > 0 and line_count >= max_lines:
                break
            line_count += 1
            row = json.loads(s)
            new_row = {}
            for model_id, paths in row.items():
                mapped = []
                for p in paths:
                    path_count += 1
                    try:
                        origin = resolve_origin_path(
                            copied_path=str(p),
                            model_id=str(model_id),
                            eval_root=eval_root,
                            one_lora_root=one_lora_root,
                            eval_subfolder=eval_subfolder,
                            one_subfolder=one_subfolder,
                            cache=cache,
                        )
                        mapped.append(str(origin))
                    except Exception as e:
                        fail_count += 1
                        print(f"[WARN] model_id={model_id} copied={p} err={e}")
                new_row[str(model_id)] = mapped
            fout.write(json.dumps(new_row, ensure_ascii=False) + "\n")
    return {"lines": line_count, "paths": path_count, "failed": fail_count}


def main():
    parser = argparse.ArgumentParser(description="把观察用拷贝图路径回溯成原始图片路径，并输出新 jsonl")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--one-lora-root", required=True)
    parser.add_argument("--eval-subfolder", default="eval_images_with_negative")
    parser.add_argument("--one-subfolder", default="eval_images_with_negative_new")
    parser.add_argument("--max-lines", type=int, default=0, help="仅处理前N行，0为全量")
    args = parser.parse_args()

    stats = process_jsonl(
        input_jsonl=Path(args.input_jsonl),
        output_jsonl=Path(args.output_jsonl),
        eval_root=str(args.eval_root),
        one_lora_root=str(args.one_lora_root),
        eval_subfolder=args.eval_subfolder.strip("/"),
        one_subfolder=args.one_subfolder.strip("/"),
        max_lines=int(args.max_lines),
    )
    print(
        f"[DONE] lines={stats['lines']} paths={stats['paths']} failed={stats['failed']} "
        f"output={args.output_jsonl}"
    )


if __name__ == "__main__":
    main()
