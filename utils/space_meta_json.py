#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python /data/LoraPipeline/utils/space_meta_json.py  --img-root /data/LoraPipeline/output/copy200  --out-json /data/LoraPipeline/output/copy200_showcase_items.json
python /data/LoraPipeline/utils/space_meta_json.py  --img-root /mnt/jfs/flux_9grid  --out-json /data/LoraPipeline/output/flux_full.json
python /data/LoraPipeline/utils/space_meta_json.py  --img-root /mnt/jfs/flux_9grid  --out-json /data/LoraPipeline/output/flux_full.json
python /data/LoraPipeline/utils/space_meta_json.py  --img-root /mnt/jfs/9grid/illustrious_9grid     --out-json /data/LoraPipeline/output/illustrious_batch_1.json
python /data/LoraPipeline/utils/space_meta_json.py  --img-root /mnt/jfs/9grid/sxdxl_9grid  --out-json /data/LoraPipeline/output/sdxl_batch_1.json
python /data/LoraPipeline/utils/space_meta_json.py  --img-root /mnt/jfs/9grid/illustrious_9grid_new/illustrious_0111_triplets_subset --out-json /data/LoraPipeline/output/sdxl_batch_2.json
"""
import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]


def norm_exts(exts: List[str]) -> List[str]:
    out = []
    for e in exts:
        e = e.lower()
        if not e.startswith("."):
            e = "." + e
        out.append(e)
    return out


def abs_uri_no_leading_slash(p: Path) -> str:
    # 绝对路径 + 去掉最前面的 '/'python /data/LoraPipeline/utils/space_meta_json.py  --img-root /mnt/jfs/9grid/illustrious_9grid_new/illustrious_0111_triplets_subset --out-json /data/LoraPipeline/output/illustrious_batch_2.json

    ap = str(p.resolve())
    return ap.lstrip(os.sep)


def parse_item_and_kind(stem: str) -> Tuple[str, str]:
    """
    约定文件名形如：
      <item_id>_ours.jpg
      <item_id>_uno.jpg
      <item_id>_idpatch.jpg
      <item_id>_uniportrait.jpg
      <item_id>_omnigen.jpg
      <item_id>_ori.jpg
      <item_id>_ref_2.jpg  (注意 ref_2 是两段)

    返回 (item_id, kind)
    """
    m = re.match(r"^(?P<item>.+)_ref_(?P<n>\d+)$", stem)
    if m:
        return m.group("item"), f"ref_{m.group('n')}"

    if stem.endswith("_ori"):
        return stem[:-4], "ori"

    if "_" not in stem:
        # 兜底：整段当 item_id，kind 给个默认
        return stem, "unknown"

    item_id, kind = stem.rsplit("_", 1)
    return item_id, kind


def kind_to_answer_label(kind: str) -> str:
    # 按你模板常见的 A~E + 参考图命名（可自行扩展）
    mapping = {
        "ours": "A 生成的图像",
        "uno": "B 生成的图像",
        "idpatch": "C 生成的图像",
        "uniportrait": "D 生成的图像",
        "omnigen": "E 生成的图像",
        "ori": "原始参考图",
    }
    if kind in mapping:
        return mapping[kind]
    m = re.match(r"^ref_(\d+)$", kind)
    if m:
        return f"参考人物{m.group(1)}"
    return f"其他({kind})"


def sort_kind_key(kind: str) -> Tuple[int, int, str]:
    # 让输出 answers 顺序更稳定：ours/uno/idpatch/uniportrait/omnigen/ori/ref_2/ref_3...
    order = {
        "ours": 0,
        "uno": 1,
        "idpatch": 2,
        "uniportrait": 3,
        "omnigen": 4,
        "ori": 5,
    }
    if kind in order:
        return (order[kind], 0, kind)
    m = re.match(r"^ref_(\d+)$", kind)
    if m:
        return (6, int(m.group(1)), kind)
    return (99, 0, kind)


def main():
    ap = argparse.ArgumentParser(
        description="递归收集目录下所有图片，按 item_id 组织为模板 JSON；attrs 置空；urls/uris 使用绝对路径去掉首个/。"
    )
    ap.add_argument("--img-root", required=True, help="图片根目录（递归扫描）")
    ap.add_argument("--out-json", required=True, help="输出 JSON 路径")
    ap.add_argument("--exts", nargs="+", default=DEFAULT_EXTS, help="允许的图片扩展名")
    ap.add_argument("--human-question", default="", help="每个 item 的 Human.question（默认空串）")
    ap.add_argument(
        "--question-map",
        default="",
        help="可选：一个 JSON 文件，形如 {\"item_id\": \"question...\", ...}，用于给不同 item 填不同 question",
    )
    args = ap.parse_args()

    img_root = Path(args.img_root).expanduser().resolve()
    if not img_root.exists() or not img_root.is_dir():
        raise SystemExit(f"[ERROR] img-root 不是有效目录：{img_root}")

    exts = set(norm_exts(args.exts))

    question_map: Dict[str, str] = {}
    if args.question_map:
        qpath = Path(args.question_map).expanduser().resolve()
        with open(qpath, "r", encoding="utf-8") as f:
            question_map = json.load(f)

    # group[item_id][kind] = [uri1, uri2, ...]
    group: Dict[str, Dict[str, List[str]]] = {}

    for p in img_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue

        item_id, kind = parse_item_and_kind(p.stem)
        uri = abs_uri_no_leading_slash(p)

        group.setdefault(item_id, {}).setdefault(kind, []).append(uri)

    items = []
    for item_id in sorted(group.keys()):
        kind2uris = group[item_id]

        answers = []
        all_urls: List[str] = []

        for kind in sorted(kind2uris.keys(), key=sort_kind_key):
            urls = sorted(set(kind2uris[kind]))
            if not urls:
                continue

            answers.append(
                {
                    "answer": kind_to_answer_label(kind),
                    "from": "Candidate",
                    "urls": urls,
                }
            )
            all_urls.extend(urls)

        # uris 必须包含所有 answers.urls（去重）
        uris = sorted(set(all_urls))

        item = {
            "item_id": item_id,
            "uris": uris,
            "data": {
                "attrs": [],  # 按你要求：空
                "chat": [
                    {
                        "from": "Human",
                        "question": question_map.get(item_id, args.human_question),
                    },
                    {
                        "from": "Assistant",
                        "answers": answers,
                    },
                ],
            },
        }
        items.append(item)

    out = {"items": items}

    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[OK] items: {len(items)}")
    print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    main()
