#!/usr/bin/env python3
"""
python3 /data/benchmark_metrics/utils/build_triplet_jsonl_from_key_list.py \
  --keys-txt /data/benchmark_metrics/lora_pipeline/meta/model_ids/flux_content_sample__x__selections_with_origin_style_flux0325_keys.txt \
  --image-root /mnt/jfs/loras_combine/flux_0215 \
  --out-jsonl /data/benchmark_metrics/logs/triplets_style_and_content_only_from_flux0215.jsonl \
  --missing-txt /data/benchmark_metrics/logs/triplets_style_and_content_only_from_flux0215_missing.txt \
  --preferred-name 000_0.png \
  --preferred-name 00000_0.png \
  --preferred-name 000_0.jpg \
  --preferred-name 00000_0.jpg
"""
import argparse
import json
from pathlib import Path
from typing import List


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--keys-txt", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--missing-txt", required=True)
    p.add_argument("--preferred-name", action="append", default=[])
    p.add_argument("--ext", action="append", default=[".png", ".jpg", ".jpeg", ".webp", ".bmp"])
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--progress-every", type=int, default=20000)
    return p.parse_args()


def read_keys(path: Path) -> List[str]:
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def pick_image(key_dir: Path, preferred: List[str], exts: List[str], recursive: bool) -> Path | None:
    if not key_dir.is_dir():
        return None
    for name in preferred:
        p = key_dir / name
        if p.is_file():
            return p
    ext_set = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}
    it = key_dir.rglob("*") if recursive else key_dir.iterdir()
    for p in sorted(it):
        if p.is_file() and p.suffix.lower() in ext_set:
            return p
    return None


def verify_jsonl(path: Path) -> tuple[bool, int]:
    lines = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            lines += 1
            obj = json.loads(s)
            if not isinstance(obj, dict) or len(obj) != 1:
                return False, lines
            _k, v = next(iter(obj.items()))
            if not (isinstance(v, list) and len(v) == 1 and isinstance(v[0], str) and Path(v[0]).is_file()):
                return False, lines
    return True, lines


def main():
    args = parse_args()
    keys_txt = Path(args.keys_txt)
    image_root = Path(args.image_root)
    out_jsonl = Path(args.out_jsonl)
    missing_txt = Path(args.missing_txt)
    if not keys_txt.is_file():
        raise RuntimeError(f"keys文件不存在: {keys_txt}")
    if not image_root.is_dir():
        raise RuntimeError(f"图片根目录不存在: {image_root}")
    preferred = args.preferred_name or ["000_0.png", "00000_0.png", "000_0.jpg", "00000_0.jpg"]
    keys = read_keys(keys_txt)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    missing_txt.parent.mkdir(parents=True, exist_ok=True)
    found = 0
    missing: List[str] = []
    with out_jsonl.open("w", encoding="utf-8") as fout:
        for i, k in enumerate(keys, 1):
            p = pick_image(image_root / k, preferred, args.ext, args.recursive)
            if p is not None and p.is_file():
                fout.write(json.dumps({k: [str(p.resolve())]}, ensure_ascii=False) + "\n")
                found += 1
            else:
                missing.append(k)
            if args.progress_every > 0 and i % args.progress_every == 0:
                print(f"progress {i}/{len(keys)} found={found} missing={len(missing)}", flush=True)
    missing_txt.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
    ok, lines = verify_jsonl(out_jsonl)
    print(f"keys_total={len(keys)}")
    print(f"found={found}")
    print(f"missing={len(missing)}")
    print(f"out_jsonl={out_jsonl}")
    print(f"missing_txt={missing_txt}")
    print(f"verify_paths_exists={ok}")
    print(f"written_lines={lines}")


if __name__ == "__main__":
    main()
