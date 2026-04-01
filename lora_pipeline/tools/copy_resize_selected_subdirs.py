#!/usr/bin/env python3
"""
python3 /data/benchmark_metrics/utils/copy_resize_selected_subdirs.py \
  --src-root /mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content \
  --dst-root /mnt/jfs/lora_combine/logs/sample_800_sref_200_content_jpg1024 \
  --subdir cref \
  --subdir sref \
  --subdir qwen-edit \
  --long-edge 1024 \
  --quality 75 \
  --workers 64

python3 /data/benchmark_metrics/utils/copy_resize_selected_subdirs.py \
  --src-root /mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content \
  --dst-root /mnt/jfs/lora_combine/logs/sample_800_sref_200_content_jpg1024_flux2 \
  --subdir cref \
  --subdir sref \
  --subdir newnew800_flux_9b \
  --long-edge 1024 \
  --quality 75 \
  --workers 64

python3 /data/benchmark_metrics/utils/copy_resize_selected_subdirs.py \
  --src-root /mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content \
  --dst-root /mnt/jfs/lora_combine/logs/sample_800_sref_200_content_jpg1024_ours \
  --subdir cref \
  --subdir sref \
  --subdir ours \
  --long-edge 1024 \
  --quality 75 \
  --workers 64
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Set

from PIL import Image, ImageOps


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src-root", required=True)
    p.add_argument("--dst-root", required=True)
    p.add_argument("--subdir", action="append", default=[])
    p.add_argument("--subdir-file", default="")
    p.add_argument("--long-edge", type=int, default=1024)
    p.add_argument("--quality", type=int, default=85)
    p.add_argument("--recursive", action="store_true", default=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--progress-every", type=int, default=1000)
    return p.parse_args()


def load_subdirs(args) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for x in args.subdir:
        s = str(x).strip().strip("/")
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    if args.subdir_file:
        p = Path(args.subdir_file)
        if not p.is_file():
            raise RuntimeError(f"subdir-file 不存在: {p}")
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip().strip("/")
            if s and s not in seen:
                seen.add(s)
                out.append(s)
    if not out:
        raise RuntimeError("至少提供一个 --subdir 或 --subdir-file")
    return out


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def resize_keep_ratio(img: Image.Image, long_edge: int) -> Image.Image:
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    m = max(w, h)
    if m <= long_edge:
        return img
    scale = float(long_edge) / float(m)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def convert_one(src: Path, dst: Path, long_edge: int, quality: int, overwrite: bool) -> bool:
    if dst.exists() and not overwrite:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode not in ("RGB",):
            im = im.convert("RGB")
        im = resize_keep_ratio(im, long_edge=long_edge)
        im.save(dst, format="JPEG", quality=quality, optimize=True)
    return True


def collect_jobs(src_root: Path, dst_root: Path, subdirs: List[str], recursive: bool) -> List[tuple[Path, Path]]:
    jobs: List[tuple[Path, Path]] = []
    for sub in subdirs:
        src_sub = src_root / sub
        if not src_sub.is_dir():
            print(f"[SKIP_SUBDIR] not_found={src_sub}")
            continue
        iterator = src_sub.rglob("*") if recursive else src_sub.iterdir()
        for p in iterator:
            if not p.is_file() or not is_image(p):
                continue
            rel = p.relative_to(src_root)
            dst = (dst_root / rel).with_suffix(".jpg")
            jobs.append((p, dst))
    return jobs


def run_one_job(src: Path, dst: Path, long_edge: int, quality: int, overwrite: bool) -> str:
    try:
        wrote = convert_one(src=src, dst=dst, long_edge=long_edge, quality=quality, overwrite=overwrite)
        return "written" if wrote else "skipped"
    except Exception:
        return "failed"


def main():
    args = parse_args()
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    if not src_root.is_dir():
        raise RuntimeError(f"src-root 不存在: {src_root}")
    if not (1 <= int(args.quality) <= 100):
        raise RuntimeError("quality 必须在 1~100")
    if int(args.long_edge) <= 0:
        raise RuntimeError("long-edge 必须 > 0")
    if int(args.workers) <= 0:
        raise RuntimeError("workers 必须 > 0")
    subdirs = load_subdirs(args)
    jobs = collect_jobs(src_root=src_root, dst_root=dst_root, subdirs=subdirs, recursive=bool(args.recursive))
    total_seen = len(jobs)
    total_written = 0
    total_skipped = 0
    total_failed = 0
    done = 0
    with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futures = [
            ex.submit(
                run_one_job,
                src=src,
                dst=dst,
                long_edge=int(args.long_edge),
                quality=int(args.quality),
                overwrite=bool(args.overwrite),
            )
            for src, dst in jobs
        ]
        for fut in as_completed(futures):
            status = fut.result()
            done += 1
            if status == "written":
                total_written += 1
            elif status == "skipped":
                total_skipped += 1
            else:
                total_failed += 1
            if args.progress_every > 0 and done % int(args.progress_every) == 0:
                print(
                    f"progress seen={total_seen} done={done} written={total_written} skipped={total_skipped} failed={total_failed}",
                    flush=True,
                )
    print(f"src_root={src_root}")
    print(f"dst_root={dst_root}")
    print(f"subdirs={len(subdirs)}")
    print(f"workers={int(args.workers)}")
    print(f"seen={total_seen}")
    print(f"done={done}")
    print(f"written={total_written}")
    print(f"skipped={total_skipped}")
    print(f"failed={total_failed}")


if __name__ == "__main__":
    main()
