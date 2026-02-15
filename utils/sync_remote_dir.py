import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, List, Optional, Tuple
"""
python3 /data/LoraPipeline/utils/sync_remote_dir.py \
  --src s3://lanjinghong-data/loras_eval_flux_debug_1226 \
  --dst /mnt/jfs/loras_combine/flux_0111 \
  --workers 32

"""
from megfile.smart import smart_listdir, smart_exists, smart_copy

try:
    from megfile.smart import smart_isdir  # type: ignore
except Exception:
    smart_isdir = None


def norm_root(p: str) -> str:
    return p.rstrip("/")


def join_path(root: str, name: str) -> str:
    if root.endswith("/"):
        return root + name.lstrip("/")
    return root + "/" + name.lstrip("/")


def is_dir_path(name_or_path: str) -> bool:
    return str(name_or_path).endswith("/")


def iter_files_smart(root: str) -> Iterable[str]:
    root = norm_root(root) + "/"
    stack = [root]
    while stack:
        cur = stack.pop()
        try:
            names = smart_listdir(cur)
        except Exception:
            continue
        for n in names:
            n = str(n)
            p = join_path(cur, n)
            is_dir = False
            if smart_isdir is not None:
                try:
                    is_dir = bool(smart_isdir(p))
                except Exception:
                    is_dir = False
            else:
                is_dir = is_dir_path(n)
            if is_dir:
                stack.append(norm_root(p) + "/")
            else:
                yield p


def strip_prefix(path: str, prefix: str) -> str:
    prefix = norm_root(prefix) + "/"
    if path.startswith(prefix):
        return path[len(prefix) :]
    return os.path.basename(path)


def copy_one(src_path: str, src_root: str, dst_root: str, overwrite: bool) -> Tuple[bool, str]:
    rel = strip_prefix(src_path, src_root).replace("\\", "/")
    dst_path = os.path.join(dst_root, rel)
    if (not overwrite) and os.path.exists(dst_path):
        return False, dst_path
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    smart_copy(src_path, dst_path)
    return True, dst_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="远程目录，例如 s3://bucket/prefix")
    ap.add_argument("--dst", required=True, help="本地目录，例如 /mnt/jfs/xxx")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-progress", action="store_true", help="禁用进度显示")
    args = ap.parse_args()

    src_root = norm_root(args.src)
    dst_root = os.path.abspath(args.dst)
    os.makedirs(dst_root, exist_ok=True)

    print(f"[INFO] scanning remote files: {src_root}")
    files = list(iter_files_smart(src_root))
    if not files:
        print(f"[ERR] no files under: {src_root}")
        raise SystemExit(2)
    print(f"[INFO] scanned files={len(files)} dst={dst_root}")

    copied = 0
    skipped = 0
    failed = 0
    done = 0

    pbar = None
    if not args.no_progress:
        try:
            from tqdm import tqdm  # type: ignore
            pbar = tqdm(total=len(files), desc="Copy", unit="file")
        except Exception:
            pbar = None

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = [ex.submit(copy_one, p, src_root, dst_root, bool(args.overwrite)) for p in files]
        for fut in as_completed(futs):
            try:
                did, _ = fut.result()
            except Exception:
                failed += 1
                did = None
            if did is True:
                copied += 1
            elif did is False:
                skipped += 1
            done += 1
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(copied=copied, skipped=skipped, failed=failed)
            elif done % 200 == 0:
                print(f"[PROGRESS] done={done}/{len(files)} copied={copied} skipped={skipped} failed={failed}")

    if pbar is not None:
        pbar.close()

    print(f"[DONE] files={len(files)} copied={copied} skipped={skipped} failed={failed} dst={dst_root}")


if __name__ == "__main__":
    main()
