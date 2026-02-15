import os
import re
import shutil
import argparse
from typing import List, Tuple

"""
python3 /data/LoraPipeline/utils/clean_pair_subdirs.py --root /mnt/jfs/loras_combine/flux_0111
"""
RE_PAIR_DIR = re.compile(r"^\d+__\d+$")


def list_subdirs(root: str) -> List[str]:
    out: List[str] = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p):
            out.append(name)
    out.sort()
    return out


def classify(root: str) -> Tuple[List[str], List[str]]:
    keep: List[str] = []
    remove: List[str] = []
    for name in list_subdirs(root):
        if RE_PAIR_DIR.fullmatch(name):
            keep.append(name)
        else:
            remove.append(name)
    return keep, remove


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="要清理的本地目录")
    ap.add_argument("--apply", action="store_true", help="实际删除；不传则 dry-run")
    ap.add_argument("--limit", type=int, default=50, help="最多打印多少条将删除的目录名")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise SystemExit(f"not a directory: {root}")

    keep, remove = classify(root)
    print(f"[INFO] root={root}")
    print(f"[INFO] keep={len(keep)} remove={len(remove)} dry_run={not args.apply}")

    if remove:
        for name in remove[: max(0, int(args.limit))]:
            print(f"[DEL] {name}")
        if len(remove) > int(args.limit):
            print(f"[INFO] ... {len(remove) - int(args.limit)} more")

    if not args.apply:
        return

    deleted = 0
    failed = 0
    for name in remove:
        p = os.path.join(root, name)
        try:
            shutil.rmtree(p)
            deleted += 1
        except Exception:
            failed += 1
    print(f"[DONE] deleted={deleted} failed={failed}")


if __name__ == "__main__":
    main()
