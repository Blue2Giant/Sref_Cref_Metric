#!/usr/bin/env python3
"""
Collect image paths from local model folders and S3 model folders into JSONL.

Each output line is shaped as:
{"<model_id>": ["/abs/path/a.png", "s3://bucket/.../b.jpg", ...]}
"""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import megfile


IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect image paths from multiple local roots and S3 roots into JSONL"
    )
    parser.add_argument(
        "--local-root",
        action="append",
        default=[],
        help="Local root directory. Its direct subfolders are treated as model_id. Repeatable.",
    )
    parser.add_argument(
        "--s3-root",
        action="append",
        default=[],
        help="S3 root directory. Its direct subfolders are treated as model_id. Repeatable.",
    )
    parser.add_argument(
        "--s3-subdir",
        default="eval_images_with_negative",
        help="Only collect images from this child directory under each S3 model_id.",
    )
    parser.add_argument(
        "--out-jsonl",
        required=True,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--model-id-file",
        help="Optional text file containing allowed model_id values, one per line.",
    )
    parser.add_argument(
        "--local-non-recursive",
        action="store_true",
        help="If set, only collect images directly inside each local model directory.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Write model_id with an empty list when no image is found.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Print progress every N processed model folders. Set 0 to disable.",
    )
    parser.add_argument(
        "--s3-workers",
        type=int,
        default=32,
        help="Concurrent workers for S3 listing.",
    )
    return parser.parse_args()


def is_image_name(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTS


def normalize_s3_dir(path: str) -> str:
    return path.rstrip("/") + "/"


def load_allowed_model_ids(path: str | None) -> set[str] | None:
    if not path:
        return None

    allowed = set()
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            model_id = line.strip()
            if not model_id or model_id.startswith("#"):
                continue
            allowed.add(model_id)
    return allowed


def list_local_images(model_dir: Path, recursive: bool) -> list[str]:
    if recursive:
        iterator = model_dir.rglob("*")
    else:
        iterator = model_dir.iterdir()
    images = [
        str(path.resolve())
        for path in iterator
        if path.is_file() and is_image_name(path.name)
    ]
    images.sort()
    return images


def merge_paths(mapping: dict[str, list[str]], model_id: str, paths: list[str], include_empty: bool):
    if paths or include_empty:
        mapping[model_id].extend(paths)


def collect_local_root(
    root_dir: str,
    mapping: dict[str, list[str]],
    recursive: bool,
    include_empty: bool,
    progress_every: int,
    allowed_model_ids: set[str] | None,
):
    root = Path(root_dir)
    if not root.is_dir():
        raise RuntimeError(f"local root does not exist: {root}")

    if allowed_model_ids is None:
        subdirs = sorted([path for path in root.iterdir() if path.is_dir()], key=lambda path: path.name)
    else:
        subdirs = []
        for model_id in sorted(allowed_model_ids):
            path = root / model_id
            if path.is_dir():
                subdirs.append(path)
    total = len(subdirs)
    written = 0
    empty = 0

    for idx, subdir in enumerate(subdirs, start=1):
        images = list_local_images(subdir, recursive=recursive)
        if not images:
            empty += 1
        else:
            written += 1
        merge_paths(mapping, subdir.name, images, include_empty)
        if progress_every > 0 and idx % progress_every == 0:
            print(
                f"[local] root={root} progress={idx}/{total} non_empty={written} empty={empty}",
                flush=True,
            )

    print(f"[local] root={root} total_subdirs={total} non_empty={written} empty={empty}", flush=True)


def list_s3_images_for_model(s3_root: str, model_id: str, s3_subdir: str) -> tuple[str, list[str], str | None]:
    prefix = f"{normalize_s3_dir(s3_root)}{model_id}/{s3_subdir.strip('/')}/"
    try:
        names = megfile.smart_listdir(prefix)
    except Exception as exc:
        return model_id, [], str(exc)

    images = [prefix + name for name in names if is_image_name(name)]
    images.sort()
    return model_id, images, None


def collect_s3_root(
    s3_root: str,
    mapping: dict[str, list[str]],
    s3_subdir: str,
    include_empty: bool,
    progress_every: int,
    workers: int,
    allowed_model_ids: set[str] | None,
):
    root = normalize_s3_dir(s3_root)
    if allowed_model_ids is None:
        model_ids = sorted(megfile.smart_listdir(root))
    else:
        model_ids = sorted(allowed_model_ids)
    total = len(model_ids)
    written = 0
    empty = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_to_model_id = {
            executor.submit(list_s3_images_for_model, root, model_id, s3_subdir): model_id
            for model_id in model_ids
        }
        for idx, future in enumerate(as_completed(future_to_model_id), start=1):
            model_id, images, error = future.result()
            if error is not None:
                errors += 1
            if not images:
                empty += 1
            else:
                written += 1
            merge_paths(mapping, model_id, images, include_empty)
            if progress_every > 0 and idx % progress_every == 0:
                print(
                    f"[s3] root={root} progress={idx}/{total} non_empty={written} empty={empty} errors={errors}",
                    flush=True,
                )

    print(
        f"[s3] root={root} total_model_ids={total} non_empty={written} empty={empty} errors={errors}",
        flush=True,
    )


def finalize_mapping(mapping: dict[str, list[str]]) -> dict[str, list[str]]:
    finalized = {}
    for model_id, paths in mapping.items():
        finalized[model_id] = sorted(set(paths))
    return finalized


def write_jsonl(out_jsonl: Path, mapping: dict[str, list[str]], include_empty: bool):
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_jsonl.open("w", encoding="utf-8") as fout:
        for model_id in sorted(mapping):
            images = mapping[model_id]
            if not images and not include_empty:
                continue
            fout.write(json.dumps({model_id: images}, ensure_ascii=False) + "\n")
            written += 1
    return written


def main():
    args = parse_args()
    mapping: dict[str, list[str]] = defaultdict(list)
    allowed_model_ids = load_allowed_model_ids(args.model_id_file)

    recursive = not args.local_non_recursive

    if allowed_model_ids is not None:
        print(f"allowed_model_ids={len(allowed_model_ids)}", flush=True)

    for local_root in args.local_root:
        collect_local_root(
            root_dir=local_root,
            mapping=mapping,
            recursive=recursive,
            include_empty=args.include_empty,
            progress_every=args.progress_every,
            allowed_model_ids=allowed_model_ids,
        )

    for s3_root in args.s3_root:
        collect_s3_root(
            s3_root=s3_root,
            mapping=mapping,
            s3_subdir=args.s3_subdir,
            include_empty=args.include_empty,
            progress_every=args.progress_every,
            workers=args.s3_workers,
            allowed_model_ids=allowed_model_ids,
        )

    finalized = finalize_mapping(mapping)
    out_jsonl = Path(args.out_jsonl)
    written = write_jsonl(out_jsonl=out_jsonl, mapping=finalized, include_empty=args.include_empty)

    total_images = sum(len(images) for images in finalized.values())
    print(f"out_jsonl={out_jsonl}")
    print(f"model_ids={len(finalized)}")
    print(f"written={written}")
    print(f"total_images={total_images}")


if __name__ == "__main__":
    main()
