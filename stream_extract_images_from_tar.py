import argparse
import os
import tarfile
from typing import List, Optional

from megfile.smart import smart_makedirs, smart_open as mopen


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True, help="tar 文件路径或 s3:// URI")
    ap.add_argument("--out-dir", required=True, help="输出目录（本地路径或 s3:// URI）")
    ap.add_argument("--num", type=int, required=True, help="要抽取的图片数量")
    ap.add_argument(
        "--exts",
        default=".jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff,.gif,.avif,.heic,.heif,.jxl",
        help="图片后缀列表（逗号分隔，不区分大小写）",
    )
    ap.add_argument("--skip-existing", action="store_true", help="若目标文件已存在则跳过")
    return ap.parse_args()


def norm_dir(p: str) -> str:
    return p if p.endswith("/") else (p + "/")


def pick_output_name(out_dir: str, name: str, skip_existing: bool) -> Optional[str]:
    base = os.path.basename(name)
    root, ext = os.path.splitext(base)
    if not ext:
        return None
    out_dir = norm_dir(out_dir)
    candidate = out_dir + base
    if skip_existing:
        try:
            from megfile.smart import smart_exists

            if smart_exists(candidate):
                return None
        except Exception:
            pass
    if "://" not in out_dir:
        if not os.path.exists(candidate):
            return candidate
    i = 1
    while True:
        alt = out_dir + f"{root}_{i:03d}{ext}"
        if skip_existing:
            try:
                from megfile.smart import smart_exists

                if smart_exists(alt):
                    i += 1
                    continue
            except Exception:
                pass
        if "://" not in out_dir:
            if os.path.exists(alt):
                i += 1
                continue
        return alt


def main():
    args = parse_args()
    if args.num <= 0:
        raise SystemExit("--num 必须 > 0")
    exts: List[str] = [x.strip().lower() for x in args.exts.split(",") if x.strip()]
    out_dir = args.out_dir
    smart_makedirs(out_dir)

    extracted = 0
    scanned = 0
    with mopen(args.tar, "rb") as f:
        tf = tarfile.open(fileobj=f, mode="r|*")
        for member in tf:
            scanned += 1
            if not member.isreg():
                continue
            name = member.name
            low = name.lower()
            if not any(low.endswith(e) for e in exts):
                continue
            out_path = pick_output_name(out_dir, name, args.skip_existing)
            if out_path is None:
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            with src:
                with mopen(out_path, "wb") as w:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        w.write(chunk)
            extracted += 1
            if extracted % 50 == 0:
                print(f"[progress] extracted={extracted} scanned={scanned}")
            if extracted >= args.num:
                break
    print(f"[done] extracted={extracted} scanned={scanned} out_dir={out_dir}")


if __name__ == "__main__":
    main()

