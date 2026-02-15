import os
import re
import json
import argparse
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from megfile.smart import (
        smart_exists,
        smart_listdir,
        smart_open as mopen,
        smart_makedirs,
    )
except Exception:
    smart_exists = None
    smart_listdir = None
    smart_makedirs = None
    mopen = None


def is_s3_path(path: str) -> bool:
    return path.startswith("s3://") or path.startswith("oss://")


def ensure_parent_dir(path: str) -> None:
    if is_s3_path(path):
        if smart_makedirs is None:
            raise RuntimeError("megfile.smart not available for s3/oss output")
        parent = path.rsplit("/", 1)[0]
        smart_makedirs(parent)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def read_text_any(path: str) -> str:
    if is_s3_path(path):
        if mopen is None:
            raise RuntimeError("megfile.smart not available for s3/oss input")
        with mopen(path, "rb") as f:
            raw = f.read()
    else:
        with open(path, "rb") as f:
            raw = f.read()
    return raw.decode("utf-8-sig")


def read_json_any(path: str) -> Dict:
    if is_s3_path(path):
        if mopen is None:
            raise RuntimeError("megfile.smart not available for s3/oss input")
        with mopen(path, "r", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def exists_any(path: str) -> bool:
    if is_s3_path(path):
        if smart_exists is None:
            raise RuntimeError("megfile.smart not available for s3/oss input")
        return bool(smart_exists(path))
    return os.path.exists(path)


def listdir_any(path: str) -> List[str]:
    if is_s3_path(path):
        if smart_listdir is None:
            raise RuntimeError("megfile.smart not available for s3/oss input")
        return [str(x).rstrip("/") for x in smart_listdir(path)]
    return os.listdir(path)


def load_ids(ids_txt: str) -> List[str]:
    ids: List[str] = []
    for line in read_text_any(ids_txt).splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.lower().endswith(".png"):
            s = s[:-4]
        m = re.search(r"\d+", s)
        ids.append(m.group(0) if m else s)
    return ids


def find_meta_json(root: str, model_id: str) -> Optional[str]:
    root = root.rstrip("/")
    direct = f"{root}/{model_id}/{model_id}.json"
    if exists_any(direct):
        return direct
    subdir = f"{root}/{model_id}"
    if not exists_any(subdir):
        return None
    try:
        names = listdir_any(subdir)
    except Exception:
        return None
    json_names = [n for n in names if str(n).lower().endswith(".json")]
    if not json_names:
        return None
    json_names.sort()
    return f"{subdir}/{json_names[0]}"


def extract_checkpoint_info(meta: Dict) -> Optional[Dict[str, str]]:
    images = meta.get("images")
    if isinstance(images, list) and images:
        img = images[0]
        if isinstance(img, dict):
            ru = img.get("resource_used")
            if isinstance(ru, list):
                for item in ru:
                    if not isinstance(item, dict):
                        continue
                    mt = item.get("modelType") or item.get("model_type")
                    if isinstance(mt, str) and mt.lower() == "checkpoint":
                        vn = item.get("versionName") or item.get("version_name") or item.get("modelName")
                        url = item.get("url")
                        if isinstance(vn, str) and vn.strip():
                            info = {"versionName": vn.strip()}
                            if isinstance(url, str) and url.strip():
                                info["url"] = url.strip()
                            return info
    vn = meta.get("versionName")
    mt = meta.get("modelType")
    if isinstance(vn, str) and vn.strip() and (not mt or (isinstance(mt, str) and mt.lower() == "checkpoint")):
        return {"versionName": vn.strip()}
    return None


def collect_one(root: str, model_id: str) -> Tuple[str, Optional[Dict[str, str]], Optional[str]]:
    path = find_meta_json(root, model_id)
    if not path:
        return model_id, None, "missing_json"
    try:
        meta = read_json_any(path)
    except Exception:
        return model_id, None, "read_failed"
    info = extract_checkpoint_info(meta)
    if not info:
        return model_id, None, "no_checkpoint"
    return model_id, info, None


def write_json_any(path: str, data: Dict) -> None:
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    if is_s3_path(path):
        if mopen is None:
            raise RuntimeError("megfile.smart not available for s3/oss output")
        ensure_parent_dir(path)
        with mopen(path, "w", encoding="utf-8") as f:
            f.write(payload)
    else:
        ensure_parent_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="远程桶根目录，例如 s3://bucket/prefix")
    ap.add_argument("--ids-txt", required=True, help="每行一个 model_id 的 txt（本地或 s3://）")
    ap.add_argument("--output-json", required=True, help="输出 json 路径（本地或 s3://）")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    ids = load_ids(args.ids_txt)
    if not ids:
        raise SystemExit("ids-txt 为空")

    result: Dict[str, Optional[Dict[str, str]]] = {}
    errors: Dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = [ex.submit(collect_one, args.root, mid) for mid in ids]
        for fut in as_completed(futs):
            mid, info, err = fut.result()
            result[mid] = info
            if err:
                errors[err] = errors.get(err, 0) + 1

    out = {
        "root": args.root,
        "ids_count": len(ids),
        "errors": errors,
        "checkpoints": result,
    }
    write_json_any(args.output_json, out)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
