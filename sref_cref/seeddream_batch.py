#!/usr/bin/env python3
import argparse
import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple, Optional
 
import requests
from PIL import Image
 
 
ASPECT_RATIO = [
    "1024x1024",
    "864x1152",
    "1152x864",
    "1280x720",
    "720x1280",
    "832x1248",
    "1248x832",
    "1512x648",
    "2048x2048",
    "2304x1728",
    "1728x2304",
    "2848x1600",
    "1600x2848",
    "2496x1664",
    "1664x2496",
    "3136x1344",
    "4096x4096",
    "3520x4704",
    "4704x3520",
    "5504x3040",
    "3040x5504",
    "3328x4992",
    "4992x3328",
    "6240x2656",
]
 
ASPECT_RATIO_1K = [
    "1024x1024",
    "864x1152",
    "1152x864",
    "1280x720",
    "720x1280",
    "832x1248",
    "1248x832",
    "1512x648",
]
 
 
def _parse_size(s: str) -> Tuple[int, int]:
    w, h = s.lower().split("x", 1)
    return int(w), int(h)
 
 
def _select_size(width: int, height: int, candidates: list[str], min_area: int = 0) -> Optional[str]:
    target = width / float(height)
    best = None
    best_diff = None
    best_area = None
    for s in candidates:
        w, h = _parse_size(s)
        area = w * h
        if min_area and area < min_area:
            continue
        diff = abs((w / float(h)) - target)
        if best is None or diff < best_diff or (diff == best_diff and area < best_area):
            best = s
            best_diff = diff
            best_area = area
    return best
 
 
def _encode_image_data_url(path: Path, fmt: str) -> str:
    img = Image.open(path).convert("RGB")
    buf = BytesIO()
    if fmt == "jpeg":
        img.save(buf, format="JPEG", quality=90)
        mime = "image/jpeg"
    elif fmt == "png":
        img.save(buf, format="PNG")
        mime = "image/png"
    else:
        raise ValueError(f"unsupported fmt: {fmt}")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"
 
 
def _extract_min_area(msg: str) -> Optional[int]:
    needle = "image size must be at least "
    if needle not in msg:
        return None
    digits = ""
    started = False
    for ch in msg[msg.index(needle) + len(needle) :]:
        if ch.isdigit():
            digits += ch
            started = True
        elif started:
            break
    if not digits:
        return None
    try:
        return int(digits)
    except Exception:
        return None
 
 
def _list_images(dir_path: Path) -> Dict[str, Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    out: Dict[str, Path] = {}
    for p in dir_path.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            out[p.stem] = p
    return out
 
 
def _gen_one(
    key: str,
    prompt: str,
    cref_path: Path,
    sref_path: Path,
    out_path: Path,
    api_key: str,
    base_url: str,
    model: str,
    resolution: str,
    image_format: str,
    timeout_s: int,
    download_timeout_s: int,
    overwrite: bool,
) -> Tuple[str, bool, str]:
    if (not overwrite) and out_path.exists():
        return key, True, "skipped"
 
    content_img = Image.open(cref_path).convert("RGB")
    if resolution:
        size = resolution
    else:
        size = _select_size(content_img.width, content_img.height, ASPECT_RATIO_1K, min_area=0)
        if not size:
            size = _select_size(content_img.width, content_img.height, ASPECT_RATIO, min_area=0)
        if not size:
            return key, False, "no_valid_size"
 
    session = requests.Session()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
 
    def do_request(chosen_size: str):
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [
                _encode_image_data_url(cref_path, image_format),
                _encode_image_data_url(sref_path, image_format),
            ],
            "size": chosen_size,
            "response_format": "url",
            "sequential_image_generation": "auto",
            "sequential_image_generation_options": {"max_images": 1},
        }
        return session.post(base_url, headers=headers, json=payload, timeout=timeout_s)
 
    resp = do_request(size)
    if resp.status_code != 200:
        try:
            body = resp.json()
            msg = str(body.get("msg", ""))
        except Exception:
            body = resp.text[:2000]
            msg = str(body)
 
        min_area = _extract_min_area(msg)
        if (not resolution) and resp.status_code == 400 and min_area:
            size2 = _select_size(content_img.width, content_img.height, ASPECT_RATIO, min_area=min_area)
            if size2 and size2 != size:
                resp2 = do_request(size2)
                if resp2.status_code == 200:
                    resp = resp2
                    size = size2
                else:
                    try:
                        body2 = resp2.json()
                        return key, False, f"http_{resp2.status_code}:{body2}"
                    except Exception:
                        return key, False, f"http_{resp2.status_code}:{resp2.text[:2000]}"
        else:
            return key, False, f"http_{resp.status_code}:{body}"
 
    data = resp.json()
    url = data["data"][0]["url"]
    img_resp = session.get(url, timeout=download_timeout_s)
    img_resp.raise_for_status()
 
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.open(BytesIO(img_resp.content)).convert("RGB").save(out_path)
    return key, True, f"ok:{size}"
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cref_dir", required=True)
    ap.add_argument("--sref_dir", required=True)
    ap.add_argument("--prompts_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="doubao-seedream-4.5")
    ap.add_argument("--base_url", default="https://models-proxy.stepfun-inc.com/v1/images/generations")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", type=str, default="")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--resolution",
        type=str,
        default="",
        help='Force size as "WIDTHxHEIGHT". If empty, auto-pick by aspect ratio.',
    )
    ap.add_argument("--image_format", choices=["jpeg", "png"], default="jpeg")
    ap.add_argument("--timeout_s", type=int, default=600)
    ap.add_argument("--download_timeout_s", type=int, default=300)
    args = ap.parse_args()
 
    api_key = os.getenv("SEEDREAM_API_KEY", "")
    if not api_key:
        raise SystemExit("SEEDREAM_API_KEY is required")
 
    cref_dir = Path(args.cref_dir)
    sref_dir = Path(args.sref_dir)
    out_dir = Path(args.out_dir)
    with open(args.prompts_json, "r", encoding="utf-8") as f:
        prompts: Dict[str, str] = {str(k): str(v) for k, v in json.load(f).items()}
 
    cref_map = _list_images(cref_dir)
    sref_map = _list_images(sref_dir)
 
    keys = [k for k in prompts.keys() if k in cref_map and k in sref_map]
    keys.sort()
 
    if args.ids.strip():
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        keys = [k for k in keys if k in wanted]
 
    if args.limit and args.limit > 0:
        keys = keys[: args.limit]
 
    if not keys:
        raise SystemExit("no matched basenames between prompts_json, cref_dir and sref_dir")
 
    try:
        from tqdm import tqdm
 
        pbar = tqdm(total=len(keys), unit="img")
    except Exception:
        tqdm = None
        pbar = None
 
    ok = 0
    failed = 0
    skipped = 0
 
    futures = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        for k in keys:
            out_path = out_dir / f"{k}.png"
            futures.append(
                ex.submit(
                    _gen_one,
                    k,
                    prompts[k],
                    cref_map[k],
                    sref_map[k],
                    out_path,
                    api_key,
                    args.base_url,
                    args.model,
                    args.resolution,
                    args.image_format,
                    int(args.timeout_s),
                    int(args.download_timeout_s),
                    bool(args.overwrite),
                )
            )
 
        for fut in as_completed(futures):
            k, success, msg = fut.result()
            if success:
                if msg == "skipped":
                    skipped += 1
                else:
                    ok += 1
            else:
                failed += 1
                print(f"[FAIL] id={k} {msg}")
            if pbar is not None:
                pbar.update(1)
 
    if pbar is not None:
        pbar.close()
 
    print(f"[DONE] total={len(keys)} ok={ok} skipped={skipped} failed={failed} out_dir={out_dir}")
 
 
if __name__ == "__main__":
    main()
