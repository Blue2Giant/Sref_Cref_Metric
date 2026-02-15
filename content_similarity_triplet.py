#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import json
import time
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel

from megfile.smart import (
    smart_open as mopen,
    smart_scandir,
    smart_exists,
    smart_isdir,
    smart_makedirs,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# 本地模型路径
DINO_MODEL_NAME = "/mnt/jfs/model_zoo/dinov2-base"
CLIP_MODEL_NAME = "/mnt/jfs/model_zoo/clip-vit-large-patch14"


# --------------------- path utils ---------------------
def is_remote(path: str) -> bool:
    return path.startswith("s3://") or path.startswith("oss://")


def join_path(a: str, b: str) -> str:
    return a.rstrip("/") + "/" + b.lstrip("/")


def basename_noext(p: str) -> str:
    base = os.path.basename(p.rstrip("/"))
    stem, _ = os.path.splitext(base)
    return stem


def list_images_recursive_megfile(root: str) -> List[str]:
    paths: List[str] = []
    root = root.rstrip("/")

    if is_remote(root):
        stack = [root]
        while stack:
            cur = stack.pop()
            try:
                for e in smart_scandir(cur):
                    try:
                        if e.is_dir():
                            stack.append(e.path)
                        else:
                            ext = os.path.splitext(e.name)[1].lower()
                            if ext in IMG_EXTS:
                                paths.append(e.path)
                    except Exception:
                        continue
            except FileNotFoundError:
                continue
    else:
        rp = Path(root)
        if not rp.is_dir():
            return []
        for p in rp.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                paths.append(str(p))

    paths.sort()
    return paths


def dir_exists(path: str) -> bool:
    if is_remote(path):
        return smart_exists(path) and smart_isdir(path)
    return os.path.isdir(path)


def load_image_any(path: str) -> Image.Image:
    if is_remote(path):
        with mopen(path, "rb") as f:
            data = f.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    else:
        return Image.open(path).convert("RGB")


def save_torch_pt_any(path: str, obj: Dict) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        smart_makedirs(out_dir, exist_ok=True)

    buf = io.BytesIO()
    torch.save(obj, buf)
    buf.seek(0)
    with mopen(path, "wb") as f:
        f.write(buf.read())


def load_torch_pt_any(path: str) -> Dict:
    with mopen(path, "rb") as f:
        data = f.read()
    return torch.load(io.BytesIO(data), map_location="cpu")


# --------------------- id list ---------------------
def parse_id_from_line(line: str) -> Optional[str]:
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    base = os.path.basename(s)
    stem, _ = os.path.splitext(base)
    stem = stem.strip()
    return stem or None


def load_id_list(txt_path: str) -> Set[str]:
    ids: Set[str] = set()
    with mopen(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            x = parse_id_from_line(line)
            if x:
                ids.add(x)
    return ids


# --------------------- backend wrapper ---------------------
class ImgBackend:
    def __init__(self, backend: str, device: torch.device):
        self.backend = backend
        self.device = device

        if backend == "clip":
            self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
            self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
        elif backend == "dino":
            self.processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
            self.model = AutoModel.from_pretrained(DINO_MODEL_NAME).to(device).eval()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    @torch.inference_mode()
    def extract_batch(self, pil_images: List[Image.Image]) -> torch.Tensor:
        """
        return: [B, D] unit vectors (L2-normalized), float32 on self.device
        """
        if not pil_images:
            return torch.empty((0, 1), device=self.device, dtype=torch.float32)

        inputs = self.processor(images=pil_images, return_tensors="pt")
        # 只取 pixel_values
        pixel_values = inputs["pixel_values"].to(self.device, non_blocking=True)

        # H100: bf16 + TF32 更快
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            autocast = torch.autocast(device_type="cpu", dtype=torch.float32)

        with autocast:
            if self.backend == "clip":
                feats = self.model.get_image_features(pixel_values=pixel_values)  # [B, D]
            else:
                out = self.model(pixel_values=pixel_values)
                # out[0] == last_hidden_state: [B, T, D]
                last = out[0]
                feats = last[:, 0, :]  # CLS: [B, D]

        feats = F.normalize(feats, p=2, dim=-1).float()
        return feats


def load_images_safe(paths: List[str]) -> Tuple[List[str], List[Image.Image]]:
    ok_paths: List[str] = []
    ok_imgs: List[Image.Image] = []
    for p in paths:
        try:
            ok_imgs.append(load_image_any(p))
            ok_paths.append(p)
        except (UnidentifiedImageError, OSError, ValueError):
            continue
    return ok_paths, ok_imgs


# --------------------- mean feature cache ---------------------
def mean_feat_cache_path(cache_root: str, content_id: str, backend: str) -> str:
    # 存到：<cache_root>/<content_id>/mean_feat_<backend>.pt
    return join_path(join_path(cache_root, content_id), f"mean_feat_{backend}.pt")


def build_or_load_gallery_mean_feat(
    backend: ImgBackend,
    gallery_dir: str,
    cache_path: str,
    batch_size: int,
    overwrite_cache: bool,
    gallery_max_images: int = 0,
    verbose: bool = True,
) -> Dict:
    """
    计算并缓存 gallery 的 “单位向量均值”：
      mean_vec = (1/N) * sum_i unit(g_i)
    对单位 probe 向量：dot(probe, mean_vec) == mean_i dot(probe, unit(g_i))（精确等价原 mean_similarity）
    """
    if (not overwrite_cache) and smart_exists(cache_path):
        try:
            obj = load_torch_pt_any(cache_path)
            if "mean_vec" in obj and isinstance(obj["mean_vec"], torch.Tensor):
                return obj
        except Exception:
            pass

    if not dir_exists(gallery_dir):
        raise RuntimeError(f"gallery_dir not exists: {gallery_dir}")

    gallery_imgs = list_images_recursive_megfile(gallery_dir)
    if gallery_max_images and gallery_max_images > 0:
        gallery_imgs = gallery_imgs[:gallery_max_images]

    if not gallery_imgs:
        raise RuntimeError(f"gallery empty: {gallery_dir}")

    t0 = time.time()
    sum_vec: Optional[torch.Tensor] = None
    count = 0

    # 分 batch：先读图，再一次性 forward
    for i in range(0, len(gallery_imgs), batch_size):
        chunk = gallery_imgs[i : i + batch_size]
        ok_paths, ok_imgs = load_images_safe(chunk)
        if not ok_imgs:
            continue
        feats = backend.extract_batch(ok_imgs)  # [B, D] unit
        if sum_vec is None:
            sum_vec = feats.sum(dim=0).detach().to("cpu")
        else:
            sum_vec += feats.sum(dim=0).detach().to("cpu")
        count += feats.shape[0]

    if sum_vec is None or count == 0:
        raise RuntimeError(f"gallery all bad images: {gallery_dir}")

    mean_vec = (sum_vec / float(count)).contiguous()  # CPU float32, NOT re-normalized

    obj = {
        "backend": backend.backend,
        "gallery_dir": gallery_dir,
        "num_gallery_images": int(count),
        "mean_vec": mean_vec,  # CPU tensor [D], float32
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_torch_pt_any(cache_path, obj)

    if verbose:
        dt = time.time() - t0
        print(f"[CACHE] build mean feat: {cache_path} (N={count}, {dt:.1f}s)")

    return obj


# --------------------- pair_dir processing ---------------------
def parse_pair_dir(pair_dir: str) -> Optional[Tuple[str, str]]:
    base = os.path.basename(pair_dir.rstrip("/"))
    if "__" not in base:
        return None
    a, b = base.split("__", 1)
    return a.strip(), b.strip()  # style_id, content_id


def process_pair_dir_fast(
    pair_dir: str,
    style_id: str,
    content_id: str,
    backend: ImgBackend,
    mean_vec_cpu: torch.Tensor,
    output_json: str,
    overwrite_out: bool,
    batch_size: int,
    verbose: bool = True,
) -> None:
    if (not overwrite_out) and smart_exists(output_json):
        if verbose:
            print(f"[SKIP] {pair_dir}: out exists {output_json}")
        return

    probe_imgs = list_images_recursive_megfile(pair_dir)
    if not probe_imgs:
        if verbose:
            print(f"[WARN] {pair_dir}: no probe images")
        return

    mean_vec = mean_vec_cpu.to(backend.device, non_blocking=True)  # [D]
    per_img: Dict[str, float] = {}
    total = 0.0
    valid = 0

    for i in range(0, len(probe_imgs), batch_size):
        chunk = probe_imgs[i : i + batch_size]
        ok_paths, ok_imgs = load_images_safe(chunk)
        if not ok_imgs:
            continue
        feats = backend.extract_batch(ok_imgs)  # [B, D] unit
        sims = (feats * mean_vec.unsqueeze(0)).sum(dim=-1)  # [B]
        sims = sims.detach().float().cpu().tolist()

        for p, s in zip(ok_paths, sims):
            per_img[p] = float(s)
            total += float(s)
            valid += 1

    if valid == 0:
        if verbose:
            print(f"[WARN] {pair_dir}: all probes failed")
        return

    overall_mean = total / float(valid)

    out = {
        "backend": backend.backend,
        "pair_dir": pair_dir,
        "style_id": style_id,
        "content_id": content_id,
        "num_probe_images": int(valid),
        "per_image_mean_similarity": per_img,
        "overall_mean_similarity": float(overall_mean),
    }

    out_dir = os.path.dirname(output_json)
    if out_dir:
        smart_makedirs(out_dir, exist_ok=True)
    with mopen(output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"[OK] {pair_dir}: probes={valid}, overall={overall_mean:.4f} -> {output_json}")


# --------------------- grouping + worker ---------------------
def assign_groups_to_workers(groups: Dict[str, List[str]], num_workers: int) -> List[List[Tuple[str, List[str]]]]:
    """
    groups: content_id -> [pair_dir...]
    return: workers -> list of (content_id, pair_dirs)
    """
    items = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    buckets: List[List[Tuple[str, List[str]]]] = [[] for _ in range(num_workers)]
    loads = [0] * num_workers

    for cid, dirs in items:
        j = int(np.argmin(loads))
        buckets[j].append((cid, dirs))
        loads[j] += len(dirs)
    return buckets


def worker_main(
    worker_id: int,
    work_items: List[Tuple[str, List[str]]],  # [(content_id, [pair_dirs...]), ...]
    root: str,
    content_root: str,
    cache_root: str,
    gallery_subdir: str,
    backend_name: str,
    output_name: str,
    overwrite_out: bool,
    overwrite_cache: bool,
    batch_size: int,
    gallery_max_images: int,
    gpu_id: Optional[int],
) -> None:
    # 控制每进程 CPU 线程数，避免多进程把 CPU 线程打爆
    torch.set_num_threads(1)

    if torch.cuda.is_available() and gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        print(f"[W{worker_id}] GPU {gpu_id} -> {device}")
    else:
        device = torch.device("cpu")
        print(f"[W{worker_id}] CPU mode")

    backend = ImgBackend(backend_name, device)

    # 进程内缓存：content_id -> mean_vec_cpu
    mean_cache: Dict[str, torch.Tensor] = {}

    for content_id, pair_dirs in work_items:
        # gallery: <content_root>/<content_id>/<gallery_subdir>
        gallery_dir = join_path(join_path(content_root, content_id), gallery_subdir)
        cache_path = mean_feat_cache_path(cache_root, content_id, backend_name)

        try:
            obj = build_or_load_gallery_mean_feat(
                backend=backend,
                gallery_dir=gallery_dir,
                cache_path=cache_path,
                batch_size=batch_size,
                overwrite_cache=overwrite_cache,
                gallery_max_images=gallery_max_images,
                verbose=True,
            )
            mean_vec_cpu = obj["mean_vec"]
            mean_cache[content_id] = mean_vec_cpu
        except Exception as e:
            print(f"[W{worker_id}] [ERROR] content_id={content_id} build/load mean feat failed: {e}")
            continue

        for pair_dir in pair_dirs:
            parsed = parse_pair_dir(pair_dir)
            if not parsed:
                continue
            style_id, cid2 = parsed
            if cid2 != content_id:
                # 理论上不会发生（分组就是按 content_id）
                continue
            out_json = join_path(pair_dir, output_name)
            try:
                process_pair_dir_fast(
                    pair_dir=pair_dir,
                    style_id=style_id,
                    content_id=content_id,
                    backend=backend,
                    mean_vec_cpu=mean_vec_cpu,
                    output_json=out_json,
                    overwrite_out=overwrite_out,
                    batch_size=batch_size,
                    verbose=True,
                )
            except Exception as e:
                print(f"[W{worker_id}] [ERROR] pair_dir={pair_dir} failed: {e}")


# --------------------- main ---------------------
def main():
    parser = argparse.ArgumentParser(
        description="多进程(每GPU一进程) + content_id 均值特征缓存：快速计算 triplet_content_similarity"
    )
    parser.add_argument("--root", required=True, help="包含 <style_id>__<content_id> 子目录的根目录（本地或 s3://）")
    parser.add_argument("--content-root", required=True, help="内容图库根目录：<content_id>/<gallery_subdir>/ 存图")
    parser.add_argument(
        "--cache-root",
        default=None,
        help="均值特征缓存根目录（默认=content-root）。会写 <cache-root>/<content_id>/mean_feat_<backend>.pt",
    )
    parser.add_argument("--gallery-subdir", default="content_100", help="每个 content_id 下用作 gallery 的子目录名")
    parser.add_argument("--backend", choices=["clip", "dino"], default="clip", help="默认 clip（建议）")
    parser.add_argument("--output-name", default="triplet_content_similarity.json", help="每个 pair_dir 输出 JSON 文件名")
    parser.add_argument("--overwrite", action="store_true", help="覆盖 pair_dir 下已有输出 JSON")
    parser.add_argument("--overwrite-cache", action="store_true", help="覆盖已有 mean_feat_<backend>.pt，强制重算")
    parser.add_argument("--batch-size", type=int, default=64, help="提特征 batch size（CLIP/H100 可适当加大）")
    parser.add_argument("--gallery-max-images", type=int, default=0, help="限制 gallery 最大用多少张图(0=不限制)")
    parser.add_argument(
        "--id-list",
        default=None,
        help="可选：txt 文件，每行一个 id；这里按 content_id 过滤（与你当前主脚本一致）",
    )
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="逗号分隔 GPU 列表，如 '0,1,2'。不传则使用所有可见 GPU；无 GPU 则单进程 CPU",
    )

    args = parser.parse_args()

    root = args.root.rstrip("/")
    content_root = args.content_root.rstrip("/")
    cache_root = (args.cache_root.rstrip("/") if args.cache_root else content_root)

    # 1) 扫 pair_dirs
    all_dirs = []
    if is_remote(root):
        try:
            for e in smart_scandir(root):
                if e.is_dir():
                    all_dirs.append(e.path)
        except FileNotFoundError:
            all_dirs = []
    else:
        rp = Path(root)
        if rp.is_dir():
            all_dirs = [str(p) for p in rp.iterdir() if p.is_dir()]
    all_dirs.sort()
    if not all_dirs:
        raise SystemExit(f"root 下没有子目录: {root}")

    allowed: Optional[Set[str]] = None
    if args.id_list:
        allowed = load_id_list(args.id_list)
        print(f"[INFO] id-list loaded: {len(allowed)} ids")

    pair_dirs: List[str] = []
    groups: Dict[str, List[str]] = {}  # content_id -> pair_dirs
    for d in all_dirs:
        parsed = parse_pair_dir(d)
        if not parsed:
            continue
        style_id, content_id = parsed
        if allowed is not None and content_id not in allowed:
            continue
        pair_dirs.append(d)
        groups.setdefault(content_id, []).append(d)

    if not pair_dirs:
        raise SystemExit("没有任何可处理的 <style_id>__<content_id> 目录（检查 root / id-list）")

    print(f"[INFO] pair_dirs = {len(pair_dirs)}, unique content_id = {len(groups)}")

    # 2) GPU / worker 规划：默认每 GPU 一个进程
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
    else:
        n = 0

    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip() != ""]
    else:
        gpu_ids = list(range(n))

    if n == 0 or not gpu_ids:
        num_workers = 1
        gpu_ids = [None]
        print("[INFO] no GPU -> CPU single process")
    else:
        num_workers = len(gpu_ids)
        print(f"[INFO] GPUs={gpu_ids}, workers={num_workers} (1 GPU 1 worker)")

    # 3) 把同 content_id 的 pair_dir 尽量放到同一个 worker（避免并发重算同一份 mean_feat）
    work_buckets = assign_groups_to_workers(groups, num_workers)

    procs: List[mp.Process] = []
    for wid in range(num_workers):
        gpu_id = gpu_ids[wid] if gpu_ids[0] is not None else None
        p = mp.Process(
            target=worker_main,
            args=(
                wid,
                work_buckets[wid],
                root,
                content_root,
                cache_root,
                args.gallery_subdir,
                args.backend,
                args.output_name,
                args.overwrite,
                args.overwrite_cache,
                args.batch_size,
                args.gallery_max_images,
                gpu_id,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("[DONE] all finished")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()