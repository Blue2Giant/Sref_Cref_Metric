#!/usr/bin/env python3
"""Sample style/content/target images and render them as a matrix.

Layout:
  - columns are style model ids
  - rows are content model ids
  - top row shows one sampled style-only image per style id
  - left column shows one sampled content-only image per content id
  - inner cells show one sampled dual/target image for content__style
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    from megfile import smart_open
except Exception:
    smart_open = None


TRIPLET_ROOT = Path("/data/benchmark_metrics/lora_pipeline/meta/triplet_jsonls")
DEFAULT_OUTPUT_ROOT = Path("/mnt/jfs/loras_combine/style_content_matrix_samples")
BASE_MODELS = ("flux", "qwen", "illustrious")


@dataclass
class Selection:
    kind: str
    key: str
    image: Image.Image
    source: str = ""
    status: str = "ok"
    error: str = ""
    output_path: str = ""
    content_model_id: str = ""
    style_model_id: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a content(row) x style(col) LoRA sample matrix from triplet JSONLs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-model", choices=BASE_MODELS, default="flux")
    parser.add_argument("--triplet-dir", default=str(TRIPLET_ROOT))
    parser.add_argument("--content-jsonl", default="", help="Override content one-lora JSONL.")
    parser.add_argument("--style-jsonl", default="", help="Override style one-lora JSONL.")
    parser.add_argument(
        "--target-jsonl",
        "--dual-jsonl",
        dest="target_jsonl",
        default="",
        help="Override dual/target JSONL. Defaults to *_dual_lora_style_content_filtered.jsonl.",
    )
    parser.add_argument(
        "--content-ids",
        nargs="*",
        default=[],
        help="Content ids. Accepts space-separated or comma-separated values.",
    )
    parser.add_argument(
        "--style-ids",
        nargs="*",
        default=[],
        help="Style ids. Accepts space-separated or comma-separated values.",
    )
    parser.add_argument("--content-id-file", default="", help="Optional file containing content ids.")
    parser.add_argument("--style-id-file", default="", help="Optional file containing style ids.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", default="", help="Exact output directory. Overrides output-root.")
    parser.add_argument("--matrix-filename", default="matrix.jpg")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--label-height", type=int, default=46)
    parser.add_argument("--gap", type=int, default=8)
    parser.add_argument("--margin", type=int, default=24)
    parser.add_argument("--font-size", type=int, default=18)
    parser.add_argument("--title-font-size", type=int, default=24)
    parser.add_argument("--jpg-quality", type=int, default=92)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-pair-label", action="store_true", help="Print content__style id below target cells.")
    parser.add_argument("--no-save-cells", action="store_true", help="Only save the matrix and manifests.")
    parser.add_argument("--allow-reversed-pair-key", action="store_true", help="Try style__content if content__style is absent.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any id/pair/image is missing.")
    return parser.parse_args()


def read_ids(values: Sequence[str], id_file: str) -> list[str]:
    chunks: list[str] = []
    chunks.extend(str(value) for value in values)
    if id_file:
        chunks.append(Path(id_file).read_text(encoding="utf-8"))

    ids: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        for raw_line in str(chunk).splitlines():
            line = raw_line.split("#", 1)[0]
            for value in re.split(r"[\s,]+", line.strip()):
                value = value.strip()
                if value and value not in seen:
                    ids.append(value)
                    seen.add(value)
    return ids


def default_jsonl_paths(base_model: str, triplet_dir: Path) -> tuple[Path, Path, Path]:
    content_jsonl = triplet_dir / f"{base_model}_content_one_lora.jsonl"
    style_jsonl = triplet_dir / f"{base_model}_style_one_lora.jsonl"
    filtered_target_jsonl = triplet_dir / f"{base_model}_dual_lora_style_content_filtered.jsonl"
    target_jsonl = filtered_target_jsonl
    if not target_jsonl.exists():
        target_jsonl = triplet_dir / f"{base_model}_dual_lora.jsonl"
    if base_model == "flux" and not target_jsonl.exists():
        target_jsonl = triplet_dir / "flux__dual_lora.jsonl"
    return content_jsonl, style_jsonl, target_jsonl


def read_jsonl_map(path: Path) -> dict[str, list[str]]:
    data: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception as exc:
                raise ValueError(f"bad json at {path}:{line_number}: {exc}") from exc
            if not isinstance(item, dict):
                continue
            for key, value in item.items():
                paths = normalize_paths(value)
                if paths:
                    data[str(key)] = paths
    return data


def normalize_paths(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        path = item.strip()
        if path and path not in seen:
            out.append(path)
            seen.add(path)
    return out


def stable_output_dir(args: argparse.Namespace, content_ids: Sequence[str], style_ids: Sequence[str]) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    digest = hashlib.sha1(
        json.dumps(
            {
                "base_model": args.base_model,
                "content_ids": list(content_ids),
                "style_ids": list(style_ids),
                "seed": args.seed,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:10]
    name = f"{args.base_model}_{len(content_ids)}x{len(style_ids)}_{digest}"
    return Path(args.output_root) / name


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output dir already exists: {path} (use --overwrite)")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def read_image(path: str) -> Image.Image:
    if path.startswith("s3://"):
        if smart_open is None:
            raise RuntimeError("megfile is required for s3:// paths")
        with smart_open(path, "rb") as handle:
            data = handle.read()
        with Image.open(io.BytesIO(data)) as image:
            image = ImageOps.exif_transpose(image)
            image.load()
            return image.convert("RGB")

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        image.load()
        return image.convert("RGB")


def local_exists(path: str) -> bool:
    return not path.startswith("s3://") and os.path.isfile(path)


def select_image(
    kind: str,
    key: str,
    paths: Sequence[str],
    seed: int,
    placeholder_size: int,
    content_model_id: str = "",
    style_model_id: str = "",
) -> Selection:
    if not paths:
        return missing_selection(kind, key, placeholder_size, "no paths", content_model_id, style_model_id)

    local_paths = [path for path in paths if local_exists(path)]
    remote_paths = [path for path in paths if path.startswith("s3://")]
    other_paths = [path for path in paths if not path.startswith("s3://") and not local_exists(path)]
    rng = random.Random(f"{seed}:{kind}:{key}")
    for group in (local_paths, remote_paths, other_paths):
        group = list(group)
        rng.shuffle(group)
        for path in group:
            try:
                image = fit_to_square(read_image(path), placeholder_size)
            except Exception as exc:
                last_error = str(exc)
                continue
            return Selection(
                kind=kind,
                key=key,
                image=image,
                source=path,
                content_model_id=content_model_id,
                style_model_id=style_model_id,
            )

    return missing_selection(kind, key, placeholder_size, last_error if "last_error" in locals() else "not found", content_model_id, style_model_id)


def missing_selection(
    kind: str,
    key: str,
    size: int,
    error: str,
    content_model_id: str = "",
    style_model_id: str = "",
) -> Selection:
    image = Image.new("RGB", (size, size), (238, 238, 238))
    draw = ImageDraw.Draw(image)
    font = load_font(max(12, size // 13))
    text = f"MISSING\n{kind}\n{key}"
    draw_multiline_centered(draw, text, (0, 0, size, size), font, fill=(120, 35, 35))
    return Selection(
        kind=kind,
        key=key,
        image=image,
        status="missing",
        error=error,
        content_model_id=content_model_id,
        style_model_id=style_model_id,
    )


def fit_to_square(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def wrap_label(label: str, max_width: int, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> list[str]:
    parts = label.splitlines()
    out: list[str] = []
    for part in parts:
        if text_size(draw, part, font)[0] <= max_width:
            out.append(part)
            continue
        current = ""
        for char in part:
            candidate = current + char
            if current and text_size(draw, candidate, font)[0] > max_width:
                out.append(current)
                current = char
            else:
                current = candidate
        if current:
            out.append(current)
    return out[:3]


def draw_multiline_centered(
    draw: ImageDraw.ImageDraw,
    label: str,
    box: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (30, 30, 30),
) -> None:
    x0, y0, x1, y1 = box
    max_width = x1 - x0 - 8
    lines = wrap_label(label, max_width, draw, font)
    line_heights = [text_size(draw, line, font)[1] for line in lines]
    total_height = sum(line_heights) + max(0, len(lines) - 1) * 4
    y = y0 + max(0, (y1 - y0 - total_height) // 2)
    for line, height in zip(lines, line_heights):
        width, _ = text_size(draw, line, font)
        draw.text((x0 + (x1 - x0 - width) // 2, y), line, font=font, fill=fill)
        y += height + 4


def render_cell(
    canvas: Image.Image,
    selection: Selection | None,
    label: str,
    x: int,
    y: int,
    image_size: int,
    label_height: int,
    font: ImageFont.ImageFont,
    border: tuple[int, int, int],
    fill: tuple[int, int, int] = (255, 255, 255),
) -> None:
    draw = ImageDraw.Draw(canvas)
    cell_height = image_size + label_height
    draw.rounded_rectangle((x, y, x + image_size, y + cell_height), radius=8, fill=fill, outline=border, width=2)
    if selection is not None:
        fitted = fit_to_square(selection.image, image_size)
        canvas.paste(fitted, (x, y))
        draw.rectangle((x, y, x + image_size, y + image_size), outline=border, width=2)
    draw_multiline_centered(
        draw,
        label,
        (x + 4, y + image_size, x + image_size - 4, y + image_size + label_height),
        font,
        fill=(25, 25, 25),
    )


def save_selected_image(selection: Selection, output_path: Path, jpg_quality: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selection.image.convert("RGB").save(output_path, format="JPEG", quality=jpg_quality, optimize=True)
    selection.output_path = str(output_path)


def sanitize(value: str, max_len: int = 160) -> str:
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip()).strip("._-")
    return (safe or "item")[:max_len]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def iter_manifest_records(selections: Iterable[Selection]) -> Iterable[dict[str, Any]]:
    for selection in selections:
        yield {
            "kind": selection.kind,
            "key": selection.key,
            "content_model_id": selection.content_model_id,
            "style_model_id": selection.style_model_id,
            "source": selection.source,
            "status": selection.status,
            "error": selection.error,
            "output_path": selection.output_path,
        }


def main() -> int:
    args = parse_args()
    content_ids = read_ids(args.content_ids, args.content_id_file)
    style_ids = read_ids(args.style_ids, args.style_id_file)
    if not content_ids or not style_ids:
        raise SystemExit("please provide both --content-ids and --style-ids, or their id files")

    triplet_dir = Path(args.triplet_dir)
    default_content_jsonl, default_style_jsonl, default_target_jsonl = default_jsonl_paths(args.base_model, triplet_dir)
    content_jsonl = Path(args.content_jsonl) if args.content_jsonl else default_content_jsonl
    style_jsonl = Path(args.style_jsonl) if args.style_jsonl else default_style_jsonl
    target_jsonl = Path(args.target_jsonl) if args.target_jsonl else default_target_jsonl
    for path in (content_jsonl, style_jsonl, target_jsonl):
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir = stable_output_dir(args, content_ids, style_ids)
    prepare_output_dir(output_dir, args.overwrite)

    content_map = read_jsonl_map(content_jsonl)
    style_map = read_jsonl_map(style_jsonl)
    target_map = read_jsonl_map(target_jsonl)

    content_selections: dict[str, Selection] = {}
    style_selections: dict[str, Selection] = {}
    target_selections: dict[tuple[str, str], Selection] = {}

    for content_id in content_ids:
        content_selections[content_id] = select_image(
            "content",
            content_id,
            content_map.get(content_id, []),
            args.seed,
            args.image_size,
            content_model_id=content_id,
        )

    for style_id in style_ids:
        style_selections[style_id] = select_image(
            "style",
            style_id,
            style_map.get(style_id, []),
            args.seed,
            args.image_size,
            style_model_id=style_id,
        )

    for content_id in content_ids:
        for style_id in style_ids:
            pair_key = f"{content_id}__{style_id}"
            paths = target_map.get(pair_key, [])
            selected_key = pair_key
            if not paths and args.allow_reversed_pair_key:
                reversed_key = f"{style_id}__{content_id}"
                paths = target_map.get(reversed_key, [])
                selected_key = reversed_key if paths else pair_key
            target_selections[(content_id, style_id)] = select_image(
                "target",
                selected_key,
                paths,
                args.seed,
                args.image_size,
                content_model_id=content_id,
                style_model_id=style_id,
            )

    if not args.no_save_cells:
        for content_id, selection in content_selections.items():
            save_selected_image(selection, output_dir / "cells" / "content" / f"{sanitize(content_id)}.jpg", args.jpg_quality)
        for style_id, selection in style_selections.items():
            save_selected_image(selection, output_dir / "cells" / "style" / f"{sanitize(style_id)}.jpg", args.jpg_quality)
        for (content_id, style_id), selection in target_selections.items():
            pair_name = sanitize(f"{content_id}__{style_id}")
            save_selected_image(selection, output_dir / "cells" / "target" / f"{pair_name}.jpg", args.jpg_quality)

    font = load_font(args.font_size)
    title_font = load_font(args.title_font_size)
    image_size = args.image_size
    label_height = args.label_height
    cell_height = image_size + label_height
    cols = len(style_ids) + 1
    rows = len(content_ids) + 1
    title_height = 48
    width = args.margin * 2 + cols * image_size + (cols - 1) * args.gap
    height = args.margin * 2 + title_height + rows * cell_height + (rows - 1) * args.gap
    canvas = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)

    title = f"{args.base_model} style/content matrix  |  rows=content ({len(content_ids)})  cols=style ({len(style_ids)})"
    draw_multiline_centered(draw, title, (args.margin, args.margin, width - args.margin, args.margin + title_height - 8), title_font)

    start_y = args.margin + title_height
    top_left_x = args.margin
    top_left_y = start_y
    render_cell(
        canvas,
        None,
        "content ↓\nstyle →",
        top_left_x,
        top_left_y,
        image_size,
        label_height,
        font,
        border=(90, 90, 90),
        fill=(235, 235, 235),
    )

    for col_index, style_id in enumerate(style_ids, start=1):
        x = args.margin + col_index * (image_size + args.gap)
        y = start_y
        render_cell(
            canvas,
            style_selections[style_id],
            f"style\n{style_id}",
            x,
            y,
            image_size,
            label_height,
            font,
            border=(70, 105, 180),
            fill=(248, 251, 255),
        )

    for row_index, content_id in enumerate(content_ids, start=1):
        x = args.margin
        y = start_y + row_index * (cell_height + args.gap)
        render_cell(
            canvas,
            content_selections[content_id],
            f"content\n{content_id}",
            x,
            y,
            image_size,
            label_height,
            font,
            border=(190, 125, 55),
            fill=(255, 250, 245),
        )
        for col_index, style_id in enumerate(style_ids, start=1):
            cell_x = args.margin + col_index * (image_size + args.gap)
            selection = target_selections[(content_id, style_id)]
            label = selection.key if args.show_pair_label else ""
            render_cell(
                canvas,
                selection,
                label,
                cell_x,
                y,
                image_size,
                label_height,
                font,
                border=(120, 120, 120) if selection.status == "ok" else (180, 70, 70),
                fill=(255, 255, 255),
            )

    matrix_path = output_dir / args.matrix_filename
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    if matrix_path.suffix.lower() in {".jpg", ".jpeg"}:
        canvas.save(matrix_path, format="JPEG", quality=args.jpg_quality, optimize=True)
    else:
        canvas.save(matrix_path)

    selections = list(content_selections.values()) + list(style_selections.values()) + list(target_selections.values())
    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in iter_manifest_records(selections):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    missing_records = [record for record in iter_manifest_records(selections) if record["status"] != "ok"]
    summary = {
        "base_model": args.base_model,
        "content_ids": content_ids,
        "style_ids": style_ids,
        "content_jsonl": str(content_jsonl),
        "style_jsonl": str(style_jsonl),
        "target_jsonl": str(target_jsonl),
        "output_dir": str(output_dir),
        "matrix_path": str(matrix_path),
        "manifest_path": str(manifest_path),
        "seed": args.seed,
        "num_content": len(content_ids),
        "num_style": len(style_ids),
        "num_target_cells": len(content_ids) * len(style_ids),
        "num_missing": len(missing_records),
        "missing_records": missing_records,
    }
    write_json(output_dir / "summary.json", summary)
    printed_summary = dict(summary)
    printed_summary["missing_records_preview"] = printed_summary.pop("missing_records")[:20]
    printed_summary["missing_records_full"] = str(output_dir / "summary.json")
    print(json.dumps(printed_summary, ensure_ascii=False, indent=2))
    return 1 if args.strict and missing_records else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        sys.exit(1)
