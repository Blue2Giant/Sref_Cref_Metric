#!/usr/bin/env python3
"""
Flux dual LoRA matrix visualization.
Y-axis: favorite style IDs (with sample image + ID label)
X-axis: favorite content IDs (with sample image + ID label)
Cell: dual LoRA triplet image for that (style, content) combination
Empty rows/columns (no any triplet hit) are removed.
"""

import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── Config ────────────────────────────────────────────────────────────────────
CELL_SIZE   = 128   # image thumbnail size (square)
CELL_PAD    = 3     # gap between cells
LABEL_H     = 20    # text label height below each image
FONT_SIZE   = 12

DATA_DIR = Path('/data/benchmark_metrics/lora_pipeline/meta/triplet_jsonls')
OUT_PATH = DATA_DIR / 'plots' / 'flux_matrix.png'

FAV_STYLES = [
    '1011520','1022047','1022259','1030307','1054574','1068179','1113245','1117845',
    '1122150','1122449','1139929','1164676','118910','1193428','1203292','1203257',
    '1201096','1263836','1268061','1296276','1342609','1364354','1563511','1678867',
    '1714169','2028655','2036394','2078235','2080967','597662','664934','739595',
    '778939','788524','768332','825840','849738','863934','868267','958440','96784',
    '974384','98833','985102','979767','992103','996317','1039259','1455014','1691207',
    '1730560','1848430','863576','873771','899122','906883','944591','814596','71159',
    '711692','711967','1135273',
]

FAV_CONTENTS = [
    '1184339','1209494','1122438','1103121','1110632','1140958','1155087','1195491',
    '1198663','1617906','1650387','1718480','196992','1974457','1974002','211589',
    '682070','802426','869338','988787','768492','1159198','738088','850003',
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path):
    data = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                data.update(json.loads(line))
    return data


def pick_local(paths):
    """Return first /mnt/ path; fall back to first path."""
    for p in paths:
        if p.startswith('/mnt/'):
            return p
    return paths[0] if paths else None


def load_thumb(path, size):
    """Load image, center-crop to square, resize to size. Returns None on failure."""
    if not path or not os.path.exists(path):
        return None
    try:
        img = Image.open(path).convert('RGB')
        w, h = img.size
        m = min(w, h)
        img = img.crop(((w - m) // 2, (h - m) // 2,
                        (w + m) // 2, (h + m) // 2))
        img = img.resize((size, size), Image.LANCZOS)
        return img
    except Exception:
        return None


def get_font(size):
    candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading JSONL files...')
style_data   = load_jsonl(DATA_DIR / 'flux_style_one_lora.jsonl')
content_data = load_jsonl(DATA_DIR / 'flux_content_one_lora.jsonl')
dual_data    = load_jsonl(DATA_DIR / 'flux__dual_lora.jsonl')
print(f'  style entries: {len(style_data)}, content entries: {len(content_data)}, dual: {len(dual_data)}')

# ── Filter to favorites present in data ───────────────────────────────────────
fav_styles   = [s for s in FAV_STYLES   if s in style_data]
fav_contents = [c for c in FAV_CONTENTS if c in content_data]
print(f'Fav styles found: {len(fav_styles)}/{len(FAV_STYLES)}, '
      f'fav contents: {len(fav_contents)}/{len(FAV_CONTENTS)}')

# ── Build matrix (key: content_id__style_id per jsonl convention) ─────────────
# matrix[s][c] = image_path or None
matrix = {}
for s in fav_styles:
    row = {}
    for c in fav_contents:
        key = f'{c}__{s}'
        if key in dual_data:
            row[c] = pick_local(dual_data[key])
        else:
            row[c] = None
    matrix[s] = row

# ── Remove rows/columns with zero hits ────────────────────────────────────────
active_styles   = [s for s in fav_styles   if any(matrix[s][c] is not None for c in fav_contents)]
active_contents = [c for c in fav_contents if any(matrix[s][c] is not None for s in active_styles)]

n_rows = len(active_styles)
n_cols = len(active_contents)
hits   = sum(1 for s in active_styles for c in active_contents if matrix[s][c] is not None)
print(f'Active styles: {n_rows}, active contents: {n_cols}, filled cells: {hits}/{n_rows*n_cols}')

# ── Layout constants ──────────────────────────────────────────────────────────
STEP_W  = CELL_SIZE + CELL_PAD   # column stride
STEP_H  = CELL_SIZE + CELL_PAD + LABEL_H  # row stride (image + label)

# Header col width = STEP_W, header row height = STEP_H
canvas_w = STEP_W  * (n_cols + 1) + CELL_PAD
canvas_h = STEP_H  * (n_rows + 1) + CELL_PAD

print(f'Canvas size: {canvas_w} x {canvas_h}')

canvas = Image.new('RGB', (canvas_w, canvas_h), (245, 245, 245))
draw   = ImageDraw.Draw(canvas)
font   = get_font(FONT_SIZE)

COLOR_BG_HEADER = (210, 225, 245)   # bluish tint for headers
COLOR_BG_EMPTY  = (215, 215, 215)   # gray for missing cells
COLOR_EMPTY_X   = (180, 180, 180)
COLOR_LABEL_FG  = (40, 40, 40)
COLOR_LABEL_HDR = (20, 60, 120)


def paste_cell(img_or_none, x, y, label, bg_color, label_color, is_header=False):
    """Paste one cell (image + label) onto canvas."""
    # Background
    draw.rectangle([x, y, x + CELL_SIZE - 1, y + CELL_SIZE - 1], fill=bg_color)

    if img_or_none is not None:
        canvas.paste(img_or_none, (x, y))
    else:
        # Draw an X for empty
        draw.line([x, y, x + CELL_SIZE - 1, y + CELL_SIZE - 1], fill=COLOR_EMPTY_X, width=2)
        draw.line([x + CELL_SIZE - 1, y, x, y + CELL_SIZE - 1], fill=COLOR_EMPTY_X, width=2)

    # Label
    label_y = y + CELL_SIZE + 2
    draw.rectangle([x, label_y, x + CELL_SIZE - 1, label_y + LABEL_H - 1],
                   fill=(230, 235, 245) if is_header else (238, 238, 238))
    # Truncate label to fit
    max_chars = CELL_SIZE // (FONT_SIZE // 2 + 1)
    short = label[:max_chars]
    draw.text((x + 2, label_y + 2), short, fill=label_color, font=font)


# ── Draw content header row (top) ─────────────────────────────────────────────
for j, c in enumerate(active_contents):
    x = STEP_W * (j + 1) + CELL_PAD
    y = CELL_PAD
    img = load_thumb(pick_local(content_data[c]), CELL_SIZE)
    paste_cell(img, x, y, c, COLOR_BG_HEADER, COLOR_LABEL_HDR, is_header=True)

# ── Draw style header column (left) ───────────────────────────────────────────
for i, s in enumerate(active_styles):
    x = CELL_PAD
    y = STEP_H * (i + 1) + CELL_PAD
    img = load_thumb(pick_local(style_data[s]), CELL_SIZE)
    paste_cell(img, x, y, s, COLOR_BG_HEADER, COLOR_LABEL_HDR, is_header=True)

# ── Draw matrix cells ─────────────────────────────────────────────────────────
for i, s in enumerate(active_styles):
    for j, c in enumerate(active_contents):
        x = STEP_W * (j + 1) + CELL_PAD
        y = STEP_H * (i + 1) + CELL_PAD
        img_path = matrix[s][c]
        img = load_thumb(img_path, CELL_SIZE) if img_path else None
        paste_cell(img, x, y, '', COLOR_BG_EMPTY if img is None else (255, 255, 255),
                   COLOR_LABEL_FG)

# ── Top-left corner label ─────────────────────────────────────────────────────
draw.rectangle([CELL_PAD, CELL_PAD, CELL_PAD + CELL_SIZE - 1, CELL_PAD + CELL_SIZE - 1],
               fill=(200, 200, 200))
draw.text((CELL_PAD + 4, CELL_PAD + CELL_SIZE // 2 - 8),
          'Style\\nContent', fill=(80, 80, 80), font=font)

# ── Save ──────────────────────────────────────────────────────────────────────
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
canvas.save(str(OUT_PATH))
print(f'Saved: {OUT_PATH}  ({canvas_w}x{canvas_h} px)')
