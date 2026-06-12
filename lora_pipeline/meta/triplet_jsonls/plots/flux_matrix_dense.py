#!/usr/bin/env python3
"""
Find the largest fully-dense sub-matrix from the Flux dual-LoRA triplet data,
then draw a new matrix visualization for it.

"Fully dense" means every (style, content) pair in the sub-matrix has a triplet image.

Algorithm
---------
1. Represent each style as a bitmask over the 23 favorite contents
   (bit i = 1  ⟺  style covers content i).
2. Build the AND-closure of all style bitmasks – this gives every possible
   "shared content subset".  Only these subsets can be the content axis of a
   fully dense matrix.
3. For each candidate content-mask m, count how many styles cover *all* contents
   in m.  Score = popcount(m) × n_styles.
4. Pick the mask with the highest score; resolve ties by preferring more styles,
   then more contents.
5. Render the found sub-matrix as a new PNG.
"""

import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── Config ────────────────────────────────────────────────────────────────────
CELL_SIZE  = 160
CELL_PAD   = 4
LABEL_H    = 22
FONT_SIZE  = 13

DATA_DIR = Path('/data/benchmark_metrics/lora_pipeline/meta/triplet_jsonls')
OUT_PATH = DATA_DIR / 'plots' / 'flux_matrix_dense.png'

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
    for p in paths:
        if p.startswith('/mnt/'):
            return p
    return paths[0] if paths else None

def load_thumb(path, size):
    if not path or not os.path.exists(path):
        return None
    try:
        img = Image.open(path).convert('RGB')
        w, h = img.size
        m = min(w, h)
        img = img.crop(((w-m)//2, (h-m)//2, (w+m)//2, (h+m)//2))
        return img.resize((size, size), Image.LANCZOS)
    except Exception:
        return None

def get_font(size):
    for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
              '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf']:
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

fav_styles   = [s for s in FAV_STYLES   if s in style_data]
fav_contents = [c for c in FAV_CONTENTS if c in content_data]
print(f'Fav styles: {len(fav_styles)}, fav contents: {len(fav_contents)}')

# ── Build coverage matrix ─────────────────────────────────────────────────────
# matrix[s][c] = image_path or None  (key: content__style)
matrix = {}
for s in fav_styles:
    row = {}
    for c in fav_contents:
        key = f'{c}__{s}'
        row[c] = pick_local(dual_data[key]) if key in dual_data else None
    matrix[s] = row

active_styles   = [s for s in fav_styles   if any(matrix[s][c] for c in fav_contents)]
active_contents = [c for c in fav_contents if any(matrix[s][c] for s in active_styles)]
print(f'Active styles: {len(active_styles)}, active contents: {len(active_contents)}')

# ── Compute per-style content bitmask ─────────────────────────────────────────
# bit i set  ⟺  style covers active_contents[i]
style_mask = {}
for s in active_styles:
    m = 0
    for i, c in enumerate(active_contents):
        if matrix[s][c] is not None:
            m |= (1 << i)
    style_mask[s] = m

# ── AND-closure of style masks ────────────────────────────────────────────────
# Every element of the closure is a valid candidate for the content-axis mask.
print('Computing AND-closure...')
closure = set(style_mask.values())
closure.add(0)          # include 0 to simplify, we skip it when scoring
while True:
    new_vals = {a & b for a in closure for b in closure}
    if new_vals <= closure:
        break
    closure |= new_vals
print(f'AND-closure size: {len(closure)}')

# ── Find best dense sub-matrix ────────────────────────────────────────────────
best_score = 0
best_content_mask = 0
best_valid_styles = []

for m in closure:
    if m == 0:
        continue
    valid = [s for s in active_styles if (style_mask[s] & m) == m]
    n_s = len(valid)
    n_c = bin(m).count('1')
    score = n_s * n_c
    if (score > best_score
            or (score == best_score and n_s > len(best_valid_styles))):
        best_score = score
        best_content_mask = m
        best_valid_styles = valid

best_contents = [active_contents[i]
                 for i in range(len(active_contents))
                 if best_content_mask & (1 << i)]

print(f'\n=== Best dense sub-matrix ===')
print(f'  Styles  : {len(best_valid_styles)}')
print(f'  Contents: {len(best_contents)}')
print(f'  Total cells: {len(best_valid_styles) * len(best_contents)}  (all filled)')
print(f'  Style IDs  : {best_valid_styles}')
print(f'  Content IDs: {best_contents}')

# ── Render the dense matrix ───────────────────────────────────────────────────
dense_styles   = best_valid_styles
dense_contents = best_contents

STEP_W = CELL_SIZE + CELL_PAD
STEP_H = CELL_SIZE + CELL_PAD + LABEL_H

n_rows = len(dense_styles)
n_cols = len(dense_contents)

canvas_w = STEP_W * (n_cols + 1) + CELL_PAD
canvas_h = STEP_H * (n_rows + 1) + CELL_PAD
print(f'Canvas: {canvas_w} x {canvas_h}')

canvas = Image.new('RGB', (canvas_w, canvas_h), (245, 245, 245))
draw   = ImageDraw.Draw(canvas)
font   = get_font(FONT_SIZE)

COLOR_BG_HDR   = (195, 220, 250)
COLOR_LABEL_HDR = (20, 60, 130)
COLOR_LABEL_FG  = (40, 40, 40)

def paste_cell(img_or_none, x, y, label, bg, label_fg, is_header=False):
    draw.rectangle([x, y, x+CELL_SIZE-1, y+CELL_SIZE-1], fill=bg)
    if img_or_none is not None:
        canvas.paste(img_or_none, (x, y))
    label_y = y + CELL_SIZE + 2
    draw.rectangle([x, label_y, x+CELL_SIZE-1, label_y+LABEL_H-1],
                   fill=(220, 232, 252) if is_header else (235, 235, 235))
    max_chars = max(1, CELL_SIZE // (FONT_SIZE // 2 + 2))
    draw.text((x+3, label_y+3), str(label)[:max_chars], fill=label_fg, font=font)

# Content header (top row)
for j, c in enumerate(dense_contents):
    x = STEP_W * (j + 1) + CELL_PAD
    y = CELL_PAD
    img = load_thumb(pick_local(content_data[c]), CELL_SIZE)
    paste_cell(img, x, y, c, COLOR_BG_HDR, COLOR_LABEL_HDR, is_header=True)

# Style header (left column)
for i, s in enumerate(dense_styles):
    x = CELL_PAD
    y = STEP_H * (i + 1) + CELL_PAD
    img = load_thumb(pick_local(style_data[s]), CELL_SIZE)
    paste_cell(img, x, y, s, COLOR_BG_HDR, COLOR_LABEL_HDR, is_header=True)

# Matrix cells (all guaranteed to have an image)
for i, s in enumerate(dense_styles):
    for j, c in enumerate(dense_contents):
        x = STEP_W * (j + 1) + CELL_PAD
        y = STEP_H * (i + 1) + CELL_PAD
        img = load_thumb(matrix[s][c], CELL_SIZE)
        paste_cell(img, x, y, '', (255, 255, 255), COLOR_LABEL_FG)

# Top-left corner
draw.rectangle([CELL_PAD, CELL_PAD,
                CELL_PAD+CELL_SIZE-1, CELL_PAD+CELL_SIZE+LABEL_H],
               fill=(200, 210, 225))
draw.text((CELL_PAD+4, CELL_PAD+CELL_SIZE//2),
          'Style \\ Content', fill=(60, 80, 110), font=font)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
canvas.save(str(OUT_PATH))
print(f'\nSaved: {OUT_PATH}')
