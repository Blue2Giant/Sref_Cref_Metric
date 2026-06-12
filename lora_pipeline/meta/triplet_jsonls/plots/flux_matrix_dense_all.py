#!/usr/bin/env python3
"""
Flux dense sub-matrix visualization.

Usage
-----
# render one specific Pareto config  (N_styles x M_contents):
    python3 flux_matrix_dense_all.py --config 8x4

# render all Pareto-optimal configs:
    python3 flux_matrix_dense_all.py --config all

Axes (after swap vs original):
  - Left column  (Y-axis) : content IDs  + sample thumbnail
  - Top row      (X-axis) : style  IDs   + sample thumbnail
  - Inner cells            : dual-LoRA triplet image

Pareto configs available:  36x1  22x2  12x3  8x4  5x5  4x6  3x7  2x10  1x14
"""

import argparse
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── Config ────────────────────────────────────────────────────────────────────
CELL_SIZE  = 320       # ↑ higher resolution
CELL_PAD   = 6
LABEL_H    = 28
FONT_SIZE  = 16

DATA_DIR = Path('/data/benchmark_metrics/lora_pipeline/meta/triplet_jsonls')
PLOT_DIR  = DATA_DIR / 'plots' / 'dense_submatrices'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

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
    for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
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

matrix_paths = {}
for s in fav_styles:
    row = {}
    for c in fav_contents:
        key = f'{c}__{s}'
        row[c] = pick_local(dual_data[key]) if key in dual_data else None
    matrix_paths[s] = row

active_styles   = [s for s in fav_styles   if any(matrix_paths[s][c] for c in fav_contents)]
active_contents = [c for c in fav_contents if any(matrix_paths[s][c] for s in active_styles)]

# ── Per-style bitmask & AND-closure ──────────────────────────────────────────
style_mask = {}
for s in active_styles:
    m = 0
    for i, c in enumerate(active_contents):
        if matrix_paths[s][c] is not None:
            m |= (1 << i)
    style_mask[s] = m

print('Computing AND-closure...')
closure = set(style_mask.values()); closure.add(0)
while True:
    new_vals = {a & b for a in closure for b in closure}
    if new_vals <= closure: break
    closure |= new_vals

# ── Enumerate Pareto-optimal dense sub-matrices ───────────────────────────────
all_results = []
for m in closure:
    if m == 0: continue
    valid = [s for s in active_styles if (style_mask[s] & m) == m]
    n_s, n_c = len(valid), bin(m).count('1')
    if n_s == 0 or n_c == 0: continue
    conts = [active_contents[i] for i in range(len(active_contents)) if m & (1 << i)]
    all_results.append({'score': n_s*n_c, 'n_styles': n_s, 'n_contents': n_c,
                        'styles': valid, 'contents': conts})

all_results.sort(key=lambda x: (-x['score'], -x['n_styles']))

def pareto_frontier(results):
    pareto, seen = [], set()
    for r in results:
        dominated = any(
            (r2['n_styles'] >= r['n_styles'] and r2['n_contents'] > r['n_contents'])
            or (r2['n_styles'] > r['n_styles'] and r2['n_contents'] >= r['n_contents'])
            for r2 in results if r2 is not r
        )
        if not dominated:
            key = (r['n_styles'], r['n_contents'])
            if key not in seen:
                seen.add(key); pareto.append(r)
    pareto.sort(key=lambda x: (-x['n_styles'], -x['n_contents']))
    return pareto

pareto = pareto_frontier(all_results)

print('\nPareto-optimal configs (use --config NxM):')
print('  %-10s  %-8s  %-10s  %s' % ('--config', 'styles', 'contents', 'score'))
for r in pareto:
    tag = '%dx%d' % (r['n_styles'], r['n_contents'])
    print('  %-10s  %-8d  %-10d  %d' % (tag, r['n_styles'], r['n_contents'], r['score']))

# save pareto_configs.json (IDs)
id_records = [{'config': '%dx%d' % (r['n_styles'], r['n_contents']),
               'score': r['score'], 'n_styles': r['n_styles'],
               'n_contents': r['n_contents'],
               'style_ids': r['styles'], 'content_ids': r['contents']}
              for r in pareto]
json_path = PLOT_DIR / 'pareto_configs.json'
try:
    with open(json_path, 'w') as f:
        json.dump(id_records, f, indent=2)
except PermissionError:
    json_path = DATA_DIR / 'plots' / 'pareto_configs.json'
    with open(json_path, 'w') as f:
        json.dump(id_records, f, indent=2)

# ── Parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='all',
                    help='NxM  (e.g. 8x4) or "all" to render every Pareto config')
args = parser.parse_args()

if args.config == 'all':
    to_render = pareto
else:
    try:
        ns, nc = map(int, args.config.lower().split('x'))
    except ValueError:
        raise SystemExit('--config must be NxM (e.g. 8x4) or "all"')
    matches = [r for r in pareto if r['n_styles'] == ns and r['n_contents'] == nc]
    if not matches:
        raise SystemExit(f'No Pareto config found for {ns}x{nc}. '
                         f'Available: {["%dx%d"%(r["n_styles"],r["n_contents"]) for r in pareto]}')
    to_render = matches

# ── Render ────────────────────────────────────────────────────────────────────
# Layout (axes swapped vs v1):
#   Top row   (X-axis) = STYLES    ← columns
#   Left col  (Y-axis) = CONTENTS  ← rows

COLOR_BG_HDR    = (195, 220, 250)
COLOR_LABEL_HDR = (20, 60, 130)
COLOR_LABEL_FG  = (40, 40, 40)

STEP_W = CELL_SIZE + CELL_PAD
STEP_H = CELL_SIZE + CELL_PAD + LABEL_H
font   = get_font(FONT_SIZE)

def render(cfg):
    styles   = cfg['styles']    # X-axis  (columns)
    contents = cfg['contents']  # Y-axis  (rows)
    n_cols, n_rows = len(styles), len(contents)

    canvas_w = STEP_W * (n_cols + 1) + CELL_PAD
    canvas_h = STEP_H * (n_rows + 1) + CELL_PAD
    canvas = Image.new('RGB', (canvas_w, canvas_h), (245, 245, 245))
    draw   = ImageDraw.Draw(canvas)

    def paste_cell(img, x, y, label, bg, label_fg, is_header=False):
        draw.rectangle([x, y, x+CELL_SIZE-1, y+CELL_SIZE-1], fill=bg)
        if img is not None:
            canvas.paste(img, (x, y))
        ly = y + CELL_SIZE + 2
        draw.rectangle([x, ly, x+CELL_SIZE-1, ly+LABEL_H-1],
                       fill=(220, 232, 252) if is_header else (235, 235, 235))
        max_chars = max(1, CELL_SIZE // (FONT_SIZE // 2 + 2))
        draw.text((x+4, ly+4), str(label)[:max_chars], fill=label_fg, font=font)

    # Style header row (top, X-axis)
    for j, s in enumerate(styles):
        x = STEP_W * (j + 1) + CELL_PAD; y = CELL_PAD
        img = load_thumb(pick_local(style_data[s]), CELL_SIZE)
        paste_cell(img, x, y, s, COLOR_BG_HDR, COLOR_LABEL_HDR, is_header=True)

    # Content header col (left, Y-axis)
    for i, c in enumerate(contents):
        x = CELL_PAD; y = STEP_H * (i + 1) + CELL_PAD
        img = load_thumb(pick_local(content_data[c]), CELL_SIZE)
        paste_cell(img, x, y, c, COLOR_BG_HDR, COLOR_LABEL_HDR, is_header=True)

    # Matrix cells
    for i, c in enumerate(contents):
        for j, s in enumerate(styles):
            x = STEP_W * (j + 1) + CELL_PAD
            y = STEP_H * (i + 1) + CELL_PAD
            img = load_thumb(matrix_paths[s][c], CELL_SIZE)
            paste_cell(img, x, y, '', (255, 255, 255), COLOR_LABEL_FG)

    # Top-left corner
    draw.rectangle([CELL_PAD, CELL_PAD, CELL_PAD+CELL_SIZE-1, CELL_PAD+CELL_SIZE+LABEL_H],
                   fill=(200, 210, 225))
    draw.text((CELL_PAD+6, CELL_PAD+CELL_SIZE//2-FONT_SIZE//2),
              'Content \\ Style', fill=(60, 80, 110), font=font)

    name = 'dense_%ds_%dc_v2.png' % (n_cols, n_rows)  # n_styles cols, n_contents rows
    out  = PLOT_DIR / name
    if not os.access(str(out.parent), os.W_OK):
        out = DATA_DIR / 'plots' / name
    canvas.save(str(out))
    print('  Saved %-30s  (%d x %d px)' % (out.name, canvas_w, canvas_h))

print('\nRendering...')
for cfg in to_render:
    render(cfg)

print('\nDone. Output:', PLOT_DIR)
