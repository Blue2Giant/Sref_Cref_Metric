#!/usr/bin/env python3
import json
from pathlib import Path

DATA_DIR = Path('/data/benchmark_metrics/lora_pipeline/meta/triplet_jsonls')

def load_jsonl(path):
    data = {}
    with open(path) as f:
        for line in f:
            if line.strip(): data.update(json.loads(line))
    return data

FAV_STYLES = ['1011520','1022047','1022259','1030307','1054574','1068179','1113245','1117845','1122150','1122449','1139929','1164676','118910','1193428','1203292','1203257','1201096','1263836','1268061','1296276','1342609','1364354','1563511','1678867','1714169','2028655','2036394','2078235','2080967','597662','664934','739595','778939','788524','768332','825840','849738','863934','868267','958440','96784','974384','98833','985102','979767','992103','996317','1039259','1455014','1691207','1730560','1848430','863576','873771','899122','906883','944591','814596','71159','711692','711967','1135273']
FAV_CONTENTS = ['1184339','1209494','1122438','1103121','1110632','1140958','1155087','1195491','1198663','1617906','1650387','1718480','196992','1974457','1974002','211589','682070','802426','869338','988787','768492','1159198','738088','850003']

style_data   = load_jsonl(DATA_DIR / 'flux_style_one_lora.jsonl')
content_data = load_jsonl(DATA_DIR / 'flux_content_one_lora.jsonl')
dual_data    = load_jsonl(DATA_DIR / 'flux__dual_lora.jsonl')

fav_styles   = [s for s in FAV_STYLES   if s in style_data]
fav_contents = [c for c in FAV_CONTENTS if c in content_data]
matrix = {s: {c: (f'{c}__{s}' in dual_data) for c in fav_contents} for s in fav_styles}
active_styles   = [s for s in fav_styles   if any(matrix[s][c] for c in fav_contents)]
active_contents = [c for c in fav_contents if any(matrix[s][c] for s in active_styles)]

style_mask = {}
for s in active_styles:
    m = 0
    for i, c in enumerate(active_contents):
        if matrix[s][c]: m |= (1 << i)
    style_mask[s] = m

closure = set(style_mask.values()); closure.add(0)
while True:
    new_vals = {a & b for a in closure for b in closure}
    if new_vals <= closure: break
    closure |= new_vals

results = []
for m in closure:
    if m == 0: continue
    valid = [s for s in active_styles if (style_mask[s] & m) == m]
    n_s = len(valid); n_c = bin(m).count('1')
    results.append((n_s * n_c, n_s, n_c, m, valid))
results.sort(reverse=True)

print("Top-15 dense sub-matrices by score:")
print("score  styles  contents")
for sc, ns, nc, m, vs in results[:15]:
    print("%5d  %6d  %8d" % (sc, ns, nc))

print()
print("Best score for each min n_contents:")
for min_c in range(2, 12):
    cands = [(sc, ns, nc, m, vs) for sc, ns, nc, m, vs in results if nc >= min_c]
    if cands:
        sc, ns, nc, m, vs = cands[0]
        print("  min_contents>=%d: score=%d  (%d styles x %d contents)" % (min_c, sc, ns, nc))
