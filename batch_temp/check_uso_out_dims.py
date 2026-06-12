from pathlib import Path
from PIL import Image
from collections import Counter
root=Path('/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content/uso')
dims=Counter()
for p in root.glob('*.png'):
    try:
        dims[Image.open(p).size]+=1
    except Exception as e: print('err',p,e)
print('count',sum(dims.values()),'unique',len(dims),'top',dims.most_common(20))
