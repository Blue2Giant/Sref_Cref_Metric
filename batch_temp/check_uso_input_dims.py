from pathlib import Path
from PIL import Image
import json, collections, os
root=Path('/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content')
for sub in ['cref','sref']:
    dims=collections.Counter()
    bad=[]
    files=[]
    for p in sorted((root/sub).iterdir()):
        if p.suffix.lower() in ['.png','.jpg','.jpeg','.webp','.bmp','.tiff','.avif','.heic']:
            try:
                im=Image.open(p)
                dims[im.size]+=1
                files.append(p)
                if im.size[0] % 16 or im.size[1] % 16:
                    bad.append((p.name, im.size))
            except Exception as e:
                bad.append((p.name, str(e)))
    print(sub, 'num', len(files), 'unique dims', len(dims), 'top10', dims.most_common(10), 'non16', len(bad), 'bad_examples', bad[:5])
try:
    prompts=json.load(open(root/'prompts.json'))
    print('prompts', len(prompts), 'sample keys', list(prompts)[:5])
except Exception as e:
    print('prompts err',e)
out=root/'uso'
print('out exists', out.exists())
if out.exists():
    outs=list(out.glob('*.png'))
    print('out png count',len(outs),'first',outs[:3])
