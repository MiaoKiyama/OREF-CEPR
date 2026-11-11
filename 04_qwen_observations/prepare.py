"""Prepare the original 4x8 RGB context/target views from raw FBP images."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
Image.MAX_IMAGE_PIXELS=None

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scene-list',type=Path,help='One scene ID per line; otherwise all TIFFs')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    paths=([a.images/(s.strip()+'.tif') for s in a.scene_list.read_text().splitlines() if s.strip()]
           if a.scene_list else sorted(a.images.glob('*.tif')))
    if not paths:raise ValueError('No images')
    rows=[]
    for path in paths:
        with Image.open(path) as im:raw=np.asarray(im)
        if raw.ndim!=3 or raw.shape[2]!=4 or raw.dtype!=np.uint8:raise ValueError('Expected uint8 NIR,R,G,B')
        rgb=Image.fromarray(np.ascontiguousarray(raw[:,:,1:4]));h,w=raw.shape[:2]
        ys,xs=np.linspace(0,h,5,dtype=int),np.linspace(0,w,9,dtype=int)
        image_sha=hashlib.sha256(path.read_bytes()).hexdigest()
        for j in range(32):
            iy,ix=divmod(j,8);y,x=int(ys[iy]),int(xs[ix]);hh,ww=int(ys[iy+1]-y),int(xs[ix+1]-x)
            row=dict(scene=path.stem,roi=j,image_sha256=image_sha)
            for role,box,size in [('context',(x-ww//2,y-hh//2,x-ww//2+2*ww,y-hh//2+2*hh),512),
                                  ('target',(x,y,x+ww,y+hh),256)]:
                name=f'{path.stem}__grid{j:02d}_{role}.png'
                ImageOps.pad(rgb.crop(box),(size,size),method=Image.Resampling.LANCZOS,color=(0,0,0)).save(a.output/name)
                row[role]=name
                row[role+'_sha256']=hashlib.sha256((a.output/name).read_bytes()).hexdigest()
            rows.append(row)
        rgb.close()
    (a.output/'requests.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(dict(scenes=len(paths),requests=len(rows),labels_read=False)))

if __name__=='__main__':main()
