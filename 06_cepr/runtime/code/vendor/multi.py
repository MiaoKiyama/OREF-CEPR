import numpy as np
import ctypes
from pathlib import Path
from scipy.ndimage import distance_transform_edt
OLD_EDGES = np.array([0, 2, 8, 32, 64, 128, 256, 512, 1024])

SAMPLE_RADII = [2, 8, 32, 128]

FIELDS = [('incumbent',5),('raw',5),('old',5),('oldbin',4),('minbin',4),
          ('meanbin',4),('candidate1',5),('candidate2',5),('distance1',2),('distance2',2)]

def encode(fields):
    out=np.zeros_like(fields['incumbent'],dtype=np.uint64);shift=0
    for name,bits in FIELDS:
        value=fields[name]
        assert np.all(value < (1<<bits))
        out |= value.astype(np.uint64)<<shift;shift+=bits
    return out

def decode(code):
    out={};shift=0
    for name,bits in FIELDS:
        out[name]=((code>>shift)&((1<<bits)-1)).astype(np.uint8);shift+=bits
    return out

def neighborhood(pred, confidence, incumbent, eligible, old_candidate=None):
    """Two distinct alternatives from 32 sampled neighbors, without targets.

Scores sum neighbor Top1 confidence / sqrt(Chebyshev sampling radius).
Each class needs >=2 supporting samples. Ties go to smaller class ID.
"""
    height,width=pred.shape
    dest=[np.zeros(pred.shape,np.uint8) for _ in range(2)]
    distance=[np.zeros(pred.shape,np.uint8) for _ in range(2)]
    for y0 in range(0,height,32):
        y1=min(height,y0+32);shape=(y1-y0,width);n=int(np.prod(shape))
        score=np.zeros((25,n),np.float32);votes=np.zeros((25,n),np.uint8)
        near=np.full((25,n),4,np.uint8)
        for ri,rad in enumerate(SAMPLE_RADII):
            for dy,dx in [(-rad,0),(0,-rad),(0,rad),(rad,0),(-rad,-rad),(-rad,rad),(rad,-rad),(rad,rad)]:
                ya,yb=max(y0,-dy),min(y1,height-dy)
                xa,xb=max(0,-dx),min(width,width-dx)
                if ya>=yb or xa>=xb:continue
                pos=((np.arange(ya,yb)[:,None]-y0)*width+np.arange(xa,xb)[None,:]).ravel()
                cls=pred[ya+dy:yb+dy,xa+dx:xb+dx].ravel()
                conf=confidence[ya+dy:yb+dy,xa+dx:xb+dx].ravel()
                score[cls,pos]+=conf/np.sqrt(float(rad))
                votes[cls,pos]+=1
                near[cls,pos]=np.minimum(near[cls,pos],ri)
        rows=np.arange(n);score[votes<2]=-1;score[0]=-1
        score[incumbent[y0:y1].ravel(),rows]=-1
        for j in range(2):
            cls=score.argmax(0).astype(np.uint8)
            use=(score[cls,rows]>0)&eligible[y0:y1].ravel()
            out=np.where(use,cls,0).astype(np.uint8)
            dest[j][y0:y1]=out.reshape(shape)
            distance[j][y0:y1]=np.where(use,near[cls,rows],0).reshape(shape)
            score[cls,rows]=-1
    assert not ((dest[0]!=0)&(dest[0]==dest[1])).any()
    return dest,distance
