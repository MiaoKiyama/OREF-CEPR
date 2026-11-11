import numpy as np
import ctypes
from pathlib import Path
from scipy.ndimage import distance_transform_edt
ROOT=Path(__file__).resolve().parent.parent
AREA=np.array([1,4,16,64,256,1024,4096,16384,65536,262144],np.int64)

SHARE=np.array([0,.1,.25,.4,.55,.7,.85,1.0])

DIFF=np.array([-.75,-.4,-.15,0,.15,.4,.75,1.0])

ASPECT=np.array([1.25,1.5,2,3,5,10,20,1e12])

FILL=np.array([.1,.25,.4,.55,.7,.85,.95,1.0])

COMPACT=np.array([4,5,6,8,12,20,40,1e12])

LEVEL=np.array([1,3,5,7,9,11,13,14])

def bins(x,edges):return np.searchsorted(edges,x,side='left').clip(0,len(edges)-1).astype(np.uint8)

def observables(sig,previous,pred):
    lib=ctypes.CDLL(str(ROOT/'context_refine.so'));ptr=lambda a:a.ctypes.data_as(ctypes.c_void_p)
    lib.label4.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int];lib.label4.restype=ctypes.c_int
    lib.region_context.argtypes=[ctypes.c_void_p]*11+[ctypes.c_int]*3
    height,width=pred.shape;lab=np.empty(pred.shape,np.uint32)
    n=lib.label4(ptr(pred),ptr(lab),height,width);assert 0<n<2000000
    area=np.zeros(n+1,np.uint64);boxes=np.zeros((n+1,4),np.int32);src=np.zeros(n+1,np.uint8)
    votes=np.zeros((6,n+1,25),np.uint32);perim=np.zeros(n+1,np.uint32);border=perim.copy();conf=np.zeros((n+1,2),np.uint64)
    lib.region_context(ptr(pred),ptr(lab),ptr(sig),ptr(previous),ptr(area),ptr(boxes),ptr(src),ptr(votes),ptr(perim),ptr(border),ptr(conf),height,width,n)
    ix=np.arange(n+1);valid=(src>0)&(area<=AREA[-1]);candidates=[]
    for j in [0,0,1,2,3,4,5]:
        rank=votes[j].copy();rank[:,0]=0;rank[ix,src]=0
        if len(candidates)==1:rank[ix,candidates[0]]=0
        dest=rank.argmax(1).astype(np.uint8);dest[(rank[ix,dest]==0)|~valid]=0;candidates.append(dest)
    candidates.append(np.where(valid,src,0).astype(np.uint8))
    totals=np.maximum(votes[:,:,1:].sum(2),1);ratios=votes/totals[:,:,None]
    bh=np.maximum(boxes[:,2]-boxes[:,0]+1,1);bw=np.maximum(boxes[:,3]-boxes[:,1]+1,1)
    common=[bins(np.maximum(bh,bw)/np.minimum(bh,bw),ASPECT),bins(area/(bh.astype(float)*bw),FILL),
            bins(perim/np.sqrt(np.maximum(area,1)),COMPACT),bins(border/np.maximum(perim,1),SHARE),
            bins(conf[:,0]/np.maximum(area,1),LEVEL),bins(conf[:,1]/np.maximum(area,1),LEVEL)]
    changed=1-votes[5,ix,src]/np.maximum(area,1)
    features=[]
    for dest in candidates:
        features.append(np.stack([*[bins(ratios[j,ix,dest],SHARE) for j in range(4)],
            bins(ratios[2,ix,dest]-ratios[2,ix,src],DIFF),*common,
            bins(ratios[4,ix,dest],SHARE),bins(ratios[5,ix,dest],SHARE),bins(changed,SHARE)]))
    return lab,dict(source=src,area=area,area_bin=bins(area,AREA),destination=np.stack(candidates),features=np.stack(features))
