import numpy as np
import ctypes
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from . import refine as t
ROOT=Path(__file__).resolve().parent.parent
SHARE=t.SHARE;DIFF=t.DIFF
CLASS_SHARE=np.array([0,.01,.03,.1,.25,.5,.75,1]);FRACTION=np.array([0,.0001,.0005,.001,.005,.01,.05,1])
AREA=np.array([1,4,16,64,256,1024,4096,16384,65536,262144,1048576],np.int64)

SPREAD=np.array([0,.25,.5,1,2,4,7,14])

def observables(sig,previous,pred):
 lib=ctypes.CDLL(str(ROOT/'context_refresh.so'));ptr=lambda a:a.ctypes.data_as(ctypes.c_void_p)
 lib.label4.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int];lib.label4.restype=ctypes.c_int
 lib.region_context_wide.argtypes=[ctypes.c_void_p]*11+[ctypes.c_int]*3
 height,width=pred.shape;lab=np.empty(pred.shape,np.uint32);n=lib.label4(ptr(pred),ptr(lab),height,width);assert 0<n<2000000
 area=np.zeros(n+1,np.uint64);boxes=np.zeros((n+1,4),np.int32);src=np.zeros(n+1,np.uint8)
 votes=np.zeros((7,n+1,25),np.uint32);per=np.zeros(n+1,np.uint32);border=per.copy();conf=np.zeros((n+1,2),np.uint64)
 lib.region_context_wide(ptr(pred),ptr(lab),ptr(sig),ptr(previous),ptr(area),ptr(boxes),ptr(src),ptr(votes),ptr(per),ptr(border),ptr(conf),height,width,n)
 ix=np.arange(n+1);valid=(src>0)&(area<=AREA[-1]);candidates=[]
 rank=votes[0].copy();rank[:,0]=0;rank[ix,src]=0
 for k in range(3):
  dst=rank.argmax(1).astype(np.uint8);dst[(rank[ix,dst]==0)|~valid]=0;candidates.append(dst);rank[ix,dst]=0
 for j in range(1,7):
  rank=votes[j].copy();rank[:,0]=0;rank[ix,src]=0;dst=rank.argmax(1).astype(np.uint8);dst[(rank[ix,dst]==0)|~valid]=0;candidates.append(dst)
 candidates.append(np.where(valid,src,0).astype(np.uint8))
 total=np.maximum(votes[:,:,1:].sum(2),1);ratios=votes/total[:,:,None]
 bh=np.maximum(boxes[:,2]-boxes[:,0]+1,1);bw=np.maximum(boxes[:,3]-boxes[:,1]+1,1)
 class_area=np.bincount(src,weights=area,minlength=25);image_area=max(float(area[src>0].sum()),1)
 common=[t.bins(np.maximum(bh,bw)/np.minimum(bh,bw),t.ASPECT),t.bins(area/(bh.astype(float)*bw),t.FILL),
  t.bins(per/np.sqrt(np.maximum(area,1)),t.COMPACT),t.bins(border/np.maximum(per,1),SHARE),
  t.bins(conf[:,0]/np.maximum(area,1),t.LEVEL),t.bins(conf[:,1]/np.maximum(area,1),t.LEVEL)]
 changed=1-votes[6,ix,src]/np.maximum(area,1);class_share=area/np.maximum(class_area[src],1)
 top=np.partition(votes[0,:,1:],-2,axis=1)[:,-2:]
 features=[]
 for dst in candidates:
  spatial=np.stack([ratios[j,ix,dst] for j in range(5)])
  bestother=np.where(votes[0,ix,dst]==top[:,1],top[:,0],top[:,1])
  margin=(votes[0,ix,dst].astype(float)-bestother)/total[0]
  features.append(np.stack([*[t.bins(spatial[j],SHARE) for j in range(5)],
   t.bins(ratios[2,ix,dst]-ratios[2,ix,src],DIFF),t.bins(ratios[4,ix,dst]-ratios[4,ix,src],DIFF),*common,
   t.bins(ratios[5,ix,dst],SHARE),t.bins(ratios[6,ix,dst],SHARE),t.bins(changed,SHARE),
   t.bins(class_share,CLASS_SHARE),t.bins(class_area[src]/image_area,FRACTION),t.bins(class_area[dst]/image_area,FRACTION),
   t.bins(spatial.max(0),SHARE),(spatial[1:]>=.25).sum(0).astype(np.uint8),t.bins(margin,DIFF),
   t.bins((conf[:,1].astype(float)-conf[:,0])/np.maximum(area,1),SPREAD)]))
 return lab,dict(source=src,area=area,area_bin=t.bins(area,AREA),destination=np.stack(candidates),features=np.stack(features))
