import numpy as np
import ctypes
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from . import multi as q
ROOT=Path(__file__).resolve().parent.parent
CONF=np.array([.5,.7,.8,.9,.95,.98,.995,1],np.float32)

RANGE=np.array([.01,.03,.07,.15,.25,.4,.65,1],np.float32)

STD=np.array([.005,.015,.03,.06,.1,.15,.25,1],np.float32)

GAP=np.array([.005,.015,.03,.07,.15,.3,.6,1],np.float32)

REL=np.array([-.2,-.1,-.04,0,.04,.1,.2,1],np.float32)

AREA=np.array([4,16,64,256,1024,4096,16384,65536],np.int64)

DOM=np.array([0,.25,.5,.65,.8,.9,.95,1],np.float64)

ASPECT=np.array([1.25,1.5,2,3,5,10,20,1e12],np.float64)

FILL=np.array([.1,.25,.4,.55,.7,.85,.95,1],np.float64)

def bins(value,edges):return np.searchsorted(edges,value,side='left').clip(0,7).astype(np.uint8)

def pixel_features(prob):
    mean=prob.mean(0);minimum=prob.min(0);spread=prob.max(0)-minimum
    low=np.partition(prob,1,axis=0)[:2]
    features=[bins(minimum,CONF),bins(mean,CONF),bins(prob.std(0),STD),bins(spread,RANGE),
              bins(low[1]-low[0],GAP),(prob<np.float32(.9)).sum(0).astype(np.uint8)]
    total=prob.sum(0)
    for i in range(7):features.append(bins(prob[i]-(total-prob[i])/np.float32(6),REL))
    return bins(mean,CONF),np.stack(features),np.stack([mean,spread,minimum])

def pixel_proposal(signature,current):
    obs=q.decode(signature);previous=obs['incumbent'];tag=(current!=previous).astype(np.uint8)
    dest=np.where(tag!=0,previous,0).astype(np.uint8)
    for name in ['candidate1','candidate2','old','raw']:
        cand=obs[name];use=(dest==0)&(cand>0)&(cand!=current);dest[use]=cand[use]
    return dest,tag,obs

def components(pred):
    lib=ctypes.CDLL(str(ROOT/'components.so'))
    ptr=lambda a:a.ctypes.data_as(ctypes.c_void_p)
    lib.label4.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int];lib.label4.restype=ctypes.c_int
    lib.component_stats.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*3
    pred=np.ascontiguousarray(pred,np.uint8);height,width=pred.shape
    lab=np.empty(pred.shape,np.uint32);n=lib.label4(ptr(pred),ptr(lab),height,width)
    assert 0<n<=2000000,('component_count_outside_budget',n)
    area=np.zeros(n+1,np.uint64);boxes=np.zeros((n+1,4),np.int32);classes=np.zeros(n+1,np.uint8)
    contact=np.zeros((n+1,25),np.uint32)
    lib.component_stats(ptr(pred),ptr(lab),ptr(area),ptr(boxes),ptr(classes),ptr(contact),height,width,n)
    total=contact[:,1:].sum(1);candidate=(contact[:,1:].argmax(1)+1).astype(np.uint8)
    share=contact[np.arange(n+1),candidate]/np.maximum(total,1)
    candidate[(total==0)|(area>AREA[-1])|(classes==0)]=0
    boxh=np.maximum(boxes[:,2]-boxes[:,0]+1,1);boxw=np.maximum(boxes[:,3]-boxes[:,1]+1,1)
    ratio=np.maximum(boxh,boxw)/np.maximum(np.minimum(boxh,boxw),1)
    fill=area.astype(np.float64)/(boxh.astype(np.float64)*boxw)
    geom=np.stack([bins(share,DOM),bins(ratio,ASPECT),bins(fill,FILL)])
    assert not ((candidate!=0)&(candidate==classes)).any()
    return lab,dict(source=classes,destination=candidate,area=area,area_bin=bins(area,AREA),geometry=geom)
