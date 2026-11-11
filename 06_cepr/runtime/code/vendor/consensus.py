import numpy as np
import ctypes
from pathlib import Path
from scipy.ndimage import distance_transform_edt
ORDER=np.array([2,4,6,3,5,0,1])

DIST=np.array([0,1,2,4,8,16,32,64],float)

CONF=np.array([.3,.4,.5,.6,.7,.8,.85,.9,.925,.95,.97,.98,.99,.995,1.],np.float32)

def keys(labs,pos,shape):
    h,w=shape;y,x=np.divmod(pos,w)
    roi=np.searchsorted(np.linspace(0,h,5,dtype=int)[1:],y,side='right')*8+np.searchsorted(np.linspace(0,w,9,dtype=int)[1:],x,side='right')
    return (((y//64)*((w+63)//64)+x//64)*33+roi)*25**7+labs.astype(np.int64)@(25**np.arange(6,-1,-1))

def observable_scene(folder,manifest,group_keys,parent):
    """Never opens gt/counts. Build current output and confidence from frozen predictions."""
    shape=tuple(manifest['shape']);n=int(np.prod(shape))
    base=np.empty(n,np.uint8);current=np.empty(n,np.uint8);agree=np.empty(n,bool)
    minbin=np.empty(n,np.uint8);meanbin=np.empty(n,np.uint8);mean=np.empty(n,np.float32)
    for ch in manifest['chunks']:
        lo=ch['start'];hi=lo+ch['count'];sl=slice(lo,hi)
        with np.load(folder/'compact'/ch['file'],allow_pickle=False) as z:
            labs=z['labels']
            if 'region_raw' in z.files:
                raw=np.ascontiguousarray(z['region_raw'].T).view(np.float32).reshape(3,ch['count'])
                pm,minimum=raw[0],raw[2]
            else:
                probs=np.ascontiguousarray(z['top1_fp32'].T).view(np.float32).reshape(7,ch['count'])
                pm,minimum=probs.mean(0),probs.min(0)
        assert labs.shape==(7,ch['count']) and np.isfinite(pm).all() and np.isfinite(minimum).all()
        assert np.all((minimum>=0)&(pm<=1))
        base[sl]=labs[6];current[sl]=labs[6];agree[sl]=np.all(labs==labs[6],axis=0)
        mean[sl]=pm
        minbin[sl]=np.searchsorted(CONF,minimum,side='left')
        meanbin[sl]=np.searchsorted(CONF,pm,side='left')
        ix=np.flatnonzero(~agree[sl])
        if len(ix):
            kk=keys(labs[ORDER][:,ix].T,lo+ix,shape);gi=np.searchsorted(group_keys,kk)
            np.testing.assert_array_equal(group_keys[gi],kk)
            current[lo+ix]=parent[gi]
    return dict(base=base.reshape(shape),current=current.reshape(shape),agree=agree.reshape(shape),
                minbin=minbin.reshape(shape),meanbin=meanbin.reshape(shape),mean=mean.reshape(shape))

def proposals(obs):
    """Nearest boundary's different-class neighbor, chosen by mean confidence. No truth."""
    pred=obs['current'];h,w=pred.shape
    dest=pred.copy();best=np.full(pred.shape,-1,np.float32)
    # Stable N,W,E,S order; never wrap around raster edges.
    for center,neighbor in [(np.s_[1:,:],np.s_[:-1,:]),(np.s_[:,1:],np.s_[:,:-1]),
                            (np.s_[:,:-1],np.s_[:,1:]),(np.s_[:-1,:],np.s_[1:,:])]:
        diff=pred[center]!=pred[neighbor];change=diff&(obs['mean'][neighbor]>best[center])
        best[center][change]=obs['mean'][neighbor][change];dest[center][change]=pred[neighbor][change]
    boundary=best>=0;del best
    if not boundary.any():return np.zeros(pred.shape,np.uint32)
    distance,near=distance_transform_edt(~boundary,return_indices=True)
    anchor=pred[near[0],near[1]];destination=dest[near[0],near[1]]
    eligible=obs['agree']&(distance<=DIST[-1])&(anchor==pred)&(destination!=pred)
    db=np.searchsorted(DIST,distance[eligible],side='left').astype(np.uint32)
    code=np.zeros(pred.shape,np.uint32)
    code[eligible]=(pred[eligible].astype(np.uint32)|(destination[eligible].astype(np.uint32)<<5)|(db<<10)
        |(obs['minbin'][eligible].astype(np.uint32)<<14)|(obs['meanbin'][eligible].astype(np.uint32)<<18))
    return code
