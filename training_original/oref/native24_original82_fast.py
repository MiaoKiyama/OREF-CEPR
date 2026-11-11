"""Same routing mathematics, faster exact integer accumulation and CPU kernels."""
import os,ctypes
from pathlib import Path
import numpy as np
import native24_original82_reference as ref
from native_build import routing_library
lib=ctypes.CDLL(str(routing_library()))
ptr=ctypes.c_void_p;integer=ctypes.c_int64;small=ctypes.c_int
lib.gains_cpu.argtypes=[ptr,ptr,ptr,ptr,ptr,integer,ptr,small]
lib.expected_cpu.argtypes=[ptr,ptr,ptr,integer,ptr,small]
lib.counts_cpu.argtypes=[ptr,ptr,ptr,ptr,integer,small,ptr,small]
for fn in (lib.gains_cpu,lib.expected_cpu,lib.counts_cpu):fn.restype=None
THREADS=int(os.environ.get('ORIGINAL82_CPU_THREADS','4'))
def addr(x):return None if x is None else x.ctypes.data
def byte(x):return np.ascontiguousarray(x,dtype=np.uint8)

def expected_gains(cm,hist,labels,current):
    hist=np.ascontiguousarray(hist,dtype=np.float64);labels=byte(labels);current=byte(current)
    mass=hist.sum(1);diag=cm.diagonal();union=cm.sum(0)+cm.sum(1)-diag
    summary=np.concatenate([diag,union]);out=np.empty(labels.shape,np.float64)
    lib.gains_cpu(addr(summary),addr(hist),addr(mass),addr(labels),addr(current),len(hist),addr(out),THREADS)
    return out

def expected_cm(hist,labels,choice):
    hist=np.ascontiguousarray(hist,dtype=np.float64);labels=byte(labels);choice=byte(choice);out=np.empty((24,24),np.float64)
    lib.expected_cpu(addr(hist),addr(labels),addr(choice),len(hist),addr(out),THREADS);return out

def count_hist(counts,pred,bins=None,nb=1,source=None):
    if counts.dtype!=np.uint16:
        assert counts.min(initial=0)>=0 and counts.max(initial=0)<=65535
    counts=np.ascontiguousarray(counts,dtype=np.uint16);pred=byte(pred)
    bins=None if bins is None else np.ascontiguousarray(bins,dtype=np.int64)
    source=None if source is None else byte(source);out=np.empty((nb,25,25),np.int64)
    lib.counts_cpu(addr(counts),addr(bins),addr(pred),addr(source),len(counts),nb,addr(out),THREADS);return out

def cm_from_groups(counts,labels,choice):
    return count_hist(counts,labels[np.arange(len(labels)),choice])[0]

def fast_calibrate(gains,fixed,counts,labels,eligible):
    action=gains.argmax(1).astype(np.uint8);best=gains.max(1)
    values=best[(best>0)&eligible]
    thresholds=np.unique(np.r_[0,np.quantile(values,np.linspace(0,1,21)) if len(values) else [],np.inf])
    bins=np.searchsorted(thresholds,best,side='left');bins[~eligible]=0
    pred=labels[np.arange(len(labels)),action];source=labels[:,2];nb=len(thresholds)+1
    deltas=count_hist(counts,pred,bins,nb,source)
    tails=deltas[::-1].cumsum(0)[::-1]
    base=np.full(len(labels),2,np.uint8);cm=ref.score_route(fixed,counts,labels,base)
    route=base;best_cm=cm;score=ref.miou(cm);chosen=None
    for i,t in enumerate(thresholds):
        candidate=cm+tails[i+1];s=ref.miou(candidate)
        if s>score+1e-12:score=s;best_cm=candidate;chosen=float(t)
    if chosen is not None:
        route=base.copy();ok=(best>chosen)&eligible;route[ok]=action[ok]
    assert np.array_equal(best_cm,ref.score_route(fixed,counts,labels,route))
    return route,best_cm,chosen

def class_threshold_search(counts,labels,gains,fixed,initial,score,passes=3,quantiles=25):
    n=len(labels);rows=np.arange(n);source=labels[:,2]
    action=gains.argmax(1).astype(np.uint8);best=gains.max(1)
    new=labels[rows,action];banks=[];selected=[]
    cm=fixed+cm_from_groups(counts,labels,initial)
    for cls in range(1,25):
        ix=np.flatnonzero(source==cls);yy=counts[ix];gg=best[ix];vals=gg[gg>0]
        thresholds=np.unique(np.r_[0,np.quantile(vals,np.linspace(0,1,quantiles)) if len(vals) else [],np.inf])
        nb=len(thresholds)+1;bins=np.searchsorted(thresholds,gg,side='left')
        by_new=count_hist(yy,new[ix],bins,nb)
        tails=np.cumsum(by_new[::-1],axis=0)[::-1]
        baseline=np.zeros((25,25),np.int64);baseline[:,cls]=yy.sum(0,dtype=np.int64)
        options=[]
        for j in range(len(thresholds)):
            delta=tails[j+1].copy();delta[:,cls]-=delta.sum(1);options.append(baseline+delta)
        original=count_hist(yy,labels[ix,initial[ix]])[0]
        banks.append((ix,thresholds,np.stack(options),original));selected.append(-1)
    history=[]
    for iteration in range(passes):
        changed=0
        for cls,(ix,thresholds,options,original) in enumerate(banks):
            current=original if selected[cls]<0 else options[selected[cls]]
            remainder=cm-current;current_score=score(cm);best_idx=selected[cls]
            for j,option in enumerate(options):
                value=score(remainder+option)
                if value>current_score+1e-12:current_score=value;best_idx=j
            if best_idx!=selected[cls]:cm=remainder+options[best_idx];selected[cls]=best_idx;changed+=1
        history.append(dict(iteration=iteration+1,changed_classes=changed,miou_percent=score(cm)*100))
        if not changed:break
    route=initial.copy();policy={}
    for cls,(ix,thresholds,options,original) in enumerate(banks,1):
        j=selected[cls-1]
        if j<0:policy[str(cls)]='keep_initial_global_policy';continue
        threshold=float(thresholds[j]);route[ix]=2;chosen=ix[best[ix]>threshold];route[chosen]=action[chosen]
        policy[str(cls)]=threshold if np.isfinite(threshold) else 'never_switch'
    return route,cm,dict(source_class_thresholds=policy,history=history)

def install():
    import native24_original82_policy as policy
    for name in ('expected_gains','expected_cm','cm_from_groups','fast_calibrate','class_threshold_search'):
        function=globals()[name];setattr(ref,name,function);setattr(policy,name,function)
