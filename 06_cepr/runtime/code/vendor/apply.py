import numpy as np
import ctypes
from pathlib import Path
from scipy.ndimage import distance_transform_edt
import hashlib,zipfile
FIELDS = [('incumbent',5),('raw',5),('old',5),('oldbin',4),('minbin',4),('meanbin',4),
          ('candidate1',5),('candidate2',5),('distance1',2),('distance2',2)]

REGION_STAGES = ['boundary1','boundary2','context8','context32','context128','raw_mode','previous_mode','restore_reference']

REFRESH_STAGES = ['boundary1','boundary2','boundary3','context8','context32','context128','context512','raw_mode','previous_mode','restore_reference']

def array_sha(a): return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()

def save(p, **arrays):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(p, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=2) as z:
        for k, a in arrays.items():
            with z.open(k + '.npy', 'w', force_zip64=True) as f: np.lib.format.write_array(f, np.asarray(a), allow_pickle=False)

def codes_fields(code):return [((code>>s)&mask).astype(np.uint8) for s,mask in [(0,31),(5,31),(10,15),(14,15),(18,15)]]

def band(code,rules,offset=0):
    s,d,di,mb,ab=codes_fields(code);table=np.zeros((5,25,25),np.int16);table[2]=-1;table[4]=-1
    for key,v in rules.items():a,b=map(int,key.split(':'));table[:,a,b]=v
    metric,dl,dh,cl,ch=table[:,s,d];cf=np.where(metric==0,mb,ab).astype(np.int16)-offset
    use=(code!=0)&(di>=dl)&(di<=dh)&(cf>=cl)&(cf<=ch)
    return np.where(use,d,s).astype(np.uint8)

def consensus(code,rules):
    s,d,di,mb,ab=codes_fields(code);table=np.zeros((3,25,25),np.int16);table[1:]=-1
    for key,v in rules.items():a,b=map(int,key.split(':'));table[:,a,b]=v
    metric,dist,conf=table[:,s,d];use=(code!=0)&(di<=dist)&(np.where(metric==0,mb,ab)<=conf)
    return np.where(use,d,s).astype(np.uint8)

def decode(sig):
    out={};shift=0
    for name,bits in FIELDS:out[name]=((sig>>shift)&((1<<bits)-1)).astype(np.uint8);shift+=bits
    return out

def multi(obs,policy):
    current=obs['incumbent'].copy()
    for stage in policy['stages']:
        if stage['kind']=='joint':
            rollback=current!=obs['raw'];dst=np.where(rollback,obs['raw'],obs['old']);tag=rollback.astype(np.uint8);di=obs['oldbin']
        else:
            j=stage['kind'][-1];dst=obs['candidate'+j];di=obs['distance'+j];tag=np.zeros(len(current),np.uint8)
        table=np.zeros((5,2,25,25),np.int16);table[2]=-1;table[4]=-1
        for k,v in stage['rules'].items():a,b,c=map(int,k.split(':'));table[:,a,b,c]=v
        metric,dl,dh,cl,ch=table[:,tag,current,dst];cb=np.where(metric==0,obs['minbin'],obs['meanbin'])
        use=(dst>0)&(dst!=current)&(di>=dl)&(di<=dh)&(cb>=cl)&(cb<=ch)
        current=np.where(use,dst,current).astype(np.uint8)
    return current

def bins(v,edges):return np.searchsorted(np.asarray(edges),v,side='left').clip(0,len(edges)-1).astype(np.uint8)

def confidence_features(p):
    conf=np.array([.5,.7,.8,.9,.95,.98,.995,1],np.float32);mean=p.mean(0);low=np.partition(p,1,axis=0)[:2]
    f=[bins(p.min(0),conf),bins(mean,conf),bins(p.std(0),np.array([.005,.015,.03,.06,.1,.15,.25,1],np.float32)),
       bins(p.max(0)-p.min(0),np.array([.01,.03,.07,.15,.25,.4,.65,1],np.float32)),
       bins(low[1]-low[0],np.array([.005,.015,.03,.07,.15,.3,.6,1],np.float32)),(p<np.float32(.9)).sum(0).astype(np.uint8)]
    total=p.sum(0)
    for i in range(7):f.append(bins(p[i]-(total-p[i])/np.float32(6),np.array([-.2,-.1,-.04,0,.04,.1,.2,1],np.float32)))
    return bins(mean,conf),np.stack(f)

def accept(src,dst,tag,x,features,policy):
    table=np.zeros((5,2,25,25),np.int16);table[2]=-1;table[4]=-1
    for k,v in policy['rules'].items():a,b,c=map(int,k.split(':'));table[:,a,b,c]=v
    f,xl,xh,yl,yh=table[:,tag,src,dst];y=features[f,np.arange(len(src))]
    return (dst>0)&(dst!=src)&(x>=xl)&(x<=xh)&(y>=yl)&(y<=yh)

def region_apply(obs,policy,names):
    dsts=obs['destination'].copy()
    if 'maximum_area' in policy:dsts[:,obs['area']>policy['maximum_area']]=0
    pred=obs['source'].copy()
    if 'initial_guard' in policy:pred=guard(obs,pred,policy['initial_guard'])
    for stage in policy['stages']:
        j=names.index(stage['kind']);dst=dsts[j];f=obs['features'][j];x=obs['area_bin']
        table=np.zeros((5,25,25),np.int16);table[2]=-1;table[4]=-1
        for k,v in stage['rules'].items():a,b=map(int,k.split(':'));table[:,a,b]=v
        feature,xl,xh,yl,yh=table[:,pred,dst];y=f[feature,np.arange(len(pred))]
        use=(dst>0)&(dst!=pred)&(x>=xl)&(x<=xh)&(y>=yl)&(y<=yh);pred=np.where(use,dst,pred).astype(np.uint8)
    return guard(obs,pred,policy.get('final_guard')) if 'original_source' in obs else pred

def guard(obs,pred,g):
    if g is None:return pred
    support=np.zeros(len(pred),np.uint8)
    for j,dst in enumerate(obs['destination']):use=dst==pred;support[use]=obs['features'][j,:4].max(0)[use]
    use=(pred!=obs['original_source'])&(obs['original_source']>0)&(obs['class_share']>=g['share'])&(support<=g['spatial_bin'])&(obs['class_retained']<=g['retained'])&(obs['class_fraction']<=g['fraction'])
    return np.where(use,obs['original_source'],pred).astype(np.uint8)

def guard_features(old,current):
    obs=dict(old);original=old['source'];area=old['area'];total=max(float(area[original>0].sum()),1)
    oldmass=np.bincount(original,weights=area,minlength=25);newmass=np.bincount(current,weights=area,minlength=25)
    share=area/np.maximum(oldmass[original],1);newshare=area/np.maximum(newmass[current],1);retained=newmass[original]/np.maximum(oldmass[original],1);fraction=oldmass[original]/total
    se=[0,.01,.03,.1,.25,.5,.75,1];fe=[0,.0001,.0005,.001,.005,.01,.05,1]
    common=[bins(share,se),bins(newshare,se),bins(retained,[0,.05,.1,.25,.5,.75,1,1000]),bins(fraction,fe)]
    features=[]
    for j,dst in enumerate(old['destination']):features.append(np.concatenate([old['features'][j],np.stack([*common,bins(newmass[dst]/total,fe),old['features'][j,:4].max(0)])]))
    obs.update(source=current,original_source=original,features=np.stack(features),class_share=share,class_retained=retained,class_fraction=fraction)
    return obs

def route(obs,bundle,names,original=False):
    f=bundle['fusion'];index=np.asarray(f['choice'])[obs['original_source'] if original else obs['source']];out=obs['source'].copy()
    for i in np.unique(index):
        name=f['names'][i]
        if name=='incumbent':continue
        pred=region_apply(obs,bundle['branches'][name],names);use=index==i;out[use]=pred[use]
    return out

def score(cm):
    z=np.asarray(cm,np.float64)[...,1:,1:];tp=np.diagonal(z,axis1=-2,axis2=-1)
    return 100*np.mean(tp/np.maximum(z.sum(-1)+z.sum(-2)-tp,1),axis=-1)
