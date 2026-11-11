"""Read-only 313-feature adapter; deployment geometry never reads label arrays."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
import argparse, concurrent.futures, gc, hashlib, json, time
from pathlib import Path
import numpy as np

ORDER=np.array([2,4,6,3,5,0,1]); INVERSE=np.argsort(ORDER)
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,j):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    q.write_text(json.dumps(j,indent=2,allow_nan=False),encoding='utf-8');os.replace(q,p)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for v in iter(lambda:f.read(8<<20),b''):h.update(v)
    return h.hexdigest()
def save(p,**a):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with q.open('wb') as f:np.savez_compressed(f,**a)
    os.replace(q,p)
def key(labs,pos,shape):
    h,w=shape;y=pos//w;x=pos%w
    roi=np.searchsorted(np.linspace(0,h,5,dtype=int)[1:],y,side='right')*8+np.searchsorted(np.linspace(0,w,9,dtype=int)[1:],x,side='right')
    return ((y//64)*((w+63)//64)+x//64)*33*25**7+roi*25**7+labs.astype(np.int64)@(25**np.arange(6,-1,-1))

def prepare_observable(args):
    config,domain,row=args;c=read(config);sid=row['scene'];out=Path(c['output'])/'observable'/domain/(sid+'.npz')
    if out.with_suffix('.json').exists():return read(out.with_suffix('.json'))
    folder=Path(row['source']);m=read(folder/'compact_manifest.json')
    if domain=='oof':
        with np.load(folder/'groups.npz') as z:expected=z['keys']
    else:
        with np.load(Path(c['official_aligned'])/'scenes'/(sid+'.npz')) as z:expected=z['keys']
    mass=np.zeros(len(expected),np.uint32);fixed=np.zeros(25,np.int64);cursor=0
    for ch in m['chunks']:
        with np.load(folder/'compact'/ch['file']) as z:al=z['labels']
        active=np.any(al!=al[6],axis=0);pts=np.flatnonzero(active)
        fixed+=np.bincount(al[6,~active],minlength=25)
        if len(pts):
            keys,counts=np.unique(key(al[ORDER][:,pts].T,ch['start']+pts,m['shape']),return_counts=True)
            assert np.array_equal(keys,expected[cursor:cursor+len(keys)]),(sid,cursor)
            mass[cursor:cursor+len(keys)]=counts;cursor+=len(keys)
    assert cursor==len(mass) and int(mass.sum())+int(fixed.sum())==int(np.prod(m['shape']))
    save(out,mass=mass,fixed_prediction_counts=fixed)
    receipt=dict(scene=sid,domain=domain,groups=cursor,shape=m['shape'],pixels=int(np.prod(m['shape'])),
        manifest_sha256=sha(folder/'compact_manifest.json'),observable_sha256=sha(out),ground_truth_members_read=False,time_unix=time.time())
    write(out.with_suffix('.json'),receipt);return receipt

def canonical(numeric,labels):
    """Inverse of the later adapter; restore original raw-expert probability order."""
    n=len(labels);x=np.zeros((n,313),np.float32)
    x[:,:168]=np.eye(24,dtype=np.float32)[labels-1].reshape(n,168)
    cf=numeric[:,:21].reshape(n,3,7)[:,:,INVERSE].reshape(n,21).copy()
    known=numeric[:,70:119].reshape(n,7,7)[:,INVERSE,:].copy()
    pp=(numeric[:,21:70].reshape(n,7,7)*numeric[:,70:119].reshape(n,7,7))[:,INVERSE,:].copy()
    detail=np.any(labels[:,:5]!=labels[:,2:3],axis=1)
    # Original bank has only Top1 evidence outside its first-five disagreement domain.
    missing=~detail
    if missing.any():
        cf[missing,7:]=0
        raw_labels=labels[missing][:,INVERSE]
        hit=raw_labels[:,:,None]==labels[missing,None,:]
        known[missing]=hit
        pp[missing]=cf[missing,:7,None]*hit
    x[:,168:189]=cf;x[:,189:207]=numeric[:,126:144]
    x[:,207:214]=numeric[:,119:126];x[:,214:263]=pp.reshape(n,49)
    x[:,263:312]=known.reshape(n,49);x[:,312]=detail
    assert np.isfinite(x).all();return x

class Bank:
    def __init__(self,c,ids,domain='oof',targets=True):
        if domain != 'oof' or not set(ids) <= set(c['train_ids']):
            raise ValueError('This training bank accepts train120 OOF inputs only')
        self.c=c;self.ids=list(ids);self.domain=domain;self.rows=[];self.targets=targets
        root=Path(c['store']);index=read(root/'index.json');mapping={r['scene']:r for r in index['rows']}
        self.numeric=np.load(root/'numeric.npy',mmap_mode='r') if domain=='oof' else None
        self.labels=np.load(root/'labels.npy',mmap_mode='r') if domain=='oof' else None
        self.counts=np.load(root/'counts.npy',mmap_mode='r') if domain=='oof' and targets else None
        cursor=0
        for sid in ids:
            obs=read(Path(c['output'])/'observable'/domain/(sid+'.json'))
            row=dict(mapping[sid]) if domain=='oof' else dict(scene=sid)
            row.update(local_start=cursor,local_end=cursor+obs['groups'],pixels=obs['pixels'])
            self.rows.append(row);cursor+=obs['groups']
        self.n=cursor;self.pixels=sum(r['pixels'] for r in self.rows)
    def scene(self,row,targets=None):
        targets=self.targets if targets is None else targets;sid=row['scene'];c=self.c
        with np.load(Path(c['output'])/'observable'/self.domain/(sid+'.npz')) as z:
            d={'mass':z['mass'],'fixed_prediction_counts':z['fixed_prediction_counts']}
        if self.domain=='oof':
            sl=slice(row['start'],row['end']);d.update(numeric=np.asarray(self.numeric[sl]),labels=np.asarray(self.labels[sl]))
            if targets:
                assert self.counts is not None
                meta=read(Path(c['store'])/'scenes'/sid/'completed.json')
                d.update(counts=np.asarray(self.counts[sl]),fixed=np.asarray(meta['fixed'],np.int64),baselines=np.asarray(meta['baselines'],np.int64))
        if targets:assert np.array_equal(d['counts'].sum(1,dtype=np.int64),d['mass'])
        return d
    def target_pack(self):
        assert self.targets
        counts=np.empty((self.n,25),np.uint16);labs=np.empty((self.n,7),np.uint8);mass=np.empty(self.n,np.uint32)
        fixed=np.zeros((25,25),np.int64);fixed_obs=np.zeros(25,np.int64);bases=np.zeros((7,25,25),np.int64)
        for r in self.rows:
            d=self.scene(r);sl=slice(r['local_start'],r['local_end'])
            counts[sl]=d['counts'];labs[sl]=d['labels'];mass[sl]=d['mass'];fixed+=d['fixed'];fixed_obs+=d['fixed_prediction_counts'];bases+=d['baselines']
        return dict(counts=counts,labels=labs,mass=mass,fixed=fixed,fixed_obs=fixed_obs,baselines=bases,pixels=self.pixels)

def main():
    a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--workers',type=int,default=4);v=a.parse_args();c=read(v.config)
    jobs=[(v.config,dom,row) for dom,rows in c['scene_inventory'].items() for row in rows]
    done=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=v.workers) as pool:
        for result in pool.map(prepare_observable,jobs):
            done.append(result);write(Path(c['output'])/'observable_progress.json',dict(completed=len(done),total=len(jobs),time_unix=time.time()))
    write(Path(c['output'])/'observable_completed.json',dict(status='completed',scenes=len(done),ground_truth_members_read=False,receipts=done))
if __name__=='__main__':main()
