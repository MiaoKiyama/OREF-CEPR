from common import *
import evidence as e
import gc,resource
def pack_inputs(cfg):
 rows=cfg['scenes'];n=rows[-1]['end'];labels=np.empty((n,7),np.uint8);mass=np.empty(n,np.uint32);fixed=np.zeros(25,np.int64);pixels=0
 for row in rows:
  sl=slice(row['start'],row['end']);labels[sl]=load(row['keys'],'labels')['labels'];z=load(row['observable'],'mass','fixed_prediction_counts');mass[sl]=z['mass'];fixed+=z['fixed_prediction_counts'];pixels+=int(np.prod(row['shape']))
 assert mass.sum(dtype=np.uint64)+fixed.sum()==pixels
 return dict(labels=labels,mass=mass,fixed_obs=fixed,pixels=pixels)
def raw_route(p,labels):
 out=np.empty(len(labels),np.uint8)
 for lo in range(0,len(labels),65536):
  sl=slice(lo,min(lo+65536,len(labels)));l=labels[sl];v=p[sl];out[sl]=v[np.arange(len(l))[:,None],l-1].argmax(1).astype(np.uint8)
 return out
class Posterior:
 def __init__(self,p1,p2,weight):self.p1,self.p2,self.weight=p1,p2,weight;self.shape=(p1 if p1 is not None else p2).shape
 def __len__(self):return self.shape[0]
 def __getitem__(self,sl):
  if self.p1 is None:return self.p2[sl]
  if self.p2 is None:return self.p1[sl]
  if self.weight==1:return self.p1[sl]
  if self.weight==0:return self.p2[sl]
  # Match the historical posterior mixture's arithmetic (float64 mixture).
  return self.p1[sl].astype(np.float64)*self.weight+self.p2[sl].astype(np.float64)*(1-self.weight)
def calibrated(cfg,pack,p1,p2):
 l=pack['labels'];n=len(l);rr=np.arange(n);base=l[:,2];parent=np.full(n,2,np.uint8);actions=parent.copy();best=np.zeros(n,np.float64)
 unique={}
 for cls,desc in cfg['parents'].items():unique.setdefault(json.dumps(desc,sort_keys=True),[]).append(int(cls))
 for k,(key,classes) in enumerate(unique.items()):
  desc=json.loads(key);ix=np.flatnonzero(np.isin(base,classes));method=desc['method'];weight=1.0 if desc['networks']==['oof120_graphQS_CE_8000'] else 0.0
  progress('generating frozen C decisions',policy=k,total=len(unique),classes=classes)
  if method=='baseline' or not len(ix):continue
  if method=='raw':
   p=Posterior(p1,p2,weight);a=raw_route(p,l);parent[ix]=actions[ix]=a[ix]
   v=p[ix];best[ix]=v[np.arange(len(ix)),l[ix,a[ix]]-1].astype(np.float64)-v[np.arange(len(ix)),base[ix]-1].astype(np.float64);continue
  q=cfg['parent_policies'][desc['policy']]
  if method=='direction':dp=q;cp=q['fallback_C'];histpolicy=dict(transform=dict(temperature=1,strength=0),ratio=np.ones(24),C=cp)
  elif method=='C':dp=None;cp=q['policy']['C'];histpolicy=dict(transform=dict(temperature=1,strength=0),ratio=np.ones(24),C={'prior':q['policy']['prior']})
  else:
   dp=q['direction'];cp=dp['fallback_C'];histpolicy=dict(transform=q['transform'],ratio=q['ratio'],C=cp)
   if len(desc['networks'])==2:weight=q['transform']['oof_weight']
  hist,fixed=e.history(Posterior(p1,p2,weight),pack,histpolicy);initial=np.full(n,2,np.uint8)
  g=e.gains(fixed+e.expected_cm(hist,l,initial),hist,l,initial)
  a=e.apply_c(g,l,cp) if dp is None else e.apply_d(g,l,dp)
  parent[ix]=a[ix];actions[ix]=g[ix].argmax(1).astype(np.uint8);best[ix]=g[ix].max(1)
  del hist,g,a;gc.collect()
 out=parent.copy();code=base.astype(np.int64)*25+l[rr,actions]
 for key,t in cfg['joint_policy']['direction_thresholds'].items():
  ix=np.flatnonzero(code==int(key));out[ix]=2
  if t!='never_switch':sel=ix[best[ix]>t];out[sel]=actions[sel]
 return l[rr,out]