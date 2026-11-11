"""Exact frozen C/E execution: share identical contexts and score selected classes only."""
from common import *
import gc
import evidence as e
from group_cases import Posterior,raw_route

def enable_parallel_kernels(threads=8):
 """Parallelize independent rows/classes without changing any reduction order."""
 def expected(hist,labels,current):
  out=np.empty((24,24),np.float64);labels=np.ascontiguousarray(labels);current=np.ascontiguousarray(current)
  e.LIB.expected_cpu(hist.ctypes.data,labels.ctypes.data,current.ctypes.data,len(hist),out.ctypes.data,threads)
  return out
 def gains(cm,hist,labels,current):
  mass=hist.sum(1);d=cm.diagonal();s=np.concatenate([d,cm.sum(0)+cm.sum(1)-d]);out=np.empty(labels.shape,np.float64)
  e.LIB.gains_cpu(s.ctypes.data,hist.ctypes.data,mass.ctypes.data,labels.ctypes.data,current.ctypes.data,len(hist),out.ctypes.data,threads)
  return out
 e.expected_cm=expected;e.gains=gains

def calibrated(cfg,pack,p1,p2):
 l=pack['labels'];n=len(l);rr=np.arange(n);base=l[:,2];parent=np.full(n,2,np.uint8);actions=parent.copy();best=np.zeros(n,np.float64)
 unique={}
 for cls,desc in cfg['parents'].items():unique.setdefault(json.dumps(desc,sort_keys=True),[]).append(int(cls))
 contexts={}
 for key,classes in unique.items():
  desc=json.loads(key);ix=np.flatnonzero(np.isin(base,classes));method=desc['method'];weight=1.0 if desc['networks']==['oof120_graphQS_CE_8000'] else 0.0
  if method=='baseline' or not len(ix):continue
  if method=='raw':
   p=Posterior(p1,p2,weight);a=raw_route(p,l);parent[ix]=actions[ix]=a[ix]
   v=p[ix];best[ix]=v[np.arange(len(ix)),l[ix,a[ix]]-1].astype(np.float64)-v[np.arange(len(ix)),base[ix]-1].astype(np.float64);continue
  q=cfg['parent_policies'][desc['policy']]
  if method=='direction':dp=q;cp=q['fallback_C'];hp=dict(transform=dict(temperature=1,strength=0),ratio=[1.]*24,C=cp)
  elif method=='C':dp=None;cp=q['policy']['C'];hp=dict(transform=dict(temperature=1,strength=0),ratio=[1.]*24,C={'prior':q['policy']['prior']})
  else:
   dp=q['direction'];cp=dp['fallback_C'];hp=dict(transform=q['transform'],ratio=q['ratio'],C=cp)
   if len(desc['networks'])==2:weight=q['transform']['oof_weight']
  # history() consumes only these fields; acceptance thresholds do not enter it.
  identity=json.dumps(dict(weight=weight,transform=hp['transform'],ratio=hp['ratio'],prior=hp['C']['prior']),sort_keys=True)
  contexts.setdefault(identity,dict(policy=hp,weight=weight,jobs=[]))['jobs'].append((ix,cp,dp))
 for k,item in enumerate(contexts.values()):
  progress('frozen C shared context',context=k+1,total=len(contexts))
  hist,fixed=e.history(Posterior(p1,p2,item['weight']),pack,item['policy'])
  initial=np.full(n,2,np.uint8);cm=fixed+e.expected_cm(hist,l,initial)
  for ix,cp,dp in item['jobs']:
   local=l[ix];g=e.gains(cm,np.ascontiguousarray(hist[ix]),local,np.full(len(ix),2,np.uint8))
   a=e.apply_c(g,local,cp) if dp is None else e.apply_d(g,local,dp)
   parent[ix]=a;actions[ix]=g.argmax(1).astype(np.uint8);best[ix]=g.max(1)
   del g,local
  del hist,cm;gc.collect()
 out=parent.copy();code=base.astype(np.int64)*25+l[rr,actions]
 for key,t in cfg['joint_policy']['direction_thresholds'].items():
  ix=np.flatnonzero(code==int(key));out[ix]=2
  if t!='never_switch':sel=ix[best[ix]>t];out[sel]=actions[sel]
 return l[rr,out]

def proposal(p,pack,parent,policy,novel,ix):
 hist,fixed=e.history(p,pack,policy);n=len(parent);initial=np.full(n,2,np.uint8);current=np.repeat(parent[:,None],7,axis=1)
 expected=fixed+e.expected_cm(hist,current,initial)
 hs=np.ascontiguousarray(hist[ix]);current=np.ascontiguousarray(current[ix]);del hist;gc.collect()
 labs_original=pack['labels'][ix];dest=parent[ix].copy();best=np.zeros(len(ix),np.float64);rr=np.arange(len(ix));initial=np.full(len(ix),2,np.uint8)
 for lo in range(1,25,6):
  labs=current.copy();slots=[0,1,3,4,5,6]
  for slot,cls in zip(slots,range(lo,lo+6)):labs[:,slot]=cls
  g=e.gains(expected,hs,labs,initial)
  if novel:
   for slot,cls in zip(slots,range(lo,lo+6)):g[(labs_original==cls).any(1),slot]=0
  action=g.argmax(1);value=g[rr,action];use=value>best;dest[use]=labs[rr[use],action[use]];best[use]=value[use]
  del g,labs
 del hs,current;gc.collect();labs=np.repeat(parent[ix,None],7,axis=1);labs[:,0]=dest;g=np.zeros((len(ix),7),np.float64);g[:,0]=best
 route=e.apply_d(g,labs,policy['direction']);result=labs[rr,route]
 if novel:assert not (((labs_original==result[:,None]).any(1))&(result!=parent[ix])).any()
 return result

def expansion(cfg,pack,p,base):
 l=pack['labels'];expanded=base.copy()
 for name,policy in cfg['open_policies'].items():
  choice=np.asarray(cfg['open_choice']['choices']);names=cfg['open_choice']['names'];ix=np.flatnonzero(np.isin(choice[l[:,2]],[i for i,x in enumerate(names) if x==name+'_D']))
  if len(ix):progress('frozen E selected classes',policy=name,groups=len(ix));expanded[ix]=proposal(p,pack,l[:,2],policy,False,ix)
 final=expanded.copy()
 for name,policy in cfg['novel_policies'].items():
  choice=np.asarray(cfg['novel_choice']['choices']);names=cfg['novel_choice']['names'];ix=np.flatnonzero(np.isin(choice[l[:,2]],[i for i,x in enumerate(names) if x==name+'_D']))
  if len(ix):progress('frozen E residual selected classes',policy=name,groups=len(ix));final[ix]=proposal(p,pack,expanded,policy,True,ix)
 return expanded,final
