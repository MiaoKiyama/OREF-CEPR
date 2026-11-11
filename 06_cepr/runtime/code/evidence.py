from common import *
import ctypes,gc,resource

LIB=ctypes.CDLL(str(PACKAGE/'code/gains.so'));ptr=ctypes.c_void_p
LIB.gains_cpu.argtypes=[ptr,ptr,ptr,ptr,ptr,ctypes.c_int64,ptr,ctypes.c_int]
LIB.expected_cpu.argtypes=[ptr,ptr,ptr,ctypes.c_int64,ptr,ctypes.c_int]
def expected_cm(hist,labels,current):
 out=np.empty((24,24),np.float64);labels=np.ascontiguousarray(labels);current=np.ascontiguousarray(current)
 LIB.expected_cpu(hist.ctypes.data,labels.ctypes.data,current.ctypes.data,len(hist),out.ctypes.data,2);return out
def gains(cm,hist,labels,current):
 mass=hist.sum(1);d=cm.diagonal();s=np.concatenate([d,cm.sum(0)+cm.sum(1)-d]);out=np.empty(labels.shape,np.float64)
 LIB.gains_cpu(s.ctypes.data,hist.ctypes.data,mass.ctypes.data,labels.ctypes.data,current.ctypes.data,len(hist),out.ctypes.data,2);return out
def history(p,pack,policy):
 item=policy['transform'];T=item['temperature'];L=item['strength'];ratio=np.asarray(policy['ratio']);prior=policy['C']['prior']
 hist=np.empty(p.shape,np.float64);scale=prior['reference_pixels']/pack['pixels'];rates=np.asarray(prior['valid_rate'])
 for lo in range(0,len(p),65536):
  hi=min(lo+65536,len(p));q=p[lo:hi].astype(np.float64)
  if T!=1 or L!=0:
   z=np.log(np.maximum(q,1e-30))/T+L*np.log(ratio)[None];z-=z.max(1)[:,None];q=np.exp(z);q/=q.sum(1)[:,None]
  mass=pack['mass'][lo:hi].astype(np.float64)*rates[pack['labels'][lo:hi,2]]*scale;hist[lo:hi]=q*mass[:,None]
 fixed=np.asarray(prior['agreement_conditional'])[:,1:]*pack['fixed_obs'][None,1:]*scale
 return hist,fixed
def apply_c(g,l,p):
 out=np.full(len(l),2,np.uint8);a=g.argmax(1).astype(np.uint8);v=g.max(1)
 for cls in range(1,25):
  t=p['source_class_thresholds'][str(cls)]
  if t=='keep_initial_global_policy':t=p['global_threshold']
  if t is None or t=='never_switch':continue
  ix=(l[:,2]==cls)&(v>float(t));out[ix]=a[ix]
 return out
def apply_d(g,l,p):
 out=apply_c(g,l,p['fallback_C']);a=g.argmax(1).astype(np.uint8);v=g.max(1);codes=l[:,2].astype(np.int64)*25+l[np.arange(len(l)),a]
 for key,t in p['direction_thresholds'].items():
  ix=np.flatnonzero(codes==int(key));out[ix]=2
  if t!='never_switch':keep=ix[v[ix]>t];out[keep]=a[keep]
 return out
def proposal(p,pack,parent,policy,novel):
 hist,fixed=history(p,pack,policy);n=len(parent);initial=np.full(n,2,np.uint8);current=np.repeat(parent[:,None],7,axis=1)
 expected=fixed+expected_cm(hist,current,initial);dest=parent.copy();best=np.zeros(n,np.float64);rr=np.arange(n)
 for lo in range(1,25,6):
  labs=current.copy();slots=[0,1,3,4,5,6]
  for slot,cls in zip(slots,range(lo,lo+6)):labs[:,slot]=cls
  g=gains(expected,hist,labs,initial)
  if novel:
   for slot,cls in zip(slots,range(lo,lo+6)):g[(pack['labels']==cls).any(1),slot]=0
  action=g.argmax(1);value=g[rr,action];use=value>best;dest[use]=labs[rr[use],action[use]];best[use]=value[use]
  del g,labs;gc.collect()
 del hist,current;gc.collect();labs=np.repeat(parent[:,None],7,axis=1);labs[:,0]=dest;g=np.zeros((n,7),np.float64);g[:,0]=best
 route=apply_d(g,labs,policy['direction']);result=labs[rr,route];del labs,g
 if novel:assert not (((pack['labels']==result[:,None]).any(1))&(result!=parent)).any()
 return result