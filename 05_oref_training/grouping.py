import numpy as np
from raw_primitives import unpacked
from raw_graph import pixel_keys
from features import ORDER

def aligned_chunk(z,start,shape):
 al=z['labels'];pts=np.flatnonzero(np.any(al!=al[6],axis=0));fixed=np.bincount(al[6,np.all(al==al[6],axis=0)],minlength=25)
 if not len(pts):return None,fixed,np.zeros(0,np.int64)
 labs=al[ORDER][:,pts].T;keys=pixel_keys(labs,start+pts,shape)
 keys,first,inv,mass=np.unique(keys,return_index=True,return_inverse=True,return_counts=True);n=len(keys)
 top1=unpacked(z['top1_fp32'],(7,al.shape[1]))[:,pts];cf=np.zeros((n,21),np.float32)
 for j in range(7):cf[:,j]=np.bincount(inv,weights=top1[j],minlength=n)/mass
 weighted=np.zeros((n,7,7),np.float32);known=np.zeros_like(weighted)
 eligible=np.any(labs[:,:5]!=labs[:,2:3],axis=1);detail=eligible[first]
 for j in range(7):
  for a in range(7):
   hit=al[j,pts]==labs[:,a]
   weighted[:,j,a]=np.bincount(inv,weights=top1[j]*hit,minlength=n)/mass
   known[:,j,a]=np.bincount(inv,weights=hit,minlength=n)/mass
 di=z['detail_offsets'].astype(np.int64);assert np.array_equal(pts[eligible],di)
 if len(di):
  ii=inv[eligible];ids=z['top5_ids'];pp=unpacked(z['top5_fp32'],(7,5,len(di)));ee=unpacked(z['entropy_fp32'],(7,len(di)))
  for j in range(7):
   cf[:,7+j]=np.bincount(ii,weights=pp[j,0]-pp[j,1],minlength=n)/mass
   cf[:,14+j]=np.bincount(ii,weights=ee[j],minlength=n)/mass
   for a in range(7):
    hit=ids[j]==labs[eligible,a][None,:]
    vv=np.bincount(ii,weights=(pp[j]*hit).sum(0),minlength=n)/mass
    kk=np.bincount(ii,weights=hit.any(0),minlength=n)/mass
    weighted[detail,j,a]=vv[detail];known[detail,j,a]=kk[detail]
 pp=np.divide(weighted,known,out=np.zeros_like(weighted),where=known>0)[:,ORDER]
 kk=known[:,ORDER];cf=cf.reshape(n,3,7)[:,:,ORDER].reshape(n,21)
 numeric=np.concatenate([cf,pp.reshape(n,49),kk.reshape(n,49),np.zeros((n,7),np.float32)],1)
 return dict(keys=keys,labels=labs[first],numeric=numeric,detail=detail,mass=mass.astype(np.uint32)),fixed,np.unique(keys[detail]//25**7//33)
