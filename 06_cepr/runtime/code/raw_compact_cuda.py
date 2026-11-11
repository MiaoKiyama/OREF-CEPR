"""Original CUDA compactor with target serialization removed; no arithmetic changes."""
import numpy as np
from raw_primitives import packed
ACTION_INDICES=np.array([2,4,6,3,5])
def compact_arrays(fields,start,shape):
 import torch
 with torch.inference_mode():
  t=torch.from_numpy(np.stack(fields)).to('cuda')
  top,lab=t.max(1);labels=(lab+1).to(torch.uint8).cpu().numpy();top1=top.cpu().numpy()
  offsets=np.flatnonzero(np.any(labels[ACTION_INDICES]!=labels[6],axis=0)).astype(np.uint32)
  q=t[:,:,torch.from_numpy(offsets.astype(np.int64)).to('cuda')]
  order=torch.argsort(q,dim=1,descending=True,stable=True)[:,:5]
  ids=(order+1).to(torch.uint8).cpu().numpy();pp=torch.gather(q,1,order).cpu().numpy()
  ee=np.stack([-(v[:,offsets]*np.log(np.maximum(v[:,offsets],np.float32(1e-30)))).sum(0,dtype=np.float32) for v in fields])
  arrays=dict(start=np.array(start,np.int64),count=np.array(labels.shape[1],np.int64),image_shape=np.array(shape,np.int32),
   labels=labels,top1_fp32=packed(top1),detail_offsets=offsets,top5_ids=ids,top5_fp32=packed(pp),entropy_fp32=packed(ee))
  cf=np.concatenate((top1[:,offsets],pp[:,0]-pp[:,1],ee),axis=0).T
  return arrays,labels,offsets,cf
