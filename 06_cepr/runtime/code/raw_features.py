"""Label-free feature extraction, preserving the frozen seven-expert arithmetic."""
import gc,hashlib,time
from pathlib import Path
import numpy as np
from common import read,write,sha
from raw_primitives import probability_fields,compact_arrays,packed,unpacked,save_npz
from features import ORDER
from vendor import region
from raw_graph import pixel_keys,graph_from_arrays,SCHEMA

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

def compact(job,fields):
 import torch
 from raw_compact_cuda import compact_arrays
 torch.set_num_threads(2)
 folder=Path(job['output']);dest=folder/'compact';dest.mkdir(exist_ok=True);shape=job['shape'];h,w=shape
 flat=[f.reshape(24,-1) for f in fields];chunks=[];parts=[];fixed=np.zeros(25,np.int64)
 observed_sar_blocks=np.zeros(((h+63)//64)*((w+63)//64),bool);hashes=[hashlib.sha256() for _ in range(7)];beg=time.time()
 for ci,start in enumerate(range(0,h*w,w*64)):
  stop=min(start+w*64,h*w);fs=probability_fields(*(v[:,start:stop] for v in flat))
  z,labels,offsets,cf=compact_arrays(fs,start,shape)
  for digest,l in zip(hashes,labels):digest.update(l.tobytes())
  part,fx,blocks=aligned_chunk(z,start,shape);fixed+=fx;observed_sar_blocks[blocks]=True
  if part is not None:parts.append(part)
  prob=unpacked(z['top1_fp32'],(7,stop-start));x,features,raw=region.pixel_features(prob)
  # Exact sufficient observations for P/R; no class posterior or target is retained.
  reduced=dict(start=z['start'],count=z['count'],image_shape=z['image_shape'],labels=labels,
   region_x=x,region_features=features,region_raw=packed(raw))
  name='chunk_%04d.npz'%ci;p=dest/name
  if p.exists():
   # Resume after interruption only when every rebuilt value is identical.
   with np.load(p,allow_pickle=False) as existing:
    assert set(existing.files)==set(reduced),('Partial chunk schema changed',p)
    for key,value in reduced.items():np.testing.assert_array_equal(existing[key],value,err_msg=str(p)+'::'+key)
   receipt={'sha256':sha(p)}
  else:receipt=save_npz(p,reduced)
  chunks.append(dict(path=str(p),start=start,count=stop-start,sha256=receipt['sha256']))
  write(folder/'progress.json',dict(stage='compact_and_features',chunk=ci+1,total_chunks=(h+63)//64,seconds=time.time()-beg,time_unix=time.time()))
 arr={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};del parts
 sar=np.load(folder/'sar_blocks.npy');assert sar.shape==(len(observed_sar_blocks),6)
 mask=observed_sar_blocks&bool(job['has_sar']);values=np.c_[sar,np.ones(len(sar),np.float32)];values[~mask]=0
 arr['numeric'][:,119:126]=values[arr['keys']//25**7//33]
 mass=arr.pop('mass');assert mass.sum(dtype=np.uint64)+fixed.sum()==h*w and np.all(np.diff(arr['keys'])>0)
 save_npz(folder/'features.npz',arr);save_npz(folder/'observable.npz',dict(mass=mass,fixed_prediction_counts=fixed))
 result=dict(scene=job['scene'],shape=shape,groups=len(mass),keys=str(folder/'features.npz'),observable=str(folder/'observable.npz'),chunks=chunks,
  graph=str(folder/'graph'),q='q_sidecars/'+job['scene']+'.npz',candidate_prediction_sha256=[v.hexdigest() for v in hashes],cache_bytes=sum(Path(c['path']).stat().st_size for c in chunks),truth_read=False)
 write(folder/'compact_completed.json',result);return result

def graph(row):
 root=Path(row['graph']);root.mkdir(exist_ok=True);h,w=row['shape']
 with np.load(row['keys']) as z:keys=z['keys'];labs=z['labels']
 with np.load(row['observable']) as z:mass=z['mass']
 n=len(keys);hard=np.empty((7,h*w),np.uint8);mapping=np.full(h*w,-1,np.int32);observed=np.zeros(n,np.uint64);digest=hashlib.sha256()
 for ch in row['chunks']:
  with np.load(ch['path']) as z:labels=z['labels'];start=int(z['start'])
  stop=start+labels.shape[1];hard[:,start:stop]=labels
  active=np.flatnonzero(np.any(labels!=labels[6],axis=0));actions=labels[ORDER][:,active].T
  k=pixel_keys(actions,start+active,row['shape']);ids=np.searchsorted(keys,k)
  np.testing.assert_array_equal(keys[ids],k);np.testing.assert_array_equal(labs[ids],actions)
  mapping[start+active]=ids;observed+=np.bincount(ids,minlength=n).astype(np.uint64)
  digest.update(np.asarray([start],dtype='<i8').tobytes());digest.update(labels.tobytes())
 np.testing.assert_array_equal(observed,mass)
 data=graph_from_arrays(hard.reshape(7,h,w),mapping.reshape(h,w),n,{'graph_store':str(root.parent.parent/'graph_build')},root/'working.bin')
 del hard,mapping;gc.collect();save_npz(root/'graph.npz',data)
 write(root/'completed.json',dict(status='sealed',schema=SCHEMA,ground_truth_read=False,graph_sha256=sha(root/'graph.npz'),
  audit=dict(pixels=h*w,groups=n,observed_group_pixels=int(observed.sum()),labels_and_group_mass_exact=True,input_sha256=digest.hexdigest())))
 return row
