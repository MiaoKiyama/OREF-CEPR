import os,uuid,zipfile
from pathlib import Path
import numpy as np
from common import sha
ACTION_INDICES=np.array([2,4,6,3,5])
def require(v,msg):
 if not v:raise ValueError(msg)
def packed(a):
    a=np.ascontiguousarray(a,dtype=np.float32)
    return a.view(np.uint8).reshape(-1,4).T.copy()

def unpacked(a,shape):
    return np.ascontiguousarray(a.T).view(np.float32).reshape(shape)

def save_npz(path,arrays):
    path=Path(path);require(not path.exists(),'Refuse overwrite '+str(path))
    tmp=path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
    with zipfile.ZipFile(tmp,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for key,a in arrays.items():
            with z.open(key+'.npy','w',force_zip64=True) as f:
                a=np.asarray(a)
                np.lib.format.write_array(f,a if a.ndim==0 else np.ascontiguousarray(a),allow_pickle=False)
    # Verify lossless content, not merely successful serialization.
    with np.load(tmp,allow_pickle=False) as z:
        require(set(z.files)==set(arrays),'NPZ inventory mismatch')
        for k,a in arrays.items():require(np.array_equal(z[k],a),'NPZ readback mismatch '+k)
    os.replace(tmp,path)
    return dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path))

def probability_fields(a,b,off,on):
    po=off;pn=on
    require(a.shape[0]==b.shape[0]==off.shape[0]==on.shape[0]==24,"Native24 probabilities required")
    dual=a*np.float32(.5)+b*np.float32(.5)
    def triple(p):
        x=p*np.float32(1/3);x+=a*np.float32(1/3);x+=b*np.float32(1/3);return x
    return (a,b,dual,po,triple(po),pn,triple(pn))

def compact_arrays(fields,start,shape):
    labels=np.stack([(v.argmax(0)+1).astype(np.uint8) for v in fields])
    top1=np.stack([v.max(0) for v in fields])
    action_labels=labels[ACTION_INDICES]
    offsets=np.flatnonzero(np.any(action_labels!=action_labels[2],axis=0)).astype(np.uint32)
    ids=[];probs=[];entropy=[]
    for v in fields:
        q=v[:,offsets]
        order=np.argsort(-q,axis=0,kind='stable')[:5]
        ids.append((order+1).astype(np.uint8));probs.append(np.take_along_axis(q,order,axis=0))
        entropy.append(-(q*np.log(np.maximum(q,np.float32(1e-30)))).sum(0,dtype=np.float32))
    pp=np.stack(probs);ee=np.stack(entropy)
    arrays=dict(start=np.array(start,np.int64),count=np.array(labels.shape[1],np.int64),
                image_shape=np.array(shape,np.int32),labels=labels,
                top1_fp32=packed(top1),detail_offsets=offsets,top5_ids=np.stack(ids),
                top5_fp32=packed(pp),entropy_fp32=packed(ee))
    # Means in each (64x64 block, Q ROI, five candidate labels) preserve exact GT histograms.
    cf=np.concatenate((top1[:,offsets],pp[:,0]-pp[:,1],ee),axis=0).T
    return arrays,labels,offsets,cf
