"""Build observable groups and object graphs, then attach TRAIN-only histograms.

Requires four probability arrays per scene produced by part 3 and Q sidecars
from part 4. Graph construction requires Linux and g++.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from oref_cepr.data import splits,label_path
from common import sha
from raw_primitives import probability_fields,compact_arrays
from raw_graph import pixel_keys,graph_from_arrays,SCHEMA
from grouping import aligned_chunk
from features import canonical,ORDER
Image.MAX_IMAGE_PIXELS=None

def prepare_scene(folder,q_path,label_path,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    arrays=[np.load(folder/(name+'.npy'),mmap_mode='r',allow_pickle=False)
            for name in ('dinov2','dinov3','mos_off','mos_on')]
    shape=arrays[0].shape[1:];h,w=shape
    if any(a.shape!=(24,h,w) for a in arrays):raise ValueError('Expert geometry differs')
    with np.load(q_path,allow_pickle=False) as z:q=z['q']
    if q.shape!=(33,18):raise ValueError('Expected 33x18 Q sidecar')
    with Image.open(label_path) as im:truth=np.asarray(im)
    if truth.shape!=shape or np.any((truth>24)&(truth!=255)):raise ValueError('Invalid FBP training labels')
    truth=np.where(truth==255,0,truth).astype(np.int64)
    sar=np.load(folder/'sar_blocks.npy',allow_pickle=False)
    available=json.loads((folder/'sar.json').read_text())['available']
    expected=((h+63)//64)*((w+63)//64)
    if sar.shape!=(expected,6):raise ValueError('SAR block geometry differs')
    hard=np.lib.format.open_memmap(output/'hard.tmp.npy',mode='w+',dtype=np.uint8,shape=(7,h,w))
    mapping=np.lib.format.open_memmap(output/'mapping.tmp.npy',mode='w+',dtype=np.int32,shape=(h,w));mapping[:]=-1
    parts=[];counts=[];fixed=np.zeros((24,24),np.int64);cursor=0
    for y in range(0,h,64):
        start=y*w;stop=min(y+64,h)*w
        fields=probability_fields(*(a[:,y:y+64].reshape(24,-1) for a in arrays))
        if any(not np.isfinite(a).all() for a in fields):raise ValueError('Nonfinite probabilities')
        z,labels,_,_=compact_arrays(fields,start,shape)
        part,_,_=aligned_chunk(z,start,shape);hard.reshape(7,-1)[:,start:stop]=labels
        active=np.flatnonzero(np.any(labels!=labels[6],axis=0));gt=truth.ravel()[start:stop]
        agreed=np.all(labels==labels[6],axis=0)&(gt>0)
        fixed+=np.bincount((gt[agreed]-1)*24+labels[6,agreed]-1,minlength=576).reshape(24,24)
        if part is None:continue
        ids=np.searchsorted(part['keys'],pixel_keys(labels[ORDER][:,active].T,start+active,shape))
        mapping.ravel()[start+active]=ids+cursor
        hist=np.zeros((len(part['keys']),24),np.int64)
        known=gt[active]>0
        np.add.at(hist,(ids[known],gt[active[known]]-1),1)
        block=part['keys']//25**7//33
        values=np.c_[sar[block],np.ones(len(block),np.float32)]
        if not available:values[:]=0
        part['numeric'][:,119:126]=values
        roi=((part['keys']//25**7)%33).astype(int)
        numeric=np.concatenate([part['numeric'],q[roi]],axis=1)
        parts.append(dict(raw=canonical(numeric,part['labels']),labels=part['labels'],mass=part['mass']))
        counts.append(hist);cursor+=len(hist)
    if not cursor:raise ValueError('No disagreement groups to train on')
    for key in ('raw','labels','mass'):np.save(output/(key+'.npy'),np.concatenate([p[key] for p in parts]))
    np.save(output/'counts.npy',np.concatenate(counts));np.save(output/'fixed.npy',fixed)
    graph=graph_from_arrays(hard,mapping,cursor,dict(graph_store=str(output)),output/'graph.work.bin')
    np.savez_compressed(output/'graph.npz',**graph)
    (output/'completed.json').write_text(json.dumps(dict(status='sealed',schema=SCHEMA,
        ground_truth_read=False,graph_sha256=sha(output/'graph.npz'),audit=dict(pixels=h*w,groups=cursor))))
    # The graph ignores truth; only counts/fixed arrays above are supervised.
    hard._mmap.close();mapping._mmap.close()
    (output/'hard.tmp.npy').unlink();(output/'mapping.tmp.npy').unlink()
    return dict(scene=folder.name,path=str(output.resolve()),groups=cursor)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',required=True,type=Path);p.add_argument('--q',required=True,type=Path)
    p.add_argument('--labels',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--source',required=True,choices=['oof120','replay120'])
    a=p.parse_args();rows=[];allowed=set(splits()['train_ids'])
    for folder in sorted(a.predictions.iterdir()):
        if not folder.is_dir():continue
        if folder.name not in allowed:raise ValueError('OREF training accepts official training scenes only')
        identities={json.loads((folder/(model+'_receipt.json')).read_text())['image_sha256']
                    for model in ('dinov2','dinov3','mos')}
        with np.load(a.q/(folder.name+'.npz'),allow_pickle=False) as z:q_identity=str(z['image_sha256'])
        if identities!={q_identity}:raise ValueError('Q and expert predictions refer to different raw images')
        if a.source=='oof120':
            expected=next(f['fold_id'] for f in splits()['folds'] if folder.name in f['held_ids'])
            for model in ('dinov2','dinov3','mos'):
                receipt=json.loads((folder/(model+'_receipt.json')).read_text())
                if receipt['fold']!=expected:raise ValueError('OOF prediction fold mismatch')
        dest=a.output/folder.name
        if dest.exists():raise FileExistsError('Use a fresh group output directory')
        rows.append(prepare_scene(folder,a.q/(folder.name+'.npz'),label_path(a.labels,folder.name),dest))
        print(folder.name,flush=True)
    if not rows:raise ValueError('No prediction scenes')
    (a.output/'manifest.json').write_text(json.dumps(dict(source=a.source,scenes=rows),indent=2))

if __name__=='__main__':main()
