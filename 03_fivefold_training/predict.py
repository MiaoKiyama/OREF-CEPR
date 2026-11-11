"""Sliding-window expert probabilities for OOF or full-training replay.

One checkpoint is loaded per process. Outputs are float32 CxHxW .npy files.
Use each fold checkpoint only on that fold's held IDs for OOF features.
"""
import argparse
from contextlib import nullcontext
import json
import hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from oref_cepr.bootstrap import ROOT
from oref_cepr.data import read_image,normalize,splits
import numpy as np
import torch

def starts(n):
    if n<512:raise ValueError('Scene must be at least 512 pixels per axis')
    return sorted(set(list(range(0,n-511,256))+[n-512]))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=['dinov2','dinov3','mos'])
    p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--pretrained-dir',type=Path,default=Path('pretrained'))
    p.add_argument('--data-root',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    scope=p.add_mutually_exclusive_group(required=True)
    scope.add_argument('--fold',type=int,choices=range(5));scope.add_argument('--replay',action='store_true')
    p.add_argument('--device',default='cuda');p.add_argument('--scene-limit',type=int)
    a=p.parse_args();s=splits();ids=s['folds'][a.fold]['held_ids'] if a.fold is not None else s['train_ids']
    if a.scene_limit:ids=ids[:a.scene_limit]
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    from oref_cepr.inference_models import build_model
    model,_=build_model(a.model,a.pretrained_dir,initialize=False)
    state=payload.get('model',payload.get('state_dict'))
    if state is None:raise ValueError('Expected model/state_dict checkpoint')
    model.load_state_dict(state,strict=True);model.to(a.device)
    if a.fold is not None:
        signature=payload.get('signature')
        if signature is not None and (signature['fold']!=a.fold or set(signature['train_ids'])!=set(s['folds'][a.fold]['train_ids'])):
            raise ValueError('Checkpoint training fold overlaps requested OOF scenes')
        if signature is None:raise ValueError('OOF requires a checkpoint with a fold training signature')
    del payload
    model.eval().requires_grad_(False)
    if a.model=='mos':model.set_stage('joint').eval()
    a.output.mkdir(parents=True,exist_ok=True)
    window=np.maximum(np.outer(np.hanning(512),np.hanning(512)),1e-3).astype(np.float32)
    for sid in ids:
        image=read_image(a.data_root/'images'/(sid+'.tif'));h,w=image.shape[:2]
        folder=a.output/sid;folder.mkdir(parents=True,exist_ok=True)
        modes=['mos_off','mos_on'] if a.model=='mos' else [a.model]
        if any((folder/(mode+'.npy')).exists() for mode in modes):raise FileExistsError('Use a fresh output directory')
        arrays={mode:np.lib.format.open_memmap(folder/(mode+'.partial.npy'),mode='w+',dtype=np.float32,shape=(24,h,w)) for mode in modes}
        for array in arrays.values():array[:]=0
        denominator=np.zeros((h,w),np.float32)
        reader=None;statistics=[];coords=[]
        if a.model=='mos':
            from tcmmsline3.light_sar_data import open_sar_scene,_checked_aux
            reader=open_sar_scene(str(a.data_root.resolve()),sid)
        with torch.inference_mode():
            for y in starts(h):
                for x in starts(w):
                    ms=torch.from_numpy(normalize(image[y:y+512,x:x+512]))[None].to(a.device)
                    context=torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False) if str(a.device).startswith('cuda') else nullcontext()
                    with context:
                        if a.model!='mos':outputs={a.model:model(ms=ms)['logits']}
                        else:
                            if reader.available['sar']:
                                db,valid,optical=_checked_aux(reader.read_patch(y,x,512,modality='sar',out_size=166),512,166)
                            else:
                                db,valid,optical=np.zeros((2,166,166),np.float32),np.zeros((1,166,166),bool),np.zeros((1,512,512),bool)
                            sar={k:torch.from_numpy(np.ascontiguousarray(v))[None].to(a.device) for k,v in dict(sar_db=db,sar_valid=valid,optical_valid=optical).items()}
                            outputs=dict(mos_off=model.forward_optical(ms=ms)['logits'],
                                         mos_on=model(ms=ms,**sar,sar_mode='real')['logits'])
                            st=np.zeros(6,np.float32);st[0]=valid.mean()
                            if valid.any():
                                values=db[:,valid[0]];st[1:3]=values.mean(1);st[3:5]=values.std(1);st[5]=(values[0]-values[1]).mean()
                            statistics.append(st);coords.append((y+256,x+256))
                    for mode,z in outputs.items():arrays[mode][:,y:y+512,x:x+512]+=z[0].float().cpu().numpy()*window
                    denominator[y:y+512,x:x+512]+=window
        for mode,array in arrays.items():
            for y in range(0,h,64):
                z=np.asarray(array[:,y:y+64])/denominator[None,y:y+64];z-=z.max(0,keepdims=True)
                np.exp(z,out=z);z/=z.sum(0,keepdims=True);array[:,y:y+64]=z
            array.flush();array._mmap.close();(folder/(mode+'.partial.npy')).replace(folder/(mode+'.npy'))
        if reader:
            coords=np.asarray(coords);stats=np.asarray(statistics)
            centers=np.array([(y+32,x+32) for y in range(0,h,64) for x in range(0,w,64)])
            blocks=np.empty((len(centers),6),np.float32)
            for i in range(0,len(centers),512):
                nearest=((centers[i:i+512,None]-coords[None])**2).sum(2).argmin(1);blocks[i:i+512]=stats[nearest]
            np.save(folder/'sar_blocks.npy',blocks)
            (folder/'sar.json').write_text(json.dumps(dict(available=bool(reader.available['sar']))));reader.close()
        image_sha=hashlib.sha256((a.data_root/'images'/(sid+'.tif')).read_bytes()).hexdigest()
        (folder/(a.model+'_receipt.json')).write_text(json.dumps(dict(scene=sid,fold=a.fold,model=a.model,image_sha256=image_sha,labels_read=False)))
        print(sid,flush=True)

if __name__=='__main__':main()
