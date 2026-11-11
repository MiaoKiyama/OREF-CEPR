"""Three independent, bounded train96/held24 mechanism screens on GPU1/6/7.

No full120, no official30, no production teachers and no parent weight update.
Feature probes use fixed grid crops and native pixel-centre subsampling; their
scores are explicitly NOT full-scene segmentation or three-model fusion.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from tcmmsline3.sar_route_diagnostics import (AffinityProbe,ConditionalProbe,common_private,
    ridge_fit,shifted_pair,sample_delta,gradient_geometry,summarize_confusion)

SCHEMA='sar_three_route_diagnostics_gpu167_20260930_v1'
PARENT_SHA='0900cd1ad96ddce76372df4916958158d70440fee1589e9c206481ae9e822af8'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False))
    temp.replace(path)


def progress(out,**values):
    value=dict(time_unix=time.time(),**values)
    write(out/'progress.json',value);print(json.dumps(value),flush=True)


def verify(config):
    from tools.train_convnext_sar import validate_config
    if config['schema']!=SCHEMA or config['parent_checkpoint_sha256']!=PARENT_SHA:
        raise ValueError('Diagnostic/parent identity changed')
    if (config['train_windows'],config['held_windows'],config['seeds'],config['probe_epochs'],config['gradient_samples'])!=(16,16,[1337,2337,3337],12,256):
        raise ValueError('Prespecified diagnostic budget changed')
    if config['auto_full'] or config['official_test_auto']:raise ValueError('No automatic final training/test')
    for path,digest in config['files'].items():
        if sha(path)!=digest:raise ValueError('Bound input changed: '+path)
    training=json.loads(Path(config['parent_config']).read_text())
    validate_config(training,devices=False)
    # Parent paths remain anchored to its sealed project, not this new ROOT.
    for path,digest in training['files'].items():
        if sha(path)!=digest:raise ValueError('Parent bound input changed: '+path)
    return training


def load_parent(config,training):
    from tools.train_convnext_sar import make_model,apply_delta,state_hash,subset
    from tools.evaluate_convnext_sar import validate_delta
    payload=torch.load(config['parent_checkpoint'],map_location='cpu',weights_only=True)
    validate_delta(payload,training,epoch=16)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(training['seed']);model=make_model(training,device='cpu')
    keys=tuple(k for k in model.state_dict() if k not in set(model.mutable_keys()))
    if state_hash(subset(model,keys))!=payload['frozen_hash']:raise ValueError('Frozen parent state mismatch')
    apply_delta(model,payload['state_dict']);model.set_stage('joint');model.eval();model.requires_grad_(False)
    del payload
    return model.cuda()


def grid_positions(shape,scene,count=16):
    # Pure image geometry. No label, prediction or SAR quality affects selection.
    ys=np.rint(np.linspace(0,shape[0]-512,4)).astype(int)
    xs=np.rint(np.linspace(0,shape[1]-512,4)).astype(int)
    rows=[(int(y),int(x)) for y in ys for x in xs]
    if len(set(rows))!=16 or min(shape)<512:raise ValueError('Scene cannot support fixed grid')
    rows.sort(key=lambda p:hashlib.sha256(f'20260930:{scene}:{p}'.encode()).hexdigest())
    return rows[:count]


def project(value,channels,seed):
    generator=torch.Generator(device=value.device).manual_seed(seed)
    matrix=torch.randn((channels,value.shape[1],1,1),generator=generator,device=value.device,dtype=torch.float32)/value.shape[1]**.5
    return F.conv2d(value.float(),matrix)


@torch.no_grad()
def extract_batch(model,batch):
    from tcmmsline3.convnext_sar_model import fixed_sar_intensity,align_croma_to_ms
    with torch.autocast('cuda',dtype=torch.bfloat16):
        optical,spectral=model._optical_features(batch['ms'])
        logits=model.decoder(optical,spectral)
        real,content=model._encode_pair(batch['sar_db'],batch['sar_valid'],'real')
        observed=F.avg_pool2d(F.pad(batch['sar_valid'].float(),(1,1,1,1)),8,8)>0
        intensity=model.sar_intensity_encoder(fixed_sar_intensity(batch['sar_db'],batch['sar_valid']))
        intensity=F.avg_pool2d(F.pad(intensity*batch['sar_valid'],(1,1,1,1)),8,8)*observed
    opt=torch.cat((project(optical[2],64,1101),F.interpolate(project(optical[3],32,1102),(32,32),mode='bilinear',align_corners=False),project(spectral[3],32,1103)),1)
    sar=torch.cat((align_croma_to_ms(project(real,64,1104),(32,32)),align_croma_to_ms(project(content,32,1105),(32,32)),align_croma_to_ms(intensity.float(),(32,32))),1)
    mask=(F.avg_pool2d(batch['optical_valid'].float(),16,16)>.999)&(align_croma_to_ms(observed.float(),(32,32))>.999)
    anchor=logits[:,:,2::4,2::4].float().log_softmax(1)
    return dict(optical=opt.half(),sar=sar.half(),mask=mask,anchor=anchor.half(),
        native_mask=batch['optical_valid'][:,:,2::4,2::4],label=batch['label'][:,2::4,2::4].to(torch.uint8))


def feature_cache(config,training,out,smoke=False):
    from tcmmsline3.ms75_rebuild_data import load_manifest,read_cached_scene
    from tcmmsline3.target_dino_data import normalize_bgrn
    from tcmmsline3.light_sar_data import open_sar_scene,_checked_aux
    from tools.train_ms75_sar_joint import state_hash
    model=load_parent(config,training);before=state_hash(model.state_dict())
    meta=load_manifest(training['data']['raw_manifest'])
    started=time.time();identity=[]
    for split,scenes in [('train',meta['selection_train_ids']),('held',meta['held_ids'])]:
        if smoke:scenes=scenes[:1]
        count=2 if smoke else config[split+'_windows']
        n=len(scenes)*count;arrays={};rows=[]
        hist=np.zeros(24,np.int64)
        for scene_index,scene in enumerate(scenes):
            image,label=read_cached_scene(meta,scene)
            # Authenticate content once per scene, before selecting any windows.
            for key,value in [('image',image),('label',label)]:
                if hashlib.sha256(memoryview(value)).hexdigest()!=meta['scenes'][scene][key+'_sha256']:
                    raise ValueError('Cached raster changed '+scene+' '+key)
            reader=open_sar_scene(training['data']['data_root'],scene)
            try:
                positions=grid_positions(label.shape,scene,count)
                def read_one(position):
                    top,left=position
                    ms=normalize_bgrn(image[top:top+512,left:left+512],meta['normalization'])
                    # This closure is used serially: each native raster reader is not shared.
                    if reader.available['sar']:
                        db,sv,ov=_checked_aux(reader.read_patch(top,left,512,modality='sar',out_size=166),512,166)
                    else:
                        db=np.zeros((2,166,166),np.float32);sv=np.zeros((1,166,166),bool);ov=np.zeros((1,512,512),bool)
                    return dict(ms=ms,sar_db=db,sar_valid=sv,optical_valid=ov,label=np.array(label[top:top+512,left:left+512],copy=True))
                # A single CPU producer overlaps native I/O with GPU batches.
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future=pool.submit(lambda:[read_one(p) for p in positions[:4]])
                    for begin in range(0,len(positions),4):
                        samples=future.result()
                        if begin+4<len(positions):future=pool.submit(lambda ps=positions[begin+4:begin+8]:[read_one(p) for p in ps])
                        batch={key:torch.from_numpy(np.stack([x[key] for x in samples])).cuda() for key in samples[0]}
                        values=extract_batch(model,batch)
                        for key,value in values.items():
                            arr=value.cpu().numpy()
                            if key not in arrays:arrays[key]=np.lib.format.open_memmap(out/f'{split}_{key}.npy',mode='w+',dtype=arr.dtype,shape=(n,*arr.shape[1:]))
                            offset=scene_index*count+begin;arrays[key][offset:offset+len(samples)]=arr
                        for j,p in enumerate(positions[begin:begin+4]):
                            y=values['label'][j].cpu().numpy();hist+=np.bincount(y[y<24],minlength=24)
                            rows.append(dict(scene=scene,top=p[0],left=p[1],has_sar=bool(reader.available['sar'])))
                        del batch,values
            finally:reader.close()
            if scene_index%4==0 or scene_index+1==len(scenes):
                progress(out,stage='extract_'+split,scenes=scene_index+1,total_scenes=len(scenes),crops=len(rows),seconds=time.time()-started)
        for arr in arrays.values():arr.flush()
        del arrays
        record=dict(split=split,rows=rows,label_histogram=hist.tolist(),classes_seen=np.flatnonzero(hist).tolist(),
            pixel_selection='native512 centres 2+4*k; all16 fixed4x4 grid crops per scene; no full-scene stitching',
            feature_dim=128,feature_grid=32)
        write(out/f'{split}_cache.json',record)
        identity.append(dict(split=split,crops=n,scenes=len(scenes),classes_seen=record['classes_seen']))
    if state_hash(model.state_dict())!=before:raise RuntimeError('Feature extraction changed parent')
    receipt=dict(parent_state_sha256=before,unchanged=True,coverage=identity,elapsed_seconds=time.time()-started,
        max_memory_gib=torch.cuda.max_memory_allocated()/2**30,
        files={p.name:sha(p) for p in sorted(out.glob('*_*.npy'))})
    write(out/'cache_receipt.json',receipt)
    del model;gc.collect();torch.cuda.empty_cache()
    return receipt


def load_cache(out,split):
    keys=('optical','sar','mask','anchor','native_mask','label')
    return {k:torch.from_numpy(np.load(out/f'{split}_{k}.npy')).cuda() for k in keys}


def normalizer(train,key):
    x=train[key].float();mask=train['mask'].float()
    n=mask.sum().clamp_min(1)
    mean=(x*mask).sum((0,2,3),keepdim=True)/n
    std=(((x-mean).square()*mask).sum((0,2,3),keepdim=True)/n).sqrt().clamp_min(.05)
    return mean,std


def normalized(data,stats):
    for key,(mean,std) in stats.items():data[key]=((data[key].float()-mean)/std).clamp(-10,10).half()
    wrong,mask=shifted_pair(data['sar'],data['mask'].float())
    data['wrong']=wrong;data['mask']=mask
    data['gate']=data['native_mask'].float()*(sample_delta(mask)>.999)
    return data


def predict(probe,batch,route,arm,weights=None,intervention='real'):
    optical=batch['optical'].float();sar=batch['sar'].float()
    mask=batch['mask'].float()
    if intervention=='wrong' or arm.endswith('_mismatch'):sar=batch['wrong'].float()
    if route=='affinity':
        guide=optical if arm=='optical' else sar
        delta=probe(optical,guide,mask)
    else:
        common,private=common_private(optical,sar,weights)
        if arm=='optical':first,second=optical,optical
        elif arm=='raw':first,second=sar,sar
        elif arm=='private':first,second=private,private
        else:first,second=common,private
        delta=probe(optical,first,second,mask)
    residual=sample_delta(delta)*batch['gate']
    if intervention=='off':residual=residual*0
    return batch['anchor'].float()+residual


@torch.no_grad()
def evaluate(probe,data,rows,route,arm,weights,intervention='real',source=None):
    cms={k:torch.zeros((24,24),dtype=torch.int64,device='cuda') for k in ('all_sampled','sar_supported','no_sar','source_unshared')}
    scene_cms={};nll=0.;valid_total=0;off_error=0.
    for begin in range(0,len(rows),8):
        b={k:v[begin:begin+8] for k,v in data.items()};logits=predict(probe,b,route,arm,weights,intervention)
        target=b['label'].long();valid=target<24;pred=logits.argmax(1)
        safe=target.clamp_max(23)
        nll+=float((-logits.log_softmax(1).gather(1,safe[:,None])[:,0]*valid).sum());valid_total+=int(valid.sum())
        no_sar=~b['native_mask'].flatten(1).any(1)
        if bool(no_sar.any()):off_error=max(off_error,float((logits[no_sar]-b['anchor'][no_sar].float()).abs().max()))
        for j,row in enumerate(rows[begin:begin+8]):
            def cm_for(mask):return torch.bincount((safe[j][mask]*24+pred[j][mask]),minlength=576).reshape(24,24)
            cm=cm_for(valid[j]);cms['all_sampled']+=cm
            scene_cms.setdefault(row['scene'],torch.zeros_like(cm));scene_cms[row['scene']]+=cm
            cms['sar_supported']+=cm_for(valid[j]&(b['gate'][j,0]>.999))
            if no_sar[j]:cms['no_sar']+=cm
            if source and row['scene'] in source['held_unshared_scenes']:cms['source_unshared']+=cm
    if off_error!=0:raise RuntimeError('No-SAR fallback changed')
    return dict(metrics={k:summarize_confusion(v) for k,v in cms.items()},
        scene_metrics={k:summarize_confusion(v) for k,v in scene_cms.items()},
        nll=nll/max(1,valid_total),missing_sar_max_logit_difference=off_error,
        label='sampled native-pixel conditional probe; NOT full held24 or DINO fusion')


def run_probes(config,training,out,route,smoke=False):
    feature_cache(config,training,out,smoke)
    train,held=load_cache(out,'train'),load_cache(out,'held')
    train_rows=json.loads((out/'train_cache.json').read_text())['rows'];held_rows=json.loads((out/'held_cache.json').read_text())['rows']
    if {r['scene'] for r in train_rows}&{r['scene'] for r in held_rows}:raise ValueError('Scene leakage')
    source=json.loads(Path(config['source_support']).read_text())
    stats={k:normalizer(train,k) for k in ('optical','sar')}
    train=normalized(train,stats);held=normalized(held,stats)
    weights=None
    if route=='private':
        # Fixed every-other token subsample. No class labels in the fit.
        x=train['optical'][:,:,::2,::2].float().permute(0,2,3,1).reshape(-1,128)
        y=train['sar'][:,:,::2,::2].float().permute(0,2,3,1).reshape(-1,128)
        mask=train['mask'][:,:,::2,::2].reshape(-1)>.999
        weights=ridge_fit(x,y,mask)
        write(out/'decomposition.json',dict(fit_scenes=sorted({r['scene'] for r in train_rows}),fit_tokens=int(mask.sum()),ridge=.01,
            definition='common=linear train-only prediction of SAR from optical; private=observed SAR-common; not DeCUR',held_used_for_fit=False))
        del x,y,mask
    torch.save(dict(stats=stats,weights=weights),out/'train_only_transforms.pt')
    arms=['optical','sar','sar_mismatch'] if route=='affinity' else ['optical','raw','split','private','split_mismatch']
    seeds=[1337] if smoke else config['seeds'];epochs=1 if smoke else config['probe_epochs']
    results=[];started=time.time()
    for seed in seeds:
        for arm in arms:
            torch.manual_seed(seed);probe=(AffinityProbe() if route=='affinity' else ConditionalProbe()).cuda()
            optimizer=torch.optim.AdamW(probe.parameters(),lr=.001,weight_decay=.01)
            gen=torch.Generator().manual_seed(seed)
            initial=evaluate(probe,held,held_rows,route,arm,weights,source=source)
            baseline=evaluate(probe,held,held_rows,route,arm,weights,'off',source)
            if initial['metrics']!=baseline['metrics']:raise RuntimeError('Zero initial residual changed baseline')
            for epoch in range(1,epochs+1):
                order=torch.randperm(len(train_rows),generator=gen);loss_sum=0.;steps=0
                for begin in range(0,len(order),8):
                    ids=order[begin:begin+8].cuda();b={k:v[ids] for k,v in train.items()}
                    optimizer.zero_grad(set_to_none=True)
                    logits=predict(probe,b,route,arm,weights);target=b['label'].long()
                    valid=target<24
                    loss=(F.cross_entropy(logits,target,ignore_index=255,reduction='none')*valid).sum()/valid.sum().clamp_min(1)
                    if not torch.isfinite(loss):raise FloatingPointError('Probe loss diverged')
                    loss.backward();torch.nn.utils.clip_grad_norm_(probe.parameters(),1.)
                    optimizer.step();loss_sum+=float(loss.detach());steps+=1
                for group in optimizer.param_groups:group['lr']=.001*.5*(1+np.cos(np.pi*epoch/epochs))
                progress(out,stage='probe_train',route=route,seed=seed,arm=arm,epoch=epoch,epochs=epochs,loss=loss_sum/steps,elapsed_seconds=time.time()-started)
                if epoch in (4,8,12) or smoke:
                    conditions=['real'] if arm=='optical' or arm.endswith('_mismatch') else ['real','wrong','off']
                    for intervention in conditions:
                        result=evaluate(probe,held,held_rows,route,arm,weights,intervention,source)
                        row=dict(seed=seed,arm=arm,epoch=epoch,intervention=intervention,parameter_count=sum(p.numel() for p in probe.parameters()),**result)
                        results.append(row);write(out/f'result_{seed}_{arm}_e{epoch}_{intervention}.json',row)
                if smoke:break
            torch.save(probe.state_dict(),out/f'probe_{seed}_{arm}.pt')
            del optimizer,probe
    summary=dict(status='completed',schema=SCHEMA,route=route,parent_checkpoint_sha256=PARENT_SHA,
        protocol='frozen e16; three seeds; fixed12 epochs; all96 train and all24 held scenes but sampled pixels only',
        full_held24=False,three_model_fusion=False,held_used_for_optimization=False,auto_full=False,
        limitations=['Compressed frozen-feature probe, not a retrained full model','Same-crop feature shift is a spatial control, not another acquisition','Source-unshared subset is additionally reported; held24 is not a new blind test'],
        results=[{k:v for k,v in r.items() if k not in ('scene_metrics','metrics')}|{'metrics':{k:{kk:vv for kk,vv in m.items() if kk!='confusion_matrix'} for k,m in r['metrics'].items()}} for r in results])
    write(out/'completed.json',summary)


def run_gradients(config,training,out,smoke=False):
    from tools.train_convnext_sar import compute_losses
    from tools.train_ms75_sar_joint import sar_soft_loss,state_hash
    from tcmmsline3.ms75_sar_joint_data import MS75SARJointDataset
    from tools.audit_convnext_sar_inputs import choose_probe_indices
    model=load_parent(config,training);model.train()
    named=[(name,p) for name,p in model.named_parameters() if id(p) in model._sar_tail_ids]
    for _,p in named:p.requires_grad_(True)
    before=state_hash(model.state_dict());dataset=MS75SARJointDataset(training,epoch=16)
    indices=choose_probe_indices(dataset,config['gradient_samples'])
    if smoke:
        indices=[i for i in indices if dataset.availability[dataset.blocks[i//16][0]]][:2]
    records=[];started=time.time()
    try:
        for number,index in enumerate(indices):
            sample=dataset[index]
            row=dict(index=int(index),scene=sample['scene_id'],sampling_kind=sample.get('sampling_kind'),
                label_histogram=torch.bincount(sample['label'][sample['label']<24].flatten().long(),minlength=24).tolist())
            if not bool(sample['sar_valid'].any()&sample['optical_valid'].any()):
                records.append(dict(**row,skipped='missing SAR support'));continue
            batch={k:sample[k][None].cuda() for k in ('ms','sar_db','sar_valid','optical_valid','label')}
            # Reproduce training-mode stochastic branches, fixed per crop.
            torch.manual_seed(20260930+index);torch.cuda.manual_seed_all(20260930+index)
            with torch.autocast('cuda',dtype=torch.bfloat16):output=model(**{k:v for k,v in batch.items() if k!='label'})
            cfg=dict(training,optical_aux_weight=0.,spectral_aux_weight=0.,sar_aux_weight=0.)
            main,_=compute_losses({'logits':output['logits']},batch,cfg,'optical_warmup')
            auxiliary,support=sar_soft_loss(output['sar_aux_logits'],batch['label'],batch['optical_valid'],output['sar_aux_valid'])
            params=[p for _,p in named]
            gm=torch.autograd.grad(main,params,retain_graph=True,allow_unused=False)
            ga=torch.autograd.grad(auxiliary,params,allow_unused=False)
            total=gradient_geometry(gm,ga,training['sar_aux_weight'])
            blocks=defaultdict(list)
            for i,(name,_) in enumerate(named):blocks[name.rsplit('.',2)[0]].append(i)
            block_stats={key:gradient_geometry([gm[i] for i in ix],[ga[i] for i in ix],training['sar_aux_weight']) for key,ix in blocks.items()}
            row.update(gradient=total,blocks=block_stats,main_loss=float(main.detach()),auxiliary_loss=float(auxiliary.detach()),support=float(support))
            records.append(row)
            with (out/'crop_gradients.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            del output,main,auxiliary,gm,ga,batch
            if number%8==0 or number+1==len(indices):progress(out,stage='gradient_audit',crops=number+1,total_crops=len(indices),last_cosine=total['cosine'],seconds=time.time()-started)
    finally:dataset.close()
    if state_hash(model.state_dict())!=before:raise RuntimeError('Gradient diagnostic updated parent')
    valid=[r for r in records if r.get('gradient',{}).get('cosine') is not None]
    if not valid:raise RuntimeError('No usable shared parameter gradients')
    def aggregate(rows):
        cosine=np.array([r['gradient']['cosine'] for r in rows]);ratio=np.array([r['gradient']['weighted_norm_ratio'] for r in rows]);projection=np.array([r['gradient']['combined_main_projection_ratio'] for r in rows])
        return dict(crops=len(rows),cosine_mean=float(cosine.mean()),cosine_median=float(np.median(cosine)),conflict_fraction=float((cosine<0).mean()),weighted_aux_main_ratio_median=float(np.median(ratio)),reversed_main_projection_fraction=float((projection<0).mean()))
    by_scene={s:aggregate([r for r in valid if r['scene']==s]) for s in sorted({r['scene'] for r in valid})}
    by_class={str(c):aggregate([r for r in valid if r['label_histogram'][c]>=256]) for c in range(24) if any(r['label_histogram'][c]>=256 for r in valid)}
    result=dict(status='completed',schema=SCHEMA,route='gradient',parent_checkpoint_sha256=PARENT_SHA,parent_unchanged=True,
        sampled_crops=len(records),usable_crops=len(valid),train_scenes=len({r['scene'] for r in records}),aggregate=aggregate(valid),by_scene=by_scene,by_class_present_in_crop=by_class,
        shared_parameter_names=[n for n,_ in named],shared_parameter_count=sum(p.numel() for _,p in named),auxiliary_weight=training['sar_aux_weight'],
        class_summary_definition='Stratified by >=256 GT pixels of class in a crop; NOT gradients of that class alone',
        limitations=['Instantaneous shared-tail gradients at e16, not a training trajectory','Gradient projection is first-order SGD geometry, not the actual Adam update','Negative cosine alone does not establish harm; inspect weighted magnitude and main projection'],
        seconds=time.time()-started,max_memory_gib=torch.cuda.max_memory_allocated()/2**30)
    write(out/'all_crops.json',records);write(out/'completed.json',result)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--route',choices=['affinity','private','gradient'],required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    config=json.loads(Path(args.config).read_text());spec=config['jobs'][args.route]
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=spec['uuid']:raise ValueError('Wrong physical GPU binding')
    os.sched_setaffinity(0,set(spec['cpu_affinity']))
    from tools.launch_without_thp import disable_thp
    disable_thp();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    out=Path(config['output_dir'])/args.route/('smoke' if args.smoke else 'formal')
    out.mkdir(parents=True,exist_ok=False)
    try:
        training=verify(config);progress(out,stage='verified',route=args.route,smoke=args.smoke,physical_gpu=spec['gpu'])
        if args.route=='gradient':run_gradients(config,training,out,args.smoke)
        else:run_probes(config,training,out,args.route,args.smoke)
        progress(out,stage='completed',route=args.route,smoke=args.smoke)
    except Exception as error:
        write(out/'failed.json',dict(type=type(error).__name__,message=str(error),time_unix=time.time()));raise


if __name__=='__main__':main()
