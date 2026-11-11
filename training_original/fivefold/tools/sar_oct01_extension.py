"""Authenticated e2 -> e4 continuation; preserve the original two-epoch trial."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def interval(c, smoke=False):
    start = 960 if 'continuation' in c else 0
    return start, start + (3 if smoke else 960)


def learning_rate(c, update):
    from tools.train_target_dino import lr_multiplier
    start, target = interval(c)
    if not start <= update < target:
        raise ValueError('Update outside the configured stage')
    return (5e-5 if start else 1e-4) * lr_multiplier(update-start, target-start, 48)


def validate_protocol(c):
    from tools.train_target_dino import canonical_hash, sha256
    ext = c['continuation']
    if (ext['start_updates'], ext['target_updates'], ext['lr'], ext['warmup_updates']) != (960, 1920, 5e-5, 48):
        raise ValueError('Continuation schedule changed')
    if c['epochs'] != 4 or c['training']['sampling_epochs'] != [18, 19]:
        raise ValueError('Continuation sampling/budget changed')
    if c['training']['selected_epoch'] != 4 or c['training']['checkpoint_epochs'] != [3, 4]:
        raise ValueError('Continuation checkpoints changed')
    if c['training']['lr'] != 5e-5 or c['training']['lr_warmup_updates'] != 48:
        raise ValueError('Continuation LR metadata differs')
    path = ext['source_config']
    old = json.loads(Path(path).read_text())
    if canonical_hash(old) != ext['source_config_sha256'] or c['files'].get(path) != sha256(path):
        raise ValueError('Original configuration changed')
    if old['epochs'] != 2 or 'continuation' in old:
        raise ValueError('Only the original two-epoch run may be extended')
    protected = ('schema','direction','arms','physical_gpus','gpu_uuids','cpu_affinity',
                 'seed','updates_per_epoch','effective_batch','micro_batch','parent_config',
                 'parent_checkpoint','held_aux_bindings','reference_interventions','evaluation')
    if any(c[k] != old[k] for k in protected):
        raise ValueError('Paired experiment or evaluation contract changed')
    if c['output_dir'] == old['output_dir'] or c['project_dir'] == old['project_dir']:
        raise ValueError('Continuation must preserve the original directories')
    if any(c['files'].get(p) != h for p,h in old['files'].items()):
        raise ValueError('Original immutable bindings lost')
    return old


def tree_hash(value):
    """Device-independent digest for optimizer moments and RNG checkpoints."""
    h = hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            a = v.detach().cpu().contiguous()
            h.update(f'tensor:{a.dtype}:{list(a.shape)}:'.encode())
            h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v,np.ndarray):
            h.update(f'array:{v.dtype}:{v.shape}:'.encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            h.update(b'dict{')
            for key in sorted(v,key=lambda k:(type(k).__name__,str(k))):visit(key);visit(v[key])
            h.update(b'}')
        elif isinstance(v,(list,tuple)):
            h.update(type(v).__name__.encode()+b'[')
            for item in v:visit(item)
            h.update(b']')
        else:h.update((type(v).__name__+':'+repr(v)+';').encode())
    visit(value)
    return h.hexdigest()


def restore_state(ck,model,optimizer,*,identity,arm,frozen_hash,world,start,target):
    from tools.sar_oct01_common import SCHEMA
    from tools.train_ms75_sar_joint import apply_delta,subset,state_hash
    if (ck.get('schema') != SCHEMA or ck.get('config_sha256') != identity
            or ck.get('arm') != arm or ck.get('frozen_hash') != frozen_hash
            or type(ck.get('updates')) is not int or not start <= ck['updates'] <= target
            or [r.get('rank') for r in ck.get('rng',[])] != list(range(world))):
        raise ValueError('Recovery identity, updates or RNG ranks differ')
    keys = tuple(model.mutable_keys())
    if set(ck['ema']) != set(keys):raise ValueError('EMA inventory differs')
    # Validate EMA with the same finite/shape/dtype contract, then restore raw weights.
    apply_delta(model,ck['ema']);apply_delta(model,ck['model'])
    optimizer.load_state_dict(ck['optimizer'])
    if tree_hash(optimizer.state_dict()) != tree_hash(ck['optimizer']):
        raise ValueError('Optimizer state did not restore exactly')
    if state_hash(subset(model,keys)) != state_hash(ck['model']):
        raise ValueError('Raw weights did not restore exactly')
    return dict(raw_sha256=state_hash(ck['model']),ema_sha256=state_hash(ck['ema']),
                optimizer_sha256=tree_hash(ck['optimizer']),rng_sha256=tree_hash(ck['rng']),
                source_updates=ck['updates'])


def bind_parent(c):
    """Seal completed original training/evaluation assets before any continuation."""
    from tools.sar_oct01_common import digest,sha,write
    old = validate_protocol(c);root = Path(old['output_dir'])
    state = json.loads((root/'pipeline_state.json').read_text())
    done = json.loads((root/'held24/completed.json').read_text())
    if (state['status'] != 'completed' or state['config_sha256'] != digest(old)
            or done['status'] != 'completed' or done['scene_count'] != 24
            or done['config_sha256'] != digest(old)
            or done['metrics_sha256'] != sha(root/'held24/metrics.json')):
        raise ValueError('Original full held24 is incomplete or changed')
    files = {str(root/n):sha(root/n) for n in ('pipeline_state.json','held24/completed.json','held24/metrics.json')}
    for arm in c['arms']:
        p=root/arm/'formal';receipt=json.loads((p/'completed.json').read_text())
        if (receipt['config_sha256'] != digest(old) or receipt['updates'] != 960
                or receipt['arm'] != arm or receipt['candidate_sha256'] != sha(p/'epoch002_ema.pt')):
            raise ValueError('Original arm is not a completed two-epoch run')
        for n in ('completed.json','last.pt','epoch002_ema.pt'):files[str(p/n)]=sha(p/n)
    result=dict(status='passed',config_sha256=digest(c),source_config_sha256=digest(old),files=files)
    write(Path(c['output_dir'])/'parent_ready.json',result)
    return result


def parent_checkpoint(c,arm):
    from tools.sar_oct01_common import digest,sha
    from tools.train_ms75_sar_joint import state_hash
    old=validate_protocol(c);ready=json.loads((Path(c['output_dir'])/'parent_ready.json').read_text())
    if ready['status'] != 'passed' or ready['config_sha256'] != digest(c):raise ValueError('Missing parent binding')
    root=Path(old['output_dir'])/arm/'formal'
    for n in ('completed.json','last.pt','epoch002_ema.pt'):
        path=root/n
        if ready['files'].get(str(path)) != sha(path):raise ValueError('Parent candidate changed: '+str(path))
    ck=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
    candidate=torch.load(root/'epoch002_ema.pt',map_location='cpu',weights_only=True)
    if (candidate['config_sha256'] != digest(old) or candidate['arm'] != arm or candidate['updates'] != 960
            or state_hash(candidate['state_dict']) != state_hash(ck['ema'])):
        raise ValueError('Recovery EMA differs from evaluated epoch2 candidate')
    return ck,digest(old)


def compare_epochs(c):
    from tools.sar_oct01_common import digest,sha,write
    old=validate_protocol(c);out=Path(c['output_dir'])
    ready=json.loads((out/'parent_ready.json').read_text())
    oldpath=Path(old['output_dir'])/'held24/metrics.json';newpath=out/'held24/metrics.json'
    if ready['files'].get(str(oldpath)) != sha(oldpath):raise ValueError('Epoch2 metrics changed')
    before=json.loads(oldpath.read_text());after=json.loads(newpath.read_text())
    if before['epoch'] != 2 or after['epoch'] != 4:raise ValueError('Wrong evaluation epochs')
    if before['config_sha256'] != digest(old) or after['config_sha256'] != digest(c):
        raise ValueError('Evaluation identity mismatch')
    baselines=['base','base_triple','dual']
    if c['direction']==2:baselines+=['parent_off','parent_off_triple','parent_real','parent_real_triple']
    rows=[]
    for group in ('all24','hasSAR21','noSAR3'):
        a=before['groups'][group];b=after['groups'][group]
        for name in baselines:
            if a[name]['confusion_matrix']!=b[name]['confusion_matrix']:
                raise ValueError('Baseline drift between epochs: '+name)
        for arm in c['arms']:
            record=dict(group=group,arm=arm)
            for suffix,kind in (('','single'),('_triple','triple')):
                name=arm+'_real'+suffix
                if a[name]['target_pixels']!=b[name]['target_pixels']:raise ValueError('GT coverage changed')
                record[kind]={
                    'epoch2_miou_percent':100*a[name]['miou'],
                    'epoch4_miou_percent':100*b[name]['miou'],
                    'epoch4_minus_epoch2_pp':100*(b[name]['miou']-a[name]['miou']),
                    'epoch4_real_minus_off_pp':100*(b[name]['miou']-b[arm+'_off'+suffix]['miou']),
                    'epoch4_real_minus_mismatch_pp':100*(b[name]['miou']-b[arm+'_mismatch'+suffix]['miou']),
                    'epoch4_real_minus_MS75_pp':100*(b[name]['miou']-b['base'+suffix]['miou']),
                    'class_epoch4_minus_epoch2_pp':[100*(y-x) for x,y in zip(a[name]['class_iou'],b[name]['class_iou'])]}
            rows.append(record)
    write(out/'epoch2_vs_epoch4.json',dict(status='completed',config_sha256=digest(c),
        epoch2_metrics_sha256=sha(oldpath),epoch4_metrics_sha256=sha(newpath),rows=rows,
        scope='Full held24 development, paired 2+2 continuation, not a fresh four-epoch cosine run',
        official30_started=False,full120_started=False))
