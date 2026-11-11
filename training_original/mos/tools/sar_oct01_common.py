"""Immutable assets and model construction for the three October 1 trials."""
import json
import os
from pathlib import Path
import time
import torch
from tools.train_target_dino import atomic_json as write, sha256 as sha, canonical_hash as digest
from tools.train_ms75_sar_joint import make_optical, state_hash, subset, apply_delta

SCHEMA='sar_oct01_three_directions_v1'
ARMS={1:('coarse_gate','native_gate','native_match'),2:('plain','dual_keep'),3:('repr_none','repr_split')}
GPUS={1:(2,3),2:(4,5),3:(6,7)}


def verify(c,full=True):
    if c['schema']!=SCHEMA or c['arms']!=list(ARMS[c['direction']]):raise ValueError('Protocol identity changed')
    if c['physical_gpus']!=list(GPUS[c['direction']]):raise ValueError('GPU assignment changed')
    if (c['epochs'],c['updates_per_epoch'],c['effective_batch'],c['seed'])!=((4 if 'continuation' in c else 2),480,64,1337):raise ValueError('Budget changed')
    if 'continuation' in c:
        from tools.sar_oct01_extension import validate_protocol
        validate_protocol(c)
    if c['auto_full'] or c['official_test_auto']:raise ValueError('No automatic full120/official30')
    if full:
        for p,h in c['files'].items():
            if sha(p)!=h:raise ValueError('Bound source/asset changed: '+p)
    parent=json.loads(Path(c['parent_config']).read_text())
    if c['files'].get(c['parent_checkpoint'])!='0900cd1ad96ddce76372df4916958158d70440fee1589e9c206481ae9e822af8':raise ValueError('Wrong parent')
    return parent


def setup(c,rank=0,one_gpu=False):
    from tools.launch_without_thp import disable_thp
    disable_thp()
    os.sched_setaffinity(0,set(c['cpu_affinity'][rank]))
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    expected=c['gpu_uuids'][rank] if one_gpu else ','.join(c['gpu_uuids'])
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=expected:raise ValueError('Wrong CUDA visibility')


def make_model(c,parent,arm,device='cuda'):
    from tcmmsline3.sar_oct01_models import MS75NativeExpert, RepairExpert
    from tools.run_sar_route_diagnostics import load_parent
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(c['seed'])
        if c['direction']==2:model=RepairExpert(load_parent(c,parent))
        else:model=MS75NativeExpert(make_optical(parent),parent,arm)
    return model.to(device)


def inventories(model):
    keys=tuple(n for n,p in model.named_parameters() if p.requires_grad)
    frozen=tuple(k for k in model.state_dict() if k not in set(keys))
    return keys,frozen


def validate_candidate(payload,c,arm,model,smoke=False):
    from tools.sar_oct01_extension import interval
    keys,frozen=inventories(model)
    if (payload.get('schema')!=SCHEMA or payload.get('config_sha256')!=digest(c)
        or payload.get('arm')!=arm or payload.get('updates')!=interval(c,smoke)[1]
        or payload.get('keys')!=list(keys) or payload.get('frozen_hash')!=state_hash(subset(model,frozen))):
        raise ValueError('Candidate identity or frozen state differs')
    apply_delta(model,payload['state_dict'])
    if state_hash(subset(model,keys))!=state_hash(payload['state_dict']):raise ValueError('Candidate reload differs')


def progress(out,**values):
    row=dict(pid=os.getpid(),time_unix=time.time(),**values)
    write(Path(out)/'progress.json',row)
    print(json.dumps(row),flush=True)
