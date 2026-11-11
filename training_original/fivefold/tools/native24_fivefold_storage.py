"""Round-trip checked, campaign-owned checkpoint pool across data disks.

Logical checkpoint paths are atomic symlinks. The previous payload remains
readable until the new payload is verified and the symlink is committed.
No existing experiment files are eligible for replacement or cleanup.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid


def below(path, root):
    path, root = Path(path).absolute(), Path(root).resolve()
    return path != root and root in path.parents


def roots_for(config):
    roots = [Path(x).resolve() for x in config['storage_roots']]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError('checkpoint pool requires unique roots')
    for root in roots:
        if root.name != 'payloads' or root.parent.name != config['campaign_name']:
            raise ValueError('checkpoint storage escapes campaign')
        root.mkdir(parents=True, exist_ok=True)
    return roots


def logical_path(path, config):
    path = Path(path).absolute()
    if not below(path, config['output_dir']) or path.suffix != '.pt':
        raise ValueError('checkpoint outside this model output')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.resolve() != path.parent:
        raise ValueError('checkpoint parent must not be a symlink')
    if path.exists() and not path.is_symlink():
        raise ValueError('refuse replacing a regular existing checkpoint')
    return path


def owned_target(path, roots):
    if not path.is_symlink():
        raise ValueError('expected campaign checkpoint link')
    target = path.resolve(strict=True)
    if target.parent not in roots or not target.name.endswith('.pt'):
        raise ValueError('linked checkpoint is not campaign-owned')
    return target


@contextmanager
def locked(config):
    path = Path(config['checkpoint_lock_file'])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def event(config, row):
    p = Path(config['checkpoint_lock_file']).with_name('storage_events.jsonl')
    with p.open('a') as f:
        f.write(json.dumps(dict(time_unix=time.time(), **row), sort_keys=True) + '\n')
        f.flush()


def relocation_plan(free, items, required, margin=64*1024**2):
    """Move only permanent, small EMA payloads to combine fragmented space."""
    for source in sorted(free, key=free.get, reverse=True):
        available=dict(free); moves=[]
        for link, physical, size in sorted((x for x in items if x[1].parent==source),key=lambda x:x[2],reverse=True):
            fits=[r for r in available if r!=source and available[r]>=size+margin]
            if not fits:continue
            destination=min(fits,key=available.get)
            moves.append((link,physical,destination,size))
            available[destination]-=size;available[source]+=size
            if available[source]>=required:return moves
    return []


def relocate_ema(config, link, old, destination):
    """Caller holds the campaign lock; the old file survives every failed copy."""
    from tools import train_target_dino as engine
    temporary=destination/(uuid.uuid4().hex+'.tmp')
    final=temporary.with_suffix('.pt')
    newlink=link.with_name(link.name+'.link_'+uuid.uuid4().hex)
    committed=False
    try:
        with old.open('rb') as src,temporary.open('xb') as dst:
            shutil.copyfileobj(src,dst,8<<20);dst.flush();os.fsync(dst.fileno())
        if old.stat().st_size!=temporary.stat().st_size or engine.sha256(old)!=engine.sha256(temporary):
            raise RuntimeError('EMA relocation checksum mismatch')
        os.replace(temporary,final);os.symlink(str(final),str(newlink));os.replace(newlink,link);committed=True
        fd=os.open(str(link.parent),os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        event(config,dict(event='ema_relocated',logical=str(link),previous=str(old),physical=str(final),bytes=final.stat().st_size))
        old.unlink()
    finally:
        if temporary.exists():temporary.unlink()
        if newlink.is_symlink():newlink.unlink()
        if not committed and final.exists():final.unlink()


def combine_free_space(config, roots, estimate):
    campaign=Path(config['checkpoint_lock_file']).resolve().parent.parent
    if campaign.name!=config['campaign_name']:raise ValueError('invalid campaign control path')
    items=[]
    for link in (campaign/'folds').rglob('*.pt'):
        if not link.is_symlink() or link.parent.name not in ('candidates','checkpoints'):continue
        if not (link.name.startswith('ema_epoch') or link.name.endswith('_ema_delta.pt')):continue
        if link.parent.resolve()!=link.parent:raise ValueError('EMA parent escapes campaign')
        physical=owned_target(link,roots);items.append((link,physical,physical.stat().st_size))
    free={root:shutil.disk_usage(root).free for root in roots}
    for link,old,destination,size in relocation_plan(free,items,estimate):
        if shutil.disk_usage(destination).free<size+64*1024**2:raise RuntimeError('free space changed during EMA relocation')
        relocate_ema(config,link,old,destination)


def save(path, payload, *, reserve_bytes=0):
    import torch
    from tools import train_target_dino as engine
    config = payload['config']
    path = logical_path(path, config)
    roots = roots_for(config)
    with locked(config):
        old = owned_target(path, roots) if path.is_symlink() else None
        estimate = engine._tensor_bytes(payload) + 64 * 1024**2
        candidates = [(shutil.disk_usage(root).free, root) for root in roots]
        candidates = [(free, root) for free, root in candidates if free >= estimate]
        if not candidates:
            combine_free_space(config,roots,estimate)
            candidates=[(shutil.disk_usage(root).free,root) for root in roots if shutil.disk_usage(root).free>=estimate]
        if not candidates:
            event(config, dict(event='insufficient_actual_checkpoint_space', path=str(path), required_bytes=estimate))
            raise RuntimeError('No data disk can hold the new checkpoint; previous state retained')
        # Place small permanent EMAs in the smallest fitting hole; large
        # rolling states use the most spacious disk to limit fragmentation.
        is_recovery = payload.get('role') in ('rolling_recovery', 'convnext_oof_recovery')
        free, root = (max if is_recovery else min)(candidates, key=lambda x: x[0])
        token = hashlib.sha256(str(path).encode()).hexdigest()[:16] + '_' + uuid.uuid4().hex
        temporary, final = root/(token+'.tmp'), root/(token+'.pt')
        link = path.with_name(path.name+'.link_'+uuid.uuid4().hex)
        committed = False
        try:
            with temporary.open('xb') as f:
                torch.save(payload, f)
                f.flush()
                os.fsync(f.fileno())
            restored = torch.load(temporary, map_location='cpu', weights_only=False, mmap=True)
            engine.checkpoint_roundtrip_validate(payload, restored)
            del restored
            os.replace(temporary, final)
            os.symlink(str(final), str(link))
            os.replace(link, path)
            committed = True
            fd = os.open(str(path.parent), os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            event(config, dict(event='checkpoint_committed', logical=str(path), physical=str(final),
                bytes=final.stat().st_size, role=payload.get('role'), updates=payload.get('updates'),
                previous=str(old) if old else None, reserve_bytes=0))
            if old is not None:
                old.unlink()
        finally:
            if temporary.exists():
                temporary.unlink()
            if link.is_symlink():
                link.unlink()
            if not committed and final.exists():
                final.unlink()


def remove_owned(path, config, reason='validated transient checkpoint'):
    path = logical_path(path, config)
    if not path.is_symlink():
        return
    if config.get('retain_complete_recovery') and path.name == 'last.pt' and 'smoke' not in str(path):
        raise ValueError('D3 complete recovery is permanently retained')
    with locked(config):
        target = owned_target(path, roots_for(config))
        event(config, dict(event='transient_checkpoint_removed', logical=str(path), physical=str(target),
            bytes=target.stat().st_size, reason=reason))
        path.unlink()
        target.unlink()
