"""FBP83: one FBP_QS_OREF_CEPR.pth plus this entry; frozen official30 inference.

python infer_fbp.py --model FBP_QS_OREF_CEPR.pth --data-root /data/fbponlymsdata --output /work/run83 --gpu 0
python infer_fbp.py --model FBP_QS_OREF_CEPR.pth --verify-only
python infer_fbp.py --model FBP_QS_OREF_CEPR.pth --output /work/run83 --evaluate --labels-root /data/fbponlymsdata/labels

The checkpoint contains the original expert weights, inference source, fixed rules,
and image-bound Q observations. First use materializes a verified runtime cache.
It is not a distilled single neural network. It requires the recorded Linux/CUDA
environment; verification and unpacking alone also work without a GPU.
"""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1', NUMPY_MADVISE_HUGEPAGE='0')
import argparse
import contextlib
import hashlib
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath

MODEL_SHA256 = '8324a07a03f58f472c0c326e1c8f8d5d66da135af356d9b3e5e4c68404d6bdbe'
MANIFEST_SHA256 = '1838c0b87c2d1538ca64efdf4192910a01efc35fc76fd7e1c29b0f631fecc65d'
FORMAT = 'fbp83_single_checkpoint_v1'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def checked_path(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or not p.parts or any(x in ('..', '.') for x in p.parts) or '\\' in relative or ':' in relative:
        raise ValueError('Unsafe embedded path: ' + relative)
    target = root.joinpath(*p.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Runtime path escapes its directory: ' + relative)
    for parent in [target, *target.parents]:
        if parent == root.parent:
            break
        if parent.is_symlink():
            raise ValueError('Runtime contains a symlink: ' + str(parent))
    return target


@contextlib.contextmanager
def runtime_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.fbp83.lock').open('a+b') as f:
        if os.name == 'nt':
            import msvcrt
            f.seek(0, 2)
            if not f.tell():
                f.write(b'0'); f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == 'nt':
                f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_checkpoint(path):
    import torch
    if digest(path) != MODEL_SHA256:
        raise ValueError('FBP_QS_OREF_CEPR.pth SHA256 mismatch; inference was not started')
    obj = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if obj.get('format') != FORMAT:
        raise ValueError('Unsupported checkpoint format')
    records = obj['files']
    if len({r['path'] for r in records}) != len(records) or set(obj['payloads']) != {r['path'] for r in records}:
        raise ValueError('Invalid file inventory')
    for record in records:
        t = obj['payloads'][record['path']]
        if not isinstance(t, torch.Tensor) or t.dtype != torch.uint8 or t.ndim != 1 or not t.is_contiguous() or t.numel() != record['bytes']:
            raise ValueError('Invalid byte tensor: ' + record['path'])
        if hashlib.sha256(memoryview(t.numpy())).hexdigest() != record['sha256']:
            raise ValueError('Embedded file mismatch: ' + record['path'])
    manifest_bytes = obj['payloads']['MANIFEST.sha256.json'].numpy().tobytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
        raise ValueError('Embedded manifest mismatch')
    inventory = {r['path']: (r['bytes'], r['sha256']) for r in json.loads(manifest_bytes)['files']}
    if inventory != {r['path']: (r['bytes'], r['sha256']) for r in records if r['path'] != 'MANIFEST.sha256.json'}:
        raise ValueError('Manifest and payload inventory disagree')
    print(json.dumps(dict(checkpoint='verified', files=len(records), bytes=path.stat().st_size,
                          source_manifest_sha256=obj['source_manifest_sha256'])), flush=True)
    return obj


def materialize(obj, root):
    marker = root / '.fbp83_runtime.json'
    identity = dict(format=FORMAT, manifest_sha256=MANIFEST_SHA256)
    with runtime_lock(root):
        if marker.exists():
            if json.loads(marker.read_text()) != identity:
                raise ValueError('Runtime directory belongs to a different checkpoint')
        elif any(p.name != '.fbp83.lock' for p in root.iterdir()):
            raise ValueError('Runtime directory must be empty or an existing verified FBP83 runtime')
        else:
            marker.write_text(json.dumps(identity))
        reused = 0
        for record in obj['files']:
            path = checked_path(root, record['path'])
            if path.is_file() and path.stat().st_size == record['bytes'] and digest(path) == record['sha256']:
                reused += 1
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
            try:
                with temporary.open('xb') as f:
                    data = memoryview(obj['payloads'][record['path']].numpy())
                    for lo in range(0, len(data), 8 << 20):
                        f.write(data[lo:lo + (8 << 20)])
                if digest(temporary) != record['sha256']:
                    raise ValueError('Extracted file mismatch: ' + record['path'])
                temporary.replace(path)
            finally:
                if temporary.exists():
                    temporary.unlink()
        print(json.dumps(dict(runtime='verified', directory=str(root), reused_files=reused,
                              total_files=len(obj['files']))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', type=Path, default=Path(__file__).resolve().with_name('FBP_QS_OREF_CEPR.pth'))
    p.add_argument('--data-root', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--gpu', default='0')
    p.add_argument('--expert-gpus', default='')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--runtime-dir', type=Path, help='Optional runtime cache; defaults to OUTPUT/.fbp83_runtime/<manifest id>')
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--verify-only', action='store_true', help='Verify the entire .pth without extracting or using a GPU')
    modes.add_argument('--unpack-only', action='store_true', help='Extract and verify the embedded runtime without inference')
    modes.add_argument('--evaluate', action='store_true', help='Score existing outputs separately; this mode alone reads labels')
    p.add_argument('--labels-root', type=Path)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--scene-limit', type=int, default=0)
    a = p.parse_args()
    if a.workers < 1 or not 0 <= a.scene_limit <= 30:
        p.error('workers must be positive; scene-limit must be between 0 and 30')
    if a.scene_limit and not a.prepare_only:
        p.error('--scene-limit requires --prepare-only')
    if a.labels_root and not a.evaluate:
        p.error('--labels-root is only accepted in the separate --evaluate mode')
    if a.evaluate and (not a.labels_root or not a.output or a.prepare_only or a.scene_limit):
        p.error('--evaluate requires --output and --labels-root, without preparation options')
    if not (a.verify_only or a.unpack_only or a.evaluate) and (not a.data_root or not a.output):
        p.error('inference requires --data-root and --output')
    if not a.verify_only and not a.runtime_dir and not a.output:
        p.error('provide --runtime-dir or --output for extraction')
    obj = load_checkpoint(a.model.resolve())
    if a.verify_only:
        return
    runtime = (a.runtime_dir or (a.output / '.fbp83_runtime' / MANIFEST_SHA256[:16])).resolve()
    materialize(obj, runtime)
    del obj
    if a.unpack_only:
        return
    subprocess.run([sys.executable, str(runtime / 'verify_package.py')], check=True)
    if a.evaluate:
        command = [sys.executable, str(runtime / 'evaluate_official30.py'), '--output', str(a.output.resolve()), '--labels-root', str(a.labels_root.resolve())]
    else:
        command = [sys.executable, '-u', str(runtime / 'infer.py'), '--data-root', str(a.data_root.resolve()), '--output', str(a.output.resolve()), '--gpu', a.gpu, '--workers', str(a.workers)]
        if a.expert_gpus:
            command += ['--expert-gpus', a.expert_gpus]
        if a.prepare_only:
            command += ['--prepare-only']
        if a.scene_limit:
            command += ['--scene-limit', str(a.scene_limit)]
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
