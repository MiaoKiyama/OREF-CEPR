"""Shared runtime, draw audit and scoring primitives for train120 fitting."""
import os
import ctypes
import hashlib
from pathlib import Path
import numpy as np
from native24_original82_data import read, write


def available_gib():
    return next(
        int(line.split()[1]) / 2**20
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )


def stamp(out, stage, **extra):
    rss = next(
        int(line.split()[1]) / 2**20
        for line in Path("/proc/self/status").read_text().splitlines()
        if line.startswith("VmRSS:")
    )
    write(Path(out) / "progress.json", dict(
        stage=stage, pid=os.getpid(), rss_gib=rss, time_unix=time.time(), **extra
    ))


def configure(c, gpu, cpu_slot):
    resources = c.get("resources", {})
    gpu_config = resources.get(str(gpu), {})
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_config.get("uuid", gpu))
    cpu_config = resources.get(str(cpu_slot), {})
    cpus = cpu_config.get("cpus", [])
    if cpus:
        os.sched_setaffinity(0, set(map(int, cpus)))
    os.nice(5)
    try:
        ctypes.CDLL(None).prctl(41, 1, 0, 0, 0)
    except (AttributeError, OSError):
        pass
    import torch
    threads = int(c.get("cpu_threads", 4))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    total = torch.cuda.get_device_properties(0).total_memory
    cap = float(c.get("gpu_memory_cap_gib", 3)) * 2**30
    torch.cuda.set_per_process_memory_fraction(min(cap / total, 1.0))
    os.environ["ORIGINAL82_CPU_THREADS"] = str(threads)
    import native24_original82_fast as fast
    fast.install()
    return torch


def parameter_hash(model, parameters_only=False):
    digest = hashlib.sha256()
    items = model.named_parameters() if parameters_only else model.state_dict().items()
    for name, value in items:
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def correction_stats(counts, labels, route):
    base = labels[:, 2]
    pred = labels[np.arange(len(labels)), route]
    changed = base != pred
    rows = np.arange(len(labels))
    old_correct = counts[rows, base].astype(np.int64)
    new_correct = counts[rows, pred].astype(np.int64)
    valid = counts[:, 1:].sum(1, dtype=np.int64)
    result = []
    for classes, weights in (
        (base, old_correct), (pred, new_correct),
        (pred, valid - new_correct), (base, valid - old_correct),
    ):
        result.append(np.bincount(
            classes[changed], weights=weights[changed], minlength=25
        ).astype(np.int64))
    return np.stack(result)


def raw_posterior_route(p, labels, block=131072):
    route = np.empty(len(labels), np.uint8)
    for start in range(0, len(labels), block):
        end = min(start + block, len(labels))
        values = p[start:end][np.arange(end - start)[:, None], labels[start:end] - 1]
        route[start:end] = values.argmax(1).astype(np.uint8)
    return route


class DrawAudit:
    def __init__(self, bank):
        self.total = hashlib.sha256()
        self.interval = hashlib.sha256()
        self.total_batches = 0
        self.interval_batches = 0
        self.ends = np.asarray([row["local_end"] for row in bank.rows], np.int64)
        self.scene_draws = np.zeros(len(bank.rows), np.int64)
        self.interval_scene_draws = np.zeros(len(bank.rows), np.int64)
        self.domains = np.asarray([row["domain"] for row in bank.rows])
        self.scene_ids = [row["scene"] for row in bank.rows]

    def update(self, indices):
        indices = np.asarray(indices, dtype="<i8")
        encoded = indices.tobytes()
        self.total.update(encoded)
        self.interval.update(encoded)
        self.total_batches += 1
        self.interval_batches += 1
        which = np.searchsorted(self.ends, indices, side="right")
        counts = np.bincount(which, minlength=len(self.ends))
        self.scene_draws += counts
        self.interval_scene_draws += counts

    def snapshot(self):
        return dict(
            total_updates=self.total_batches, total_draws=int(self.scene_draws.sum()),
            cumulative_indices_sha256=self.total.hexdigest(),
            checkpoint_interval_updates=self.interval_batches,
            checkpoint_interval_indices_sha256=self.interval.hexdigest(),
            pool_draws={name: int(self.scene_draws[self.domains == name].sum())
                        for name in ("train120", "official30")},
            checkpoint_interval_pool_draws={
                name: int(self.interval_scene_draws[self.domains == name].sum())
                for name in ("train120", "official30")
            },
            per_scene_draws=dict(zip(self.scene_ids, self.scene_draws.tolist())),
            dedicated_cpu_generator=True,
        )

    def reset_interval(self):
        self.interval = hashlib.sha256()
        self.interval_batches = 0
        self.interval_scene_draws.fill(0)
