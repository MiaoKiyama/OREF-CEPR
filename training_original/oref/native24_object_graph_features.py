"""Label-blind real-object graph caches and exact group-membership aggregation."""
import argparse
import builtins
import contextlib
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

ORDER = np.asarray([2, 4, 6, 3, 5, 0, 1])
RELATIONS = ("contact", "overlap", "continuation")
SCHEMA = "native24_real_object_graph_exact_membership_v1"
GRAPH_SETTINGS = dict(
    schema=SCHEMA, candidates=7, geometry_width=12, relation_count=3, relation_width=38,
    connectivity=4, contact="4-neighbor shared boundary length within expert",
    overlap="all 21 expert pairs; exact pixel intersection area",
    continuation="same-expert same-class distinct objects; outward boundary rays at 2,4,8 pixels along four undirected axes",
    continuation_distances=[2, 4, 8], relation_neighbor_truncation=False,
    membership="exact per-group pixel masses over every real connected object",
    relation_normalization="per-object outgoing-weight mean, then exact group pixel-weight mean",
    relation_columns="neighbor geometry12,class distribution24,log1p(outgoing weight)/log1p(7*scene pixels),degree/(degree+1)",
    shuffle_seed=261016,
    shuffle="per relation, within each expert, permute destination object IDs; preserve source, edges, weights and outdegree",
    cache="lossless compressed integer topology and float32 object geometry",
    graph_uses_Q=False, graph_uses_S=False, graph_uses_GT=False,
)
_LIB = {}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.%d" % os.getpid())
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def observable_only(paths):
    allowed = {str(Path(p).resolve()) for p in paths}
    old_open, old_io, old_path = builtins.open, io.open, Path.open
    old_member = np.lib.npyio.NpzFile.__getitem__
    members = {"keys", "labels", "start", "mass", "sar", "numeric", "fixed_prediction_counts"}

    def guard(path, mode):
        if isinstance(path, (str, bytes, os.PathLike)):
            if any(x in mode for x in ("w", "a", "+")) or str(Path(os.fsdecode(path)).resolve()) not in allowed:
                raise RuntimeError("Non-observable file access blocked: " + str(path))

    def op(path, mode="r", *a, **kw):
        guard(path, mode)
        return old_open(path, mode, *a, **kw)

    def iop(path, mode="r", *a, **kw):
        guard(path, mode)
        return old_io(path, mode, *a, **kw)

    def pop(path, mode="r", *a, **kw):
        guard(path, mode)
        return old_path(path, mode, *a, **kw)

    def member(obj, key):
        if key.removesuffix(".npy") not in members:
            raise RuntimeError("GT/unapproved NPZ member blocked: " + key)
        return old_member(obj, key)

    builtins.open, io.open, Path.open = op, iop, pop
    np.lib.npyio.NpzFile.__getitem__ = member
    try:
        yield
    finally:
        builtins.open, io.open, Path.open = old_open, old_io, old_path
        np.lib.npyio.NpzFile.__getitem__ = old_member


def scene_path(c, sid, domain):
    assert domain in ("oof", "replay")
    return Path(c["graph_store"]) / domain / sid


def scene_ready(c, sid, domain):
    root = scene_path(c, sid, domain)
    return (root / "completed.json").exists() and (root / "graph.npz").exists()


def sar_ready(c, sid, domain):
    root = scene_path(c, sid, domain)
    if not (root / "sar_completed.json").exists():
        return False
    return domain != "replay" or (root / "sar.npy").exists()


def descriptor(c, sid, domain):
    source = read(Path(c["source"]) / "config.json")
    assert sid in source["official_ids" if domain == "official" else "train_ids"]
    if domain == "replay":
        destinations = read(Path(c["replay_root"]) / "destinations.json")
        folder = Path(destinations[sid])
        manifest = read(folder / "compact_manifest.json")
        assert manifest["action_indices"] == ORDER.tolist()
        assert manifest["grouping_uses_gt"] is False and manifest["mass_uses_gt"] is False
        ix = read(Path(c["replay_store"]) / "index.json")
        row = next(r for r in ix["rows"] if r["scene"] == sid)
        return dict(
            shape=manifest["shape"], key_path=folder / manifest["groups"]["file"],
            labels_path=Path(row["files"]["labels"]), start=0, end=row["end"] - row["start"],
            mass_path=Path(row["files"]["mass"]), observable_path=None,
            numeric_path=None, has_sar=bool(manifest["has_sar"]),
            chunks=sorted((folder / "predictions").glob("chunk_*.npz")),
            metadata="replay destinations and observable compact manifest; no score/target receipt")
    bank = read(Path(source["controlled"]) / "config.json") if domain == "official" else source
    ix = read(Path(bank["store"]) / "index.json")
    row = next(r for r in ix["rows"] if r["scene"] == sid)
    if domain == "official":
        folder = Path(next(r["source"] for r in bank["scene_inventory"]["official"] if r["scene"] == sid))
        key = Path(bank["official_aligned"]) / "scenes" / (sid + ".npz")
    else:
        folder, key = Path(row["source"]), Path(row["source"]) / "groups.npz"
    obs_path = Path(bank["output"]) / "observable/oof" / (sid + ".npz")
    obs = read(obs_path.with_suffix(".json"))
    assert obs["ground_truth_members_read"] is False
    return dict(
        shape=obs["shape"], key_path=key, labels_path=Path(bank["store"]) / "labels.npy",
        start=row["start"], end=row["end"], mass_path=None, observable_path=obs_path,
        numeric_path=Path(bank["store"]) / "numeric.npy",
        chunks=sorted((folder / "compact").glob("chunk_*.npz")),
        metadata="source config/index and observable scene receipt; no target receipt")


def sar_scene(c, sid, domain):
    """Read a sealed SAR sidecar only; call prepare_sar explicitly to create it."""
    root = scene_path(c, sid, domain)
    r = read(root / "sar_completed.json")
    assert r["status"] == "sealed" and r["ground_truth_read"] is False
    assert r["scene"] == sid and r["domain"] == domain
    if domain == "replay":
        data = np.load(root / "sar.npy", mmap_mode="r", allow_pickle=False)
    else:
        with observable_only([r["numeric_path"]]):
            data = np.load(r["numeric_path"], mmap_mode="r", allow_pickle=False)[r["start"]:r["end"], 119:126]
    assert data.shape == tuple(r["shape"]) and data.dtype == np.float32
    assert hashlib.sha256(data.tobytes()).hexdigest() == r["sha256"], "Sealed SAR payload changed"
    return data


def prepare_sar(c, sid, domain):
    import fcntl
    root = scene_path(c, sid, domain)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "sar.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if sar_ready(c, sid, domain):
            return sar_scene(c, sid, domain)
        d = descriptor(c, sid, domain)
        if domain == "replay":
            with observable_only([d["key_path"]]):
                with np.load(d["key_path"], allow_pickle=False) as z:
                    values = z["sar"].astype(np.float32)
            assert values.shape == (d["end"], 6)
            data = np.c_[values, np.full(len(values), float(d["has_sar"]), np.float32)]
        else:
            with observable_only([d["numeric_path"]]):
                source = np.load(d["numeric_path"], mmap_mode="r", allow_pickle=False)
                data = source[d["start"]:d["end"], 119:126]
        assert data.shape[1] == 7 and np.isfinite(data).all()
        if domain == "replay":
            tmp = root / ("sar.tmp.%d.npy" % os.getpid())
            np.save(tmp, data, allow_pickle=False)
            os.replace(tmp, root / "sar.npy")
        write(root / "sar_completed.json", dict(
            status="sealed", scene=sid, domain=domain, shape=list(data.shape),
            dtype="float32", sha256=hashlib.sha256(data.tobytes()).hexdigest(), ground_truth_read=False,
            numeric_path=str(d["numeric_path"]) if domain != "replay" else None,
            start=d["start"], end=d["end"],
            source="original groups.sar + compact has_sar" if domain == "replay" else "existing numeric119:126",
            numpy_target_members_blocked=True, time_unix=time.time()))
        return sar_scene(c, sid, domain)


def pixel_keys(labels, positions, shape):
    h, w = shape
    y, x = positions // w, positions % w
    roi = np.searchsorted(np.linspace(0, h, 5, dtype=int)[1:], y, side="right") * 8
    roi += np.searchsorted(np.linspace(0, w, 9, dtype=int)[1:], x, side="right")
    return ((y // 64) * ((w + 63) // 64) + x // 64) * 33 * 25**7 + roi * 25**7 + (
        labels.astype(np.int64) @ (25**np.arange(6, -1, -1)))


def load_hard(d):
    allowed = [d["key_path"], d["labels_path"]] + d["chunks"]
    allowed += [p for p in (d["mass_path"], d["observable_path"]) if p is not None]
    h, w = d["shape"]
    with observable_only(allowed):
        with np.load(d["key_path"], allow_pickle=False) as z:
            keys = z["keys"]
        labs = np.load(d["labels_path"], mmap_mode="r", allow_pickle=False)[d["start"]:d["end"]]
        if d["mass_path"] is not None:
            mass = np.load(d["mass_path"], mmap_mode="r", allow_pickle=False)
        else:
            with np.load(d["observable_path"], allow_pickle=False) as z:
                mass = z["mass"]
        n = len(keys)
        assert labs.shape == (n, 7) and len(mass) == n and np.all(np.diff(keys) > 0)
        hard = np.empty((7, h * w), np.uint8)
        mapping = np.full(h * w, -1, np.int32)
        observed = np.zeros(n, np.uint64)
        digest = hashlib.sha256()
        cursor = 0
        for p in d["chunks"]:
            with np.load(p, allow_pickle=False) as z:
                start, chunk = int(z["start"]), z["labels"]
            assert start == cursor and chunk.shape[0] == 7
            stop = start + chunk.shape[1]
            assert stop <= h * w
            hard[:, start:stop] = chunk
            active = np.flatnonzero(np.any(chunk != chunk[6], axis=0))
            actions = chunk[ORDER][:, active].T
            k = pixel_keys(actions, start + active, (h, w))
            ids = np.searchsorted(keys, k)
            assert np.all(ids < n)
            np.testing.assert_array_equal(keys[ids], k)
            np.testing.assert_array_equal(labs[ids], actions)
            mapping[start + active] = ids
            observed += np.bincount(ids, minlength=n).astype(np.uint64)
            digest.update(np.asarray([start], dtype="<i8").tobytes())
            digest.update(chunk.tobytes())
            cursor = stop
        assert cursor == h * w
        np.testing.assert_array_equal(observed, mass)
    return hard.reshape(7, h, w), mapping.reshape(h, w), n, dict(
        pixels=h * w, groups=n, observed_group_pixels=int(observed.sum()),
        labels_and_group_mass_exact=True, input_sha256=digest.hexdigest(), ground_truth_read=False)


def library(c):
    cpp = Path(__file__).with_suffix(".cpp")
    digest = sha(cpp)
    if digest not in _LIB:
        import fcntl
        build = Path(c["graph_store"]) / "_build"
        build.mkdir(parents=True, exist_ok=True)
        so = build / ("graph_" + digest[:16] + ".so")
        with (build / "compile.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not so.exists():
                tmp = so.with_name(so.name + ".tmp.%d" % os.getpid())
                subprocess.run(["g++", "-O3", "-std=c++17", "-fPIC", "-shared", str(cpp), "-o", str(tmp)], check=True)
                os.replace(tmp, so)
        lib = ctypes.CDLL(str(so))
        lib.build_graph.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int32),
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
                                    ctypes.c_char_p, ctypes.c_int]
        lib.build_graph.restype = ctypes.c_int
        _LIB[digest] = lib
    return _LIB[digest]


def graph_from_arrays(hard_raw, groups, n, c, temporary_path):
    """C++ output is deterministic and all topology weights remain integers."""
    hard_raw = np.ascontiguousarray(hard_raw, np.uint8)
    groups = np.ascontiguousarray(groups, np.int32)
    assert hard_raw.shape == (7,) + groups.shape
    error = ctypes.create_string_buffer(4096)
    result = library(c).build_graph(
        hard_raw.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        groups.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        groups.shape[0], groups.shape[1], n, str(temporary_path).encode(), error, len(error))
    if result:
        raise RuntimeError(error.value.decode())
    with Path(temporary_path).open("rb") as f:
        nodes, e0, e1, e2, extras = map(int, np.fromfile(f, "<u8", count=5))
        data = dict(
            geometry=np.fromfile(f, "<f4", count=nodes * 12).reshape(nodes, 12),
            labels=np.fromfile(f, "u1", count=nodes),
            expert=np.fromfile(f, "u1", count=nodes),
            area=np.fromfile(f, "<u4", count=nodes),
            primary=np.fromfile(f, "<u4", count=n * 7).reshape(n, 7),
            mass=np.fromfile(f, "<u4", count=n),
            extra=np.fromfile(f, "<u4", count=extras * 3).reshape(extras, 3))
        for name, count in zip(RELATIONS, (e0, e1, e2)):
            data[name] = np.fromfile(f, "<u4", count=count * 3).reshape(count, 3)
        assert f.read() == b""
    Path(temporary_path).unlink()
    assert np.isfinite(data["geometry"]).all()
    return data


def feature_scene(c, sid, domain):
    import fcntl
    root = scene_path(c, sid, domain)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "graph.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if scene_ready(c, sid, domain):
            return read(root / "completed.json")
        started = time.time()
        if shutil.disk_usage(root).free < 3.2 * 2**30:
            raise RuntimeError("Graph preparation requires 3 GiB remaining disk plus working space")
        prepare_sar(c, sid, domain)
        hard, groups, n, audit = load_hard(descriptor(c, sid, domain))
        data = graph_from_arrays(hard, groups, n, c, root / ("graph.tmp.%d.bin" % os.getpid()))
        del hard, groups
        tmp = root / ("graph.tmp.%d.npz" % os.getpid())
        np.savez_compressed(tmp, **data)
        os.replace(tmp, root / "graph.npz")
        import resource
        result = dict(
            status="sealed", schema=SCHEMA, scene=sid, domain=domain, groups=n,
            nodes=len(data["geometry"]), edges={r: len(data[r]) for r in RELATIONS},
            group_extra_memberships=len(data["extra"]), graph_settings=GRAPH_SETTINGS,
            graph_sha256=sha(root / "graph.npz"), bytes=(root / "graph.npz").stat().st_size,
            ground_truth_read=False, audit=audit, python_sha256=sha(__file__),
            cpp_sha256=sha(Path(__file__).with_suffix(".cpp")),
            elapsed_seconds=time.time() - started,
            peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            time_unix=time.time())
        write(root / "completed.json", result)
        return result


class SceneGraph:
    """Exact real-object convolution sufficient statistics, never group-as-object edges."""
    def __init__(self, c, sid, domain):
        root = scene_path(c, sid, domain)
        self.receipt = read(root / "completed.json")
        assert self.receipt["status"] == "sealed" and self.receipt["schema"] == SCHEMA
        assert self.receipt["ground_truth_read"] is False
        assert sha(root / "graph.npz") == self.receipt["graph_sha256"]
        with np.load(root / "graph.npz", allow_pickle=False) as z:
            self.data = {key: z[key] for key in z.files}
        self.n = len(self.data["primary"])
        self.sid = sid
        self._relations = {}
        extra = self.data["extra"]
        self.extra_rows = extra[:, 0]

    def _node_relations(self, shuffled):
        if shuffled:
            raise ValueError("Only the released full graph is supported")
        if shuffled in self._relations:
            return self._relations[shuffled]
        d = self.data
        nodes = len(d["geometry"])
        features = np.concatenate((d["geometry"], np.eye(24, dtype=np.float32)[d["labels"] - 1]), axis=1)
        packed = np.zeros((nodes, 129), np.float32)
        packed[:, :12] = d["geometry"]
        result = packed[:, 12:126].reshape(nodes, 3, 38)
        valid = packed[:, 126:129]
        pixels = self.receipt["audit"]["pixels"]
        for r, name in enumerate(RELATIONS):
            edge = d[name]
            # Undirected stored once; adjacency includes both directions.
            source = np.r_[edge[:, 0], edge[:, 1]]
            dest = np.r_[edge[:, 1], edge[:, 0]].copy()
            weight = np.r_[edge[:, 2], edge[:, 2]].astype(np.float64)
            den = np.bincount(source, weights=weight, minlength=nodes)
            degree = np.bincount(source, minlength=nodes)
            valid[:, r] = den > 0
            for col in range(36):
                result[:, r, col] = np.bincount(
                    source, weights=weight * features[dest, col], minlength=nodes) / np.maximum(den, 1)
            result[:, r, 36] = np.log1p(den) / np.log1p(pixels * 7)
            result[:, r, 37] = degree / (degree + 1.)
        self._relations[shuffled] = packed
        return packed

    def _aggregate(self, values, indices):
        d = self.data
        ix = np.asarray(indices, np.int64)
        primary = d["primary"][ix].reshape(-1)
        rows = (ix[:, None] * 7 + np.arange(7)).reshape(-1)
        mass = np.repeat(d["mass"][ix], 7).astype(np.float32)
        out = values[primary].astype(np.float32, copy=False)
        starts = np.searchsorted(self.extra_rows, rows, side="left")
        stops = np.searchsorted(self.extra_rows, rows + 1, side="left")
        lengths = stops - starts
        if lengths.sum():
            repeat = np.repeat(np.arange(len(rows)), lengths)
            offsets = np.repeat(starts - np.r_[0, np.cumsum(lengths)[:-1]], lengths) + np.arange(lengths.sum())
            extra = d["extra"][offsets]
            weight = extra[:, 2].astype(np.float32) / mass[repeat]
            shape = (len(weight),) + (1,) * (values.ndim - 1)
            adjustment = (values[extra[:, 1]] - values[primary[repeat]]) * weight.reshape(shape)
            selected = np.flatnonzero(lengths)
            boundaries = np.r_[0, np.cumsum(lengths[selected])[:-1]]
            out[selected] += np.add.reduceat(adjustment, boundaries, axis=0)
        return out.reshape((len(ix), 7) + values.shape[1:])

    def get(self, indices, structure="full"):
        if structure != "full":
            raise ValueError("Only the released full graph is supported")
        ix = np.asarray(indices, np.int64)
        packed = self._aggregate(self._node_relations(False), ix)
        geometry = packed[:, :, :12]
        relations = packed[:, :, 12:126].reshape(len(ix), 7, 3, 38)
        mask = packed[:, :, 126:129]
        return dict(geometry=geometry, relations=relations, relation_mask=mask)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--domain", required=True, choices=("oof", "replay"))
    p.add_argument("--scene", required=True)
    p.add_argument("--sar-only", action="store_true")
    a = p.parse_args()
    c = read(a.config)
    result = prepare_sar(c, a.scene, a.domain) if a.sar_only else feature_scene(c, a.scene, a.domain)
    print(json.dumps(dict(status="sealed", shape=list(result.shape)) if a.sar_only else result))


if __name__ == "__main__":
    main()
