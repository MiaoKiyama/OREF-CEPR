import ctypes,hashlib,os,subprocess
from pathlib import Path
import numpy as np
from common import sha
_LIB={}
RELATIONS=("contact","overlap","continuation")
SCHEMA="native24_real_object_graph_exact_membership_v1"
def pixel_keys(labels, positions, shape):
    h, w = shape
    y, x = positions // w, positions % w
    roi = np.searchsorted(np.linspace(0, h, 5, dtype=int)[1:], y, side="right") * 8
    roi += np.searchsorted(np.linspace(0, w, 9, dtype=int)[1:], x, side="right")
    return ((y // 64) * ((w + 63) // 64) + x // 64) * 33 * 25**7 + roi * 25**7 + (
        labels.astype(np.int64) @ (25**np.arange(6, -1, -1)))

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
