"""Read guard for the released training bank; no experiment runner."""
import os
import io
import builtins
import contextlib
from pathlib import Path
import numpy as np
from native24_original82_data import read


@contextlib.contextmanager
def official_truth_forbidden(c):
    """Permit train120 labels while rejecting target truth during fitting."""
    source = read(Path(c["source"]) / "config.json")
    allowed_store = str(Path(source["store"]).resolve()).replace("\\", "/").rstrip("/")
    original_load, original_open, original_io = np.load, builtins.open, io.open
    original_path_open = Path.open

    def check(path):
        if not isinstance(path, (str, bytes, os.PathLike)):
            return
        value = str(Path(os.fsdecode(path)).resolve()).replace("\\", "/")
        allowed = value.startswith(allowed_store + "/")
        forbidden = (
            value.endswith("/counts.npy")
            or "/targets/" in value
            or ("/scenes/" in value and value.endswith("/completed.json"))
            or value.endswith("/official_completed.npz")
            or value.endswith("/official_completed.json")
        )
        assert not forbidden or allowed, "Official/unapproved truth read during train120 fitting: " + value

    def load(path, *args, **kwargs):
        check(path)
        return original_load(path, *args, **kwargs)

    def open_file(path, *args, **kwargs):
        check(path)
        return original_open(path, *args, **kwargs)

    def open_io(path, *args, **kwargs):
        check(path)
        return original_io(path, *args, **kwargs)

    def open_path(path, *args, **kwargs):
        # Python 3.10 pathlib keeps a cached accessor, so replacing io.open
        # alone does not intercept Path.read_text/read_bytes.
        check(path)
        return original_path_open(path, *args, **kwargs)

    np.load, builtins.open, io.open = load, open_file, open_io
    Path.open = open_path
    try:
        yield
    finally:
        np.load, builtins.open, io.open = original_load, original_open, original_io
        Path.open = original_path_open
