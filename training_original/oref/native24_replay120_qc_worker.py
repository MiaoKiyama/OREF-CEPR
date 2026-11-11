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
    """Only explicitly listed prepared replay targets may be read while fitting."""
    from native24_replay120_qc_data import allowed_training_target_paths
    allowed_targets = {
        str(Path(path).resolve()).replace("\\", "/")
        for path in allowed_training_target_paths(c)
    }
    assert allowed_targets, "Prepared replay target whitelist is empty."
    original_load, original_open, original_io = np.load, builtins.open, io.open
    original_path_open = Path.open

    def check(path):
        if not isinstance(path, (str, bytes, os.PathLike)):
            return
        value = str(Path(os.fsdecode(path)).resolve()).replace("\\", "/")
        allowed = value in allowed_targets
        forbidden = (
            value.endswith("/counts.npy")
            or value.endswith("/meta.json")
            or "/targets/" in value
            or ("/scenes/" in value and value.endswith("/completed.json"))
            or value.endswith("/official_completed.npz")
            or value.endswith("/official_completed.json")
        )
        assert not forbidden or allowed, "Non-replay truth read during replay120 fitting: " + value

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
        check(path)
        return original_path_open(path, *args, **kwargs)

    np.load, builtins.open, io.open, Path.open = load, open_file, open_io, open_path
    try:
        yield
    finally:
        np.load, builtins.open, io.open, Path.open = original_load, original_open, original_io, original_path_open
