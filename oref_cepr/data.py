"""Image IO helpers for diagnostic prediction; training uses original samplers."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .bootstrap import ROOT

Image.MAX_IMAGE_PIXELS = None
MEAN = np.array([.363, .390, .371, .506], np.float32)
STD = np.array([.231, .231, .244, .254], np.float32)
RARE = (7, 9, 15, 18, 19, 22, 23)
HARD = (0, 4, 5, 6, 8, 10, 11, 12, 13, 14, 17, 20, 21)


def splits():
    return json.loads((ROOT / 'configs/fbp/splits.json').read_text())


def scene_ids(fold=None):
    s = splits()
    return s['train_ids'] if fold is None else s['folds'][fold]['train_ids']


def normalize(a):
    return ((a.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy()


def read_image(path):
    with Image.open(path) as im:
        a = np.asarray(im)
    if a.ndim != 3 or a.shape[2] != 4 or a.dtype != np.uint8:
        raise ValueError(f'Expected uint8 HxWx4 NIR,R,G,B TIFF: {path}')
    return np.ascontiguousarray(a[..., ::-1])


def label_path(folder, sid):
    for suffix in ('.png', '.tif', '.tiff'):
        path = Path(folder) / (sid + suffix)
        if path.is_file():
            return path
    raise FileNotFoundError(Path(folder) / (sid + '.png'))
