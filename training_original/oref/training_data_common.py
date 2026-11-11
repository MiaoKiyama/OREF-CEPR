"""Confusion-matrix summary shared by the two training banks."""
import numpy as np


def miou_percent(cm):
    a = np.asarray(cm, dtype=np.float64)[1:, 1:]
    d = a.diagonal()
    return float(100 * (d / np.maximum(a.sum(0) + a.sum(1) - d, 1)).mean())
