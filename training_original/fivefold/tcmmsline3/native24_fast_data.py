"""Local-only ConvNeXt inputs; identical sampling, augmentation and joint SAR.

The legacy dataset remains available as the reference implementation. Warmup
omits SAR IO because every SAR branch is disabled by the model in that stage.
"""
from __future__ import annotations

import random
import numpy as np
import torch

from .ms75_sar_joint_data import MS75SARJointDataset as Full120Dataset24
from .ms75_rebuild_data import read_cached_scene
from .light_sar_data import _checked_aux, _load_points, apply_flips, open_sar_scene
from .target_dino_data import SPECTRAL_AUGMENTATION, normalize_bgrn


def augment_local(ms, valid, *, rng, augmentation, normalization):
    """Exact MS75 local branch arithmetic and RNG draws, without other views."""
    aug = augmentation
    if not aug.get("enabled", False) or rng.random() >= float(aug.get("probability", 0)):
        return ms
    norm = normalization.get("ms", normalization)
    mean, std = np.asarray(norm["mean"], np.float32)[:, None, None], np.asarray(norm["std"], np.float32)[:, None, None]
    gain = rng.uniform(*aug.get("global_gain_range", [1, 1]))
    offset = rng.uniform(*aug.get("global_offset_range", [0, 0]))
    band_gain = np.array([rng.uniform(*aug.get("band_gain_range", [1, 1])) for _ in range(4)], np.float32)[:, None, None]
    band_offset = np.array([rng.uniform(*aug.get("band_offset_range", [0, 0])) for _ in range(4)], np.float32)[:, None, None]
    value = (np.clip((ms * std + mean) * gain * band_gain + offset + band_offset, 0, 1.5) - mean) / std
    return np.where(valid[None], value, ms).astype(np.float32, copy=False)


class FastFull120Dataset24(Full120Dataset24):
    def __init__(self, config, epoch=0, start_sample=0, *, load_sar=True):
        super().__init__(config, epoch, start_sample)
        self.load_sar = bool(load_sar)

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= int(index) < len(self):
            raise IndexError(index)
        absolute = self.start_sample + int(index)
        scene, kind, cls = self.blocks[absolute // 16]
        rng = random.Random(self.seed + self.epoch * 10000019 + absolute * 65537)
        image, labels = self._bounded(self._cache, scene, lambda: read_cached_scene(self.metadata, scene))
        h, w = labels.shape
        if min(h, w) < self.n:
            raise ValueError("Scene smaller than optical crop")
        redraw_count, fallback_used = 0, False
        if kind == "scene":
            for attempt in range(16):
                top, left = rng.randrange(h - self.n + 1), rng.randrange(w - self.n + 1)
                if np.any(labels[top:top + self.n, left:left + self.n] < 24):
                    break
                redraw_count += 1
            else:
                row = self.metadata["scenes"][scene]
                reservoirs = self._bounded(self._points, scene, lambda: _load_points(row["points"], row["points_sha256"]))
                eligible = [c for c, points in reservoirs.items() if len(points)]
                if not eligible:
                    raise ValueError("Training scene has no known optical labels")
                points = reservoirs[rng.choice(eligible)]
                cy, cx = map(int, points[rng.randrange(len(points))])
                top, left = int(np.clip(cy - self.n // 2, 0, h - self.n)), int(np.clip(cx - self.n // 2, 0, w - self.n))
                fallback_used = True
        else:
            if kind == "boundary":
                points = self._boundary(scene, labels)
            else:
                row = self.metadata["scenes"][scene]
                points = self._bounded(self._points, scene, lambda: _load_points(row["points"], row["points_sha256"]))[cls]
            if not len(points):
                raise ValueError("Chosen target scene has no target points")
            cy, cx = map(int, points[rng.randrange(len(points))])
            jitter = self.n // 4
            top = int(np.clip(cy - self.n // 2 + rng.randint(-jitter, jitter), 0, h - self.n))
            left = int(np.clip(cx - self.n // 2 + rng.randint(-jitter, jitter), 0, w - self.n))

        # ConvNeXt uses only this local512 optical input. No1536->512 short
        # resize, global thumbnail, unused augmentations or extra IPC tensors.
        ms = normalize_bgrn(image[top:top + self.n, left:left + self.n], self.metadata["normalization"])
        target = np.array(labels[top:top + self.n, left:left + self.n], dtype=np.int64, copy=True)
        valid = np.ones((self.n, self.n), bool)
        known_pixels = int(np.count_nonzero(target < 24))
        if known_pixels < 1:
            raise ValueError("Crop must retain at least one known optical GT pixel")
        if self.load_sar and self.availability[scene]:
            reader = self._bounded(self._aux, scene, lambda: open_sar_scene(self.data_root, scene))
            sar, native, optical = _checked_aux(reader.read_patch(top, left, self.n, modality="sar", out_size=self.sn), self.n, self.sn)
        else:
            sar, native, optical = np.zeros((2, self.sn, self.sn), np.float32), np.zeros((1, self.sn, self.sn), bool), np.zeros((1, self.n, self.n), bool)
        code = int(rng.random() < .5) + 2 * int(rng.random() < .5) if self.augmentation else 0
        ms, valid = apply_flips(ms, code), apply_flips(valid, code)
        if self.augmentation:
            ms = augment_local(ms, valid, rng=rng, augmentation=SPECTRAL_AUGMENTATION, normalization=self.metadata["normalization"])
        arrays = dict(ms=ms, ms_valid=valid, sar_db=apply_flips(sar, code),
            sar_valid=apply_flips(native, code), optical_valid=apply_flips(optical, code), label=apply_flips(target, code))
        result = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in arrays.items()}
        # Cached labels are already semantic0..23/ignore255; never supervise raw0.
        result.update(index=absolute, scene_id=scene, top=top, left=left, sampling_kind=kind,
            sampling_class=cls, flip_code=code, has_sar=self.availability[scene],
            known_pixels=known_pixels, redraw_count=redraw_count, fallback_used=fallback_used,
            ignored_pixels=int((result["label"] == 255).sum()), warmup_sar_skipped=not self.load_sar)
        return result
