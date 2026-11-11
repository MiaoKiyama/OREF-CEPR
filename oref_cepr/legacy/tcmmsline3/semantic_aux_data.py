"""SAR/PAN geometry from the audited reader, without clipping SAR dB values.

The existing probability-refiner reader is kept unchanged.  This subclass
captures its native sampler's unnormalised values during the SAME read, so
registration, pixel-centre conventions and conservative validity are identical.
"""
from __future__ import annotations

import numpy as np
from .dual_aux_data import AuxiliaryScene


class SemanticAuxScene(AuxiliaryScene):
    sar_value_scale = "Sigma0_dB"

    def _sample(self, modality, x, y):
        result = super()._sample(modality, x, y)
        self._semantic_native_samples = result[0]
        return result

    def read_patch(self, top, left, size, *, modality, out_size=None):
        self._semantic_native_samples = None
        result = super().read_patch(top, left, size, modality=modality, out_size=out_size)
        if modality == "sar":
            if result["available"]:
                n = result["aux"].shape[-1]
                values = self._semantic_native_samples[:, :n*n].reshape(2, n, n).copy()
                values[:, ~result["native_valid"][0]] = 0
                if not np.isfinite(values).all():
                    raise ValueError("nonfinite native SAR after validity masking")
                result["aux"] = np.ascontiguousarray(values, dtype=np.float32)
            result["quality"]["normalization"] = "none_raw_Sigma0_dB_invalid_zero"
            result["quality"]["sar_value_scale"] = self.sar_value_scale
        self._semantic_native_samples = None
        return result
