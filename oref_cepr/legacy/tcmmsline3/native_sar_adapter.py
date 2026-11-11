"""Frozen native D2 with zero-initialized, multiscale SAR feature adapters.

No pretrained optical layer is replaced.  Optical crops follow the original
512 -> reflect-padded 518 -> original four DPT scales -> cropped 512 path.
Only the new SAR projection and four conditional residual adapters are learned.
The CROMA spatial tokens are NOT a final-probability/logit correction.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .semantic_aux_model import (
    CROMA_CHECKPOINT_BYTES, CROMA_CHECKPOINT_SHA256, CROMA_COMMIT,
    CROMA_SOURCE_SHA256, _file_sha256, normalize_croma_sar,
)


def _groups(channels: int) -> int:
    return next(g for g in (8, 4, 2, 1) if channels % g == 0)


def _block(cin: int, cout: int, kernel: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, padding=kernel // 2, bias=False),
                         nn.GroupNorm(_groups(cout), cout), nn.GELU())


class NativeCromaSAR(nn.Module):
    """Authenticated, frozen CROMA S1 encoder on a native SAR window.

    166 SAR samples span the same extent as the 512 MS pixels. They are padded
    to 168 for patch8. Output remains the original 21x21 token lattice; sampling
    into each *padded* optical DPT scale happens later, with explicit coordinates.
    Fully invalid keys are suppressed and invalid query tokens are zeroed.
    """
    def __init__(self, checkpoint: str | Path, source: str | Path | None = None,
                 sar_size: int = 166, *, load_pretrained: bool = True) -> None:
        super().__init__()
        if type(sar_size) is not int or not 8 <= sar_size <= 512:
            raise ValueError('sar_size must be an integer in 8..512')
        self.sar_size = sar_size
        self.padded_size = math.ceil(sar_size / 8) * 8
        self.token_size = self.padded_size // 8
        self.pad_before = (self.padded_size - sar_size) // 2
        self.pad_after = self.padded_size - sar_size - self.pad_before
        path = Path(checkpoint).resolve(strict=True) if load_pretrained else None
        source_path = (Path(source) if source is not None else
                       Path(__file__).resolve().parents[1] / 'vendor/CROMA/use_croma.py').resolve(strict=True)
        source_bytes = source_path.read_bytes().replace(b'\r\n', b'\n').replace(b'\r', b'\n')
        if hashlib.sha256(source_bytes).hexdigest() != CROMA_SOURCE_SHA256:
            raise ValueError('official CROMA source SHA256 mismatch')
        if load_pretrained and (path.stat().st_size != CROMA_CHECKPOINT_BYTES or _file_sha256(path) != CROMA_CHECKPOINT_SHA256):
            raise ValueError('official CROMA base checkpoint size/SHA256 mismatch')
        module = ModuleType('_authenticated_native_sar_croma')
        module.__file__ = str(source_path)
        exec(compile(source_bytes, str(source_path), 'exec'), module.__dict__)
        self.encoder = module.ViT(dim=768, depth=6, in_channels=2)
        if load_pretrained:
            payload = torch.load(str(path), map_location='cpu', weights_only=True, mmap=True)
            state = payload.get('s1_encoder') if isinstance(payload, Mapping) else None
            if not isinstance(state, Mapping) or not state:
                raise ValueError('official CROMA checkpoint has no s1_encoder state')
            for name, value in state.items():
                if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
                    raise ValueError(f'invalid official CROMA tensor {name}')
            self.encoder.load_state_dict(state, strict=True)
        self.encoder.requires_grad_(False).eval()
        self.register_buffer('attention_bias', module.get_2dalibi(
            num_heads=16, num_patches=self.token_size ** 2), persistent=False)
        self.provenance = {
            'checkpoint': str(path), 'checkpoint_sha256': CROMA_CHECKPOINT_SHA256,
            'source': str(source_path), 'source_sha256': CROMA_SOURCE_SHA256,
            'commit': CROMA_COMMIT, 'component': 's1_encoder', 'strict_load': bool(load_pretrained),
            'frozen': True, 'native_size': sar_size, 'padded_size': self.padded_size,
            'token_size': self.token_size, 'pad_before': self.pad_before,
            'pad_after': self.pad_after,
            'normalization': 'valid_only_per_window_channel_mean_plusminus2_sample_std_uint8_div255',
            'zero_control': 'zero_encoder_input_preserving_identical_validity_no_real_values_used',
            'attention_validity': 'invalid_keys_minus10000_invalid_queries_zero',
        }
        if load_pretrained:
            del state, payload
        self.train(False)

    def train(self, mode: bool = True):
        super().train(False)
        self.encoder.eval()
        return self

    def forward(self, db: torch.Tensor, native_valid: torch.Tensor,
                aux_mode: str = 'real') -> torch.Tensor:
        if (db.ndim != 4 or tuple(db.shape[1:]) != (2, self.sar_size, self.sar_size) or not db.is_floating_point()
                or aux_mode not in ('real', 'zero')):
            raise ValueError('native CROMA input shape or real/zero control is invalid')
        if (native_valid.shape != (db.shape[0], 1, self.sar_size, self.sar_size)
                or native_valid.dtype != torch.bool or native_valid.device != db.device):
            raise ValueError('native SAR validity shape/dtype/device mismatch')
        if self.training or self.encoder.training or any(p.requires_grad for p in self.encoder.parameters()):
            raise RuntimeError('CROMA must remain frozen and in evaluation mode')
        with torch.no_grad():
            # Crucially, the zero arm never normalizes or derives any statistic
            # from real dB values. Coverage masks are deliberately matched.
            x = (normalize_croma_sar(db, native_valid) if aux_mode == 'real'
                 else torch.zeros_like(db, dtype=torch.float32))
            pad = (self.pad_before, self.pad_after) * 2
            x = F.pad(x, pad, value=0)
            mask = F.pad(native_valid.float(), pad, value=0)
            observed = F.avg_pool2d(mask, 8, 8).flatten(1) > 0
            key_penalty = (~observed).float()[:, None, None, :] * -10000.
            tokens = self.encoder(imgs=x, attn_bias=self.attention_bias + key_penalty)
            tokens = torch.where(observed[..., None], tokens, torch.zeros_like(tokens))
            return tokens.transpose(1, 2).reshape(
                db.shape[0], 768, self.token_size, self.token_size).detach()


def croma_to_padded_optical(feature: torch.Tensor, output_size: tuple[int, int],
                            *, sar_size: int = 166, input_size: int = 512,
                            optical_padded_size: int = 518,
                            optical_patch_size: int = 14) -> torch.Tensor:
    """Map native patch8 SAR tokens to the 518-frame DPT feature centres.

    The148/74/37 centres are (i+.5)*518/H. The19 coarse scale comes from a
    kernel3/stride2/pad1 convolution on37 tokens, so its centres are7+28*i,
    NOT uniformly spaced across518. Remove three optical padding pixels, map
    the512 extent onto SAR166, then add its one pixel padding.
    """
    if (feature.ndim != 4 or min(output_size) <= 0 or sar_size <= 0
            or optical_padded_size < input_size or optical_patch_size <= 0
            or optical_padded_size % optical_patch_size
            or (optical_padded_size - input_size) % 2):
        raise ValueError('invalid CROMA/optical mapping geometry')
    padded = math.ceil(sar_size / 8) * 8
    if feature.shape[-2:] != (padded // 8, padded // 8):
        raise ValueError('CROMA feature is not its native patch8 lattice')
    before = (padded - sar_size) // 2
    optical_pad = (optical_padded_size - input_size) // 2
    axes = []
    coarsest_size = math.ceil((optical_padded_size // optical_patch_size) / 2)
    for n in output_size:
        indices = torch.arange(n, device=feature.device).float()
        centres = ((2 * indices + .5) * optical_patch_size if n == coarsest_size
                   else (indices + .5) * optical_padded_size / n)
        axes.append(2 * (before + ((centres - optical_pad) / input_size) * sar_size) / padded - 1)
    yy, xx = torch.meshgrid(*axes, indexing='ij')
    grid = torch.stack((xx, yy), -1)[None].expand(feature.shape[0], -1, -1, -1)
    # Sampling and its backward run FP32, avoiding FP16 coordinate quantization.
    with torch.autocast(device_type=feature.device.type, enabled=False):
        return F.grid_sample(feature.float(), grid, mode='bilinear',
                             padding_mode='border', align_corners=False)


class _ConditionalResidual(nn.Module):
    def __init__(self, channels: int, width: int) -> None:
        super().__init__()
        self.optical = _block(channels, width)
        self.mix = _block(2 * width + 1, width, kernel=3)
        self.output = nn.Conv2d(width, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, optical: torch.Tensor, sar: torch.Tensor,
                coverage: torch.Tensor) -> torch.Tensor:
        x = torch.cat((self.optical(optical), sar.to(optical.dtype),
                       coverage.to(optical.dtype)), dim=1)
        residual = self.output(self.mix(x))
        return optical + residual * coverage.to(residual.dtype)


class NativeSARAdapterModel(nn.Module):
    """Preserve all native D2 weights, add four zero-initialized adapters.

    ``prepare_ms`` can be shared across all arm/checkpoint evaluations. The pack
    contains native *tile* DPT pyramids, never stitched/downsampled feature caches.
    ``encode_sar`` can be shared per real/zero mode. All invalid output pixels
    explicitly fall back to original D2 logits, including after training.
    """
    def __init__(self, native_d2: nn.Module, croma_path: str | Path,
                 croma_source: str | Path | None = None, width: int = 64,
                 sar_size: int = 166) -> None:
        super().__init__()
        if type(width) is not int or width < 4:
            raise ValueError('adapter width must be an integer >=4')
        if (native_d2.input_size, native_d2.padded_size, native_d2.pad_pixels) != (512, 518, 3):
            raise ValueError('native D2 must preserve 512->518 padding and 3px crop')
        decoder = native_d2.decoder
        if any(len(getattr(decoder, name)) != 4 for name in ('reassemble', 'scratch', 'refine')):
            raise ValueError('native D2 must have four unchanged DPT scales')
        channels = tuple(layer.out_channels for layer in decoder.scratch)
        if len(set(channels)) != 1 or decoder.head[-1].out_channels != 24:
            raise ValueError('native D2 scratch/head architecture mismatch')
        self.native_d2 = native_d2.requires_grad_(False).eval()
        self.croma = NativeCromaSAR(croma_path, croma_source, sar_size)
        self.sar_size, self.width = sar_size, width
        # All newly trainable weights live here, so checkpoint files contain no
        # duplicate frozen D2/CROMA weights and can be loaded strictly in-place.
        self.adapters = nn.ModuleDict({
            'sar_projection': _block(768, width),
            'scales': nn.ModuleList([_ConditionalResidual(c, width) for c in channels]),
        })
        self.train(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.native_d2.eval()
        self.croma.train(False)
        return self

    def adapter_parameters(self):
        return self.adapters.parameters()

    def adapter_state_dict(self) -> Mapping[str, torch.Tensor]:
        return self.adapters.state_dict()

    def load_adapter_state_dict(self, state: Mapping[str, torch.Tensor], strict: bool = True):
        if not strict:
            raise ValueError('adapter checkpoints must be loaded strictly')
        for name, value in state.items():
            if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
                raise ValueError(f'invalid adapter checkpoint tensor {name}')
        return self.adapters.load_state_dict(state, strict=True)

    def _assert_frozen(self) -> None:
        if (self.native_d2.training or self.croma.training
                or any(p.requires_grad for p in self.native_d2.parameters())
                or any(p.requires_grad for p in self.croma.parameters())):
            raise RuntimeError('native D2 and CROMA must remain frozen/eval')

    def _decode_pyramid(self, pyramid: Sequence[torch.Tensor]) -> torch.Tensor:
        decoder = self.native_d2.decoder
        value = decoder.refine[3](pyramid[3])
        value = decoder.refine[2](value, pyramid[2])
        value = decoder.refine[1](value, pyramid[1])
        value = decoder.refine[0](value, pyramid[0])
        logits = decoder.head(value)
        logits = F.interpolate(logits, size=(518, 518), mode='bilinear', align_corners=False)
        return logits[..., 3:-3, 3:-3]

    def prepare_ms(self, ms_normalized: torch.Tensor) -> dict[str, Any]:
        self._assert_frozen()
        if (ms_normalized.ndim != 4 or ms_normalized.shape[1:] != (4, 512, 512)
                or not ms_normalized.is_floating_point() or not bool(torch.isfinite(ms_normalized).all())):
            raise ValueError('normalized native MS must be finite Bx4x512x512')
        with torch.no_grad():
            rgb, nir = self.native_d2.prepare_inputs(ms_normalized)
            features = self.native_d2._backbone_features(rgb, nir)
            decoder = self.native_d2.decoder
            pyramid = tuple(lateral(reassemble(feature)).detach() for feature, reassemble, lateral
                            in zip(features, decoder.reassemble, decoder.scratch))
            if len(pyramid) != 4:
                raise RuntimeError('native backbone did not expose four DPT features')
            original = self._decode_pyramid(pyramid).detach()
        return {'pyramid': pyramid, 'original_logits': original}

    def encode_sar(self, sar_db: torch.Tensor, sar_native_valid: torch.Tensor,
                   aux_mode: str = 'real') -> dict[str, Any]:
        self._assert_frozen()
        return {'tokens': self.croma(sar_db, sar_native_valid, aux_mode),
                'aux_mode': aux_mode, 'native_valid': sar_native_valid.detach().clone()}

    def forward_prepared(self, prepared: Mapping[str, Any], sar_db: torch.Tensor,
                         sar_native_valid: torch.Tensor, valid: torch.Tensor,
                         aux_mode: str = 'real', return_original: bool = True,
                         sar_features: Mapping[str, Any] | None = None) -> dict[str, torch.Tensor]:
        self._assert_frozen()
        if aux_mode not in ('real', 'zero'):
            raise ValueError('only matched real/zero SAR modes are supported')
        original = prepared['original_logits'].detach()
        pyramid = tuple(x.detach() for x in prepared['pyramid'])
        batch = original.shape[0]
        if (original.shape != (batch, 24, 512, 512) or len(pyramid) != 4
                or valid.shape != (batch, 1, 512, 512) or valid.dtype != torch.bool
                or valid.device != original.device):
            raise ValueError('prepared native D2 pack or 512 validity has invalid shape/device')
        if (sar_db.shape != (batch, 2, self.sar_size, self.sar_size)
                or sar_native_valid.shape != (batch, 1, self.sar_size, self.sar_size)
                or sar_native_valid.dtype != torch.bool
                or sar_db.device != original.device or sar_native_valid.device != original.device):
            raise ValueError('native SAR and prepared optical shapes/devices differ')
        encoded = (self.encode_sar(sar_db, sar_native_valid, aux_mode) if sar_features is None
                   else sar_features)
        if (not isinstance(encoded, Mapping) or encoded.get('aux_mode') != aux_mode
                or not isinstance(encoded.get('native_valid'), torch.Tensor)
                or not torch.equal(encoded['native_valid'], sar_native_valid)):
            raise ValueError('precomputed CROMA mode/validity differs: real tokens cannot enter the zero control')
        tokens = encoded['tokens'].detach()
        token_size = math.ceil(self.sar_size / 8)
        if (tokens.shape != (batch, 768, token_size, token_size)
                or tokens.device != original.device or not bool(torch.isfinite(tokens).all())):
            raise ValueError('precomputed CROMA tokens have wrong shape/device or nonfinite values')
        projected = self.adapters['sar_projection'](tokens)
        padded_valid = F.pad(valid.detach().float(), (3, 3, 3, 3), value=0)
        enhanced = []
        for optical, adapter in zip(pyramid, self.adapters['scales']):
            sar = croma_to_padded_optical(projected, optical.shape[-2:], sar_size=self.sar_size)
            if optical.shape[-2:] == (19, 19):
                # Same37->19 stride2/pad1 centre lattice as reassemble[3].
                coverage = F.avg_pool2d(F.avg_pool2d(padded_valid, 14, 14), 3, 2, 1)
            else:
                coverage = F.interpolate(padded_valid, size=optical.shape[-2:], mode='area')
            enhanced.append(adapter(optical, sar, coverage))
        # This path deliberately is NOT no_grad: gradients must traverse the
        # unchanged frozen nonlinear decoder to reach the inserted adapters.
        adapted = self._decode_pyramid(enhanced)
        result = {'logits': torch.where(valid, adapted, original)}
        if return_original:
            result['original_logits'] = original
        return result

    def forward(self, ms_normalized: torch.Tensor, sar_db: torch.Tensor,
                sar_native_valid: torch.Tensor, valid: torch.Tensor,
                aux_mode: str = 'real', return_original: bool = True) -> dict[str, torch.Tensor]:
        return self.forward_prepared(self.prepare_ms(ms_normalized), sar_db,
                                     sar_native_valid, valid, aux_mode, return_original)


__all__ = ['NativeCromaSAR', 'NativeSARAdapterModel', 'croma_to_padded_optical']
