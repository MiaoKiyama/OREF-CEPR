"""Probability-free deep-feature semantic decoders for five matched aux arms.

The decoder never accepts expert probabilities or argmax labels. Its complete
24-class logits are independently supervised; any later probability mixture and
missing-modality fallback is external. SAR uses SHA-authenticated, frozen official
CROMA spatial tokens, not a silently substituted random encoder.
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


NUM_CLASSES = 24
CROMA_CHECKPOINT_SHA256 = '0238d814b53108f3574bf1ea240e38a0a6edd46173816d9a6962070561893b63'
CROMA_CHECKPOINT_BYTES = 777563846
CROMA_SOURCE_SHA256 = 'a38567beed29eb08108a47cdc97fe98aec50fd4be0bd98a5266bcd18aafb7c5b'
CROMA_COMMIT = '59505a6bcadbf36ba20767270154bf9f3067c5e7'
AUX_MODES = ('real', 'zero', 'lowpass')


def _file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def _groups(channels: int) -> int:
    return next(g for g in (8, 4, 2, 1) if channels % g == 0 and (channels // g >= 2 or g == 1))


def _block(cin: int, cout: int, stride: int = 1, kernel: int = 3) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, stride=stride, padding=kernel // 2, bias=False),
                         nn.GroupNorm(_groups(cout), cout), nn.GELU())


def _resize(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return value if value.shape[-2:] == size else F.interpolate(value, size=size, mode='bilinear', align_corners=False)


def _native_mask(aux: torch.Tensor, native_valid: torch.Tensor | None, valid: torch.Tensor) -> torch.Tensor:
    if native_valid is None:
        if not bool(valid.all()):
            raise ValueError('partial/missing auxiliary coverage requires its native_valid mask')
        native_valid = torch.ones((aux.shape[0], 1, *aux.shape[-2:]), dtype=torch.bool, device=aux.device)
    if (native_valid.shape != (aux.shape[0], 1, *aux.shape[-2:])
            or native_valid.dtype != torch.bool or native_valid.device != aux.device):
        raise ValueError('native_valid must be bool Bx1xnative_Hxnative_W on the auxiliary device')
    return native_valid.detach()


def normalize_croma_sar(db: torch.Tensor, native_valid: torch.Tensor) -> torch.Tensor:
    """Per-image/per-channel valid-only mean +/-2 std -> clipped uint8 /255.

    Follows the official README's 8-bit normalization, with explicit independent
    images and valid pixels rather than batch-dependent global statistics. Uses
    sample std (N-1). Constant valid channels map to 0.5 before quantization;
    empty channels and invalid pixels map to exact zero. No labels/test statistics.
    Input is VV,VH in raw dB, NOT the earlier [-1,1] auxiliary normalization.
    """
    if db.ndim != 4 or db.shape[1] != 2 or not db.is_floating_point():
        raise ValueError('CROMA input must be floating Bx2xHxW VV,VH dB')
    if native_valid.shape != (db.shape[0], 1, *db.shape[-2:]) or native_valid.dtype != torch.bool:
        raise ValueError('CROMA native validity shape/dtype mismatch')
    if native_valid.device != db.device:
        raise ValueError('CROMA input/validity devices differ')
    with torch.autocast(device_type=db.device.type, enabled=False):
        x = db.detach().float()
        mask = native_valid.detach().expand_as(x)
        if not bool(torch.isfinite(x[mask]).all()):
            raise ValueError('valid SAR dB pixels must be finite')
        x = torch.where(mask, x, torch.zeros_like(x))
        count = mask.sum((-2, -1), keepdim=True)
        mean = x.sum((-2, -1), keepdim=True) / count.clamp_min(1)
        centered = torch.where(mask, x - mean, torch.zeros_like(x))
        std = (centered.square().sum((-2, -1), keepdim=True) / (count - 1).clamp_min(1)).sqrt()
        scaled = (x - mean) / (4 * std.clamp_min(1e-6)) + .5
        scaled = torch.floor(scaled.clamp(0, 1) * 255) / 255
        return torch.where(mask, scaled, torch.zeros_like(scaled))


def prepare_pan(pan: torch.Tensor, native_valid: torch.Tensor, aux_mode: str,
                *, coarse_size: tuple[int, int] = (256, 256)) -> torch.Tensor:
    """Native PAN, exact zero, or validity-normalized antialiased MS-scale PAN.

    Lowpass keeps exactly the native output geometry/capacity: antialiased
    bilinear downsampling to MS resolution and bilinear upsampling to native
    resolution. The same filtering of values and validity prevents nodata zeros
    from becoming a false dark edge. Original invalid pixels remain zero.
    """
    if aux_mode not in AUX_MODES:
        raise ValueError('unknown auxiliary input control')
    if (pan.ndim != 4 or pan.shape[1] != 1 or not pan.is_floating_point()
            or native_valid.shape != pan.shape or native_valid.dtype != torch.bool
            or native_valid.device != pan.device):
        raise ValueError('PAN/native_valid must be matching floating/bool Bx1xHxW')
    if min(coarse_size) < 1 or any(c > s for c, s in zip(coarse_size, pan.shape[-2:])):
        raise ValueError('PAN lowpass target must not increase native spatial resolution')
    with torch.autocast(device_type=pan.device.type, enabled=False):
        x = pan.detach().float()
        if not bool(torch.isfinite(x[native_valid]).all()) or bool(((x[native_valid] < 0) | (x[native_valid] > 1)).any()):
            raise ValueError('valid PAN must be finite reflectance in [0,1]')
        x = torch.where(native_valid, x, torch.zeros_like(x))
        if aux_mode == 'zero':
            return torch.zeros_like(x)
        if aux_mode == 'real':
            return x
        mask = native_valid.float()
        numerator = F.interpolate(x, size=coarse_size, mode='bilinear', align_corners=False, antialias=True)
        support = F.interpolate(mask, size=coarse_size, mode='bilinear', align_corners=False, antialias=True)
        # Carry numerator/support through both resamplings; zero coarse cells
        # must not contaminate a neighbouring valid interpolation footprint.
        numerator = F.interpolate(numerator, size=pan.shape[-2:], mode='bilinear', align_corners=False)
        support = F.interpolate(support, size=pan.shape[-2:], mode='bilinear', align_corners=False)
        smoothed = numerator / support.clamp_min(1e-8)
        return torch.where(native_valid & (support > 0), smoothed.clamp(0, 1), torch.zeros_like(smoothed))


class FrozenCromaSAR(nn.Module):
    """Official base s1_encoder, exactly loaded and permanently frozen/eval.

    84x84 near-native SAR is symmetrically padded to88 for patch8. Padding is
    never treated as observed radar. Spatial token centres are mapped back to
    the original84 support when producing the32x32 decoder lattice.
    """
    def __init__(self, checkpoint: str | Path, source: str | Path | None = None) -> None:
        super().__init__()
        path = Path(checkpoint).resolve(strict=True)
        source_path = (Path(source) if source is not None else
                       Path(__file__).resolve().parents[1] / 'vendor/CROMA/use_croma.py').resolve(strict=True)
        source_bytes = source_path.read_bytes().replace(b'\r\n', b'\n').replace(b'\r', b'\n')
        if hashlib.sha256(source_bytes).hexdigest() != CROMA_SOURCE_SHA256:
            raise ValueError('official CROMA source SHA256 mismatch')
        if path.stat().st_size != CROMA_CHECKPOINT_BYTES or _file_sha256(path) != CROMA_CHECKPOINT_SHA256:
            raise ValueError('official CROMA base checkpoint size/SHA256 mismatch')
        module = ModuleType('_authenticated_official_croma')
        module.__file__ = str(source_path)
        exec(compile(source_bytes, str(source_path), 'exec'), module.__dict__)
        self.encoder = module.ViT(dim=768, depth=6, in_channels=2)
        payload = torch.load(str(path), map_location='cpu', weights_only=True, mmap=True)
        state = payload.get('s1_encoder') if isinstance(payload, Mapping) else None
        if not isinstance(state, Mapping) or not state:
            raise ValueError('official CROMA checkpoint has no s1_encoder state')
        for name, value in state.items():
            if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
                raise ValueError(f'invalid official CROMA tensor {name}')
        self.encoder.load_state_dict(state, strict=True)
        self.encoder.requires_grad_(False).eval()
        self.register_buffer('attention_bias', module.get_2dalibi(num_heads=16, num_patches=121), persistent=False)
        self.provenance = {'checkpoint': str(path), 'checkpoint_sha256': CROMA_CHECKPOINT_SHA256,
            'source': str(source_path), 'source_sha256': CROMA_SOURCE_SHA256, 'commit': CROMA_COMMIT,
            'component': 's1_encoder', 'strict_load': True, 'frozen': True,
            'native_size': 84, 'padded_size': 88, 'token_size': 11,
            'normalization': 'valid_only_per_image_channel_mean_plusminus2_sample_std_uint8_div255',
            'padding': '2_zero_pixels_each_side_excluded_from_validity',
            'attention_validity': 'fully_invalid_keys_bias_minus10000_and_invalid_query_tokens_zero'}
        del state, payload
        self.train(False)

    def train(self, mode: bool = True):
        super().train(False)
        self.encoder.eval()
        return self

    def forward(self, db: torch.Tensor, native_valid: torch.Tensor, aux_mode: str = 'real') -> torch.Tensor:
        if db.shape[-2:] != (84, 84) or aux_mode not in ('real', 'zero'):
            raise ValueError('frozen CROMA accepts SAR84x84 with real/zero control')
        if self.training or self.encoder.training or any(p.requires_grad for p in self.encoder.parameters()):
            raise RuntimeError('CROMA must remain frozen and in evaluation mode')
        with torch.no_grad():
            x = normalize_croma_sar(db, native_valid)
            if aux_mode == 'zero':
                x = torch.zeros_like(x)
            x = F.pad(x, (2, 2, 2, 2), value=0)
            mask = F.pad(native_valid.float(), (2, 2, 2, 2), value=0)
            observed_tokens = F.avg_pool2d(mask, 8, 8).flatten(1) > 0
            key_penalty = (~observed_tokens).float()[:, None, None, :] * -10000.
            tokens = self.encoder(imgs=x, attn_bias=self.attention_bias + key_penalty)
            tokens = torch.where(observed_tokens[..., None], tokens, torch.zeros_like(tokens))
            feature = tokens.transpose(1, 2).reshape(db.shape[0], 768, 11, 11).float()
            # align_corners=False centres in the original84 map into padded88.
            axis = 2 * (2 + (torch.arange(32, device=db.device).float() + .5) * 84 / 32) / 88 - 1
            yy, xx = torch.meshgrid(axis, axis, indexing='ij')
            grid = torch.stack((xx, yy), -1)[None].expand(db.shape[0], -1, -1, -1)
            return F.grid_sample(feature, grid, mode='bilinear', padding_mode='border', align_corners=False).detach()


class SemanticAuxDecoder(nn.Module):
    """Deep D2/MS75 features + raw MS + sensor -> direct24-class semantics."""
    def __init__(self, modality: str = 'sar', d2_channels: int = 256, ms75_channels: int = 256,
                 width: int = 96, croma_path: str | Path | None = None,
                 croma_source: str | Path | None = None) -> None:
        super().__init__()
        if modality not in ('sar', 'pan'):
            raise ValueError('modality must be sar or pan')
        if any(type(x) is not int or x < 4 for x in (d2_channels, ms75_channels, width)) or width % 4:
            raise ValueError('feature channels must be >=4 and decoder width must be divisible by4')
        if modality == 'sar' and croma_path is None:
            raise ValueError('SAR semantic experiment requires official CROMA weights; no random fallback')
        self.modality, self.d2_channels, self.ms75_channels, self.width = modality, d2_channels, ms75_channels, width
        self.d2_projection = _block(d2_channels, width, kernel=1)
        self.ms75_projection = _block(ms75_channels, width, kernel=1)
        self.ms_fine = _block(4, width // 2, stride=2)
        self.ms_coarse = _block(width // 2, width, stride=2)
        if modality == 'sar':
            self.sar_encoder = FrozenCromaSAR(croma_path, croma_source)
            self.aux_projection = _block(768, width, kernel=1)
        else:
            self.pan_encoder = nn.Sequential(_block(1, width // 4, 2), _block(width // 4, width // 4, 2),
                                             _block(width // 4, width // 2, 2), _block(width // 2, width, 2))
            self.aux_projection = _block(width, width)
        self.core = nn.Sequential(_block(3 * width + 1, width), _block(width, width))
        self.decoder64 = nn.Sequential(_block(3 * width, width), _block(width, width))
        # PAN's third stride stage retains128x128 native-edge information. It
        # bypasses the semantic32x32 bottleneck for all three matched PAN arms.
        fine_channels = 2 * width if modality == 'pan' else width + width // 2
        self.decoder128 = nn.Sequential(_block(fine_channels, width // 2), _block(width // 2, width // 2))
        self.classifier = nn.Conv2d(width // 2, NUM_CLASSES, 1)
        self.aux_classifier = nn.Sequential(_block(width, width), nn.Conv2d(width, NUM_CLASSES, 1))
        self.boundary_classifier = nn.Conv2d(width // 2, 1, 1)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.modality == 'sar':
            self.sar_encoder.train(False)
        return self

    def forward(self, d2_features: torch.Tensor, ms75_features: torch.Tensor, ms: torch.Tensor,
                aux: torch.Tensor, valid: torch.Tensor, native_valid: torch.Tensor | None = None,
                aux_mode: str = 'real') -> dict[str, Any]:
        if aux_mode not in AUX_MODES or (self.modality == 'sar' and aux_mode == 'lowpass'):
            raise ValueError('unsupported sensor/control combination')
        if ms.ndim != 4 or ms.shape[1:] != (4, 256, 256) or not ms.is_floating_point():
            raise ValueError('MS must be floating Bx4x256x256 B,G,R,NIR reflectance')
        if not bool(torch.isfinite(ms).all()) or bool(((ms < 0) | (ms > 1)).any()):
            raise ValueError('raw MS must be finite [0,1]')
        batch = ms.shape[0]
        if valid.shape != (batch, 1, 256, 256) or valid.dtype != torch.bool or valid.device != ms.device:
            raise ValueError('valid must be bool Bx1x256x256 on MS device')
        for name, feature, channels in [('d2', d2_features, self.d2_channels), ('ms75', ms75_features, self.ms75_channels)]:
            if (feature.shape != (batch, channels, 32, 32) or feature.device != ms.device
                    or not feature.is_floating_point() or not bool(torch.isfinite(feature).all())):
                raise ValueError(f'{name} must be finite native-feature Bx{channels}x32x32')
        expected_aux = (batch, 2, 84, 84) if self.modality == 'sar' else (batch, 1, 1024, 1024)
        if aux.shape != expected_aux or aux.device != ms.device or not aux.is_floating_point():
            raise ValueError(f'{self.modality} native auxiliary input must be floating {expected_aux}')
        native_valid = _native_mask(aux, native_valid, valid)
        dtype = self.classifier.weight.dtype
        f_d2 = self.d2_projection(d2_features.detach().to(dtype))
        f_ms75 = self.ms75_projection(ms75_features.detach().to(dtype))
        f_ms128 = self.ms_fine(ms.detach().to(dtype))
        f_ms64 = self.ms_coarse(f_ms128)
        if self.modality == 'sar':
            f_aux = self.aux_projection(self.sar_encoder(aux, native_valid, aux_mode).to(dtype))
        else:
            prepared = prepare_pan(aux, native_valid, aux_mode)
            f_pan128 = self.pan_encoder[:3](prepared.to(dtype))
            f_aux = self.aux_projection(self.pan_encoder[3](f_pan128))
            f_pan128 = f_pan128 * F.interpolate(valid.detach().float(), size=(128, 128), mode='area')
        # Preserve optical context outside sensor coverage; only the auxiliary
        # representation is masked. The loss and external fusion use valid too.
        aux_mask = F.interpolate(valid.detach().float(), size=f_aux.shape[-2:], mode='area')
        f_aux = f_aux * aux_mask
        aux32 = _resize(f_aux, (32, 32))
        core = self.core(torch.cat((f_d2, f_ms75, aux32,
                                   F.interpolate(valid.float(), size=(32, 32), mode='area')), 1))
        decoded64 = self.decoder64(torch.cat((_resize(core, (64, 64)), f_ms64, _resize(f_aux, (64, 64))), 1))
        fine_inputs = [_resize(decoded64, (128, 128)), f_ms128]
        if self.modality == 'pan':
            fine_inputs.append(f_pan128)
        decoded128 = self.decoder128(torch.cat(fine_inputs, 1))
        return {'logits': _resize(self.classifier(decoded128).float(), (256, 256)),
                'aux_logits': _resize(self.aux_classifier(aux32).float(), (256, 256)),
                'boundary_logits': _resize(self.boundary_classifier(decoded128).float(), (256, 256))}


def boundary_targets(target: torch.Tensor, valid: torch.Tensor, ignore_index: int = 255):
    """Four-neighbour GT boundary; exclude unknown neighbours and image edges."""
    if target.ndim != 3 or valid.shape != (target.shape[0], 1, *target.shape[-2:]) or valid.dtype != torch.bool:
        raise ValueError('boundary targets require BHW labels and B1HW valid mask')
    mask = valid[:, 0] & (target != ignore_index)
    if bool((mask & ((target < 0) | (target >= NUM_CLASSES))).any()):
        raise ValueError('boundary valid labels out of range')
    edge = torch.zeros_like(mask)
    horizontal = (target[:, :, 1:] != target[:, :, :-1]) & mask[:, :, 1:] & mask[:, :, :-1]
    vertical = (target[:, 1:, :] != target[:, :-1, :]) & mask[:, 1:, :] & mask[:, :-1, :]
    edge[:, :, 1:] |= horizontal; edge[:, :, :-1] |= horizontal
    edge[:, 1:, :] |= vertical; edge[:, :-1, :] |= vertical
    interior = torch.zeros_like(mask)
    interior[:, 1:-1, 1:-1] = (mask[:, 1:-1, 1:-1] & mask[:, :-2, 1:-1] & mask[:, 2:, 1:-1]
                                      & mask[:, 1:-1, :-2] & mask[:, 1:-1, 2:])
    return edge[:, None].float(), interior[:, None]


def semantic_loss(outputs: Mapping[str, torch.Tensor], target: torch.Tensor, valid: torch.Tensor, *,
                  class_weights: torch.Tensor | Sequence[float] | None = None, dice_weight: float = .5,
                  aux_weight: float = .2, boundary_weight: float = 0., ignore_index: int = 255) -> dict[str, torch.Tensor]:
    """Stable weighted log-softmax CE + Dice + existing auxiliary supervision.

    All valid supervised pixels are used, never only expert errors. Boundary
    labels enter only this training loss. No probability floor or bounded-logit
    restriction exists in the final CE path.
    """
    if any(not math.isfinite(x) or x < 0 for x in (dice_weight, aux_weight, boundary_weight)):
        raise ValueError('loss weights must be finite and nonnegative')
    logits = outputs['logits']
    if logits.ndim != 4 or logits.shape[1] != NUM_CLASSES or not logits.is_floating_point():
        raise ValueError('semantic logits must be floating Bx24xHxW')
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if (target.shape != (logits.shape[0], *logits.shape[-2:]) or target.device != logits.device
            or target.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)):
        raise ValueError('target must be integer BHW on logits device')
    if valid.shape != (logits.shape[0], 1, *logits.shape[-2:]) or valid.dtype != torch.bool or valid.device != logits.device:
        raise ValueError('semantic loss valid mask must be matching bool B1HW')
    aux_logits = outputs['aux_logits']
    if aux_logits.shape != logits.shape or aux_logits.device != logits.device:
        raise ValueError('auxiliary semantic logits must match final logits')
    boundary_logits = outputs.get('boundary_logits')
    if boundary_weight and (boundary_logits is None or boundary_logits.shape != valid.shape or boundary_logits.device != logits.device):
        raise ValueError('boundary supervision requires matching B1HW logits')
    for value in (logits, aux_logits, boundary_logits):
        if value is not None and not bool(torch.isfinite(value).all()):
            raise ValueError('semantic logits must be finite')
    target = target.long()
    mask = valid[:, 0] & (target != ignore_index)
    if bool((mask & ((target < 0) | (target >= NUM_CLASSES))).any()):
        raise ValueError('valid semantic target outside 0..23')
    count = mask.sum()
    if not bool(mask.any()):
        zero = (logits.float().sum() + aux_logits.float().sum()) * 0
        if boundary_logits is not None: zero = zero + boundary_logits.float().sum() * 0
        return {'loss': zero, 'ce': zero, 'final_ce': zero, 'dice': zero, 'aux_ce': zero,
                'boundary': zero, 'valid_pixels': count}
    with torch.autocast(device_type=logits.device.type, enabled=False):
        if class_weights is None:
            counts = torch.bincount(target[mask], minlength=NUM_CLASSES).float()
            present = counts > 0
            weights = torch.where(present, counts.clamp_min(1).rsqrt(), torch.zeros_like(counts))
            weights = weights / weights[present].mean().clamp_min(1e-8)
            weights = weights.clamp_max(4)
        else:
            weights = torch.as_tensor(class_weights, dtype=torch.float32, device=logits.device).detach()
            if weights.shape != (NUM_CLASSES,) or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
                raise ValueError('class weights must be24 finite nonnegative values')
        safe = target.masked_fill(~mask, 0)
        pixel_weights = weights[safe] * mask
        denominator = pixel_weights.sum()
        if not bool(denominator > 0):
            raise ValueError('no positive class weight among valid labels')
        log_probability = F.log_softmax(logits.float(), dim=1)
        final_ce = (-log_probability.gather(1, safe[:, None])[:, 0] * pixel_weights).sum() / denominator
        aux_ce = (-F.log_softmax(aux_logits.float(), dim=1).gather(1, safe[:, None])[:, 0] * pixel_weights).sum() / denominator
        probability = log_probability.exp() * mask[:, None]
        onehot = F.one_hot(safe, NUM_CLASSES).permute(0, 3, 1, 2).float() * mask[:, None]
        intersection = (probability * onehot).sum((0, 2, 3))
        support = onehot.sum((0, 2, 3))
        class_dice = 1 - (2 * intersection + 1e-6) / (probability.sum((0, 2, 3)) + support + 1e-6)
        dice_weights = weights * (support > 0)
        dice = (class_dice * dice_weights).sum() / dice_weights.sum().clamp_min(1e-8)
        boundary = logits.float().sum() * 0
        if boundary_weight:
            edges, boundary_mask = boundary_targets(target, valid, ignore_index)
            if bool(boundary_mask.any()):
                positives = edges[boundary_mask].sum()
                negatives = boundary_mask.sum() - positives
                pos_weight = torch.where(positives > 0, (negatives / positives.clamp_min(1)).clamp(1, 20), positives.new_ones(()))
                loss_map = F.binary_cross_entropy_with_logits(boundary_logits.float(), edges,
                                                              pos_weight=pos_weight, reduction='none')
                boundary = loss_map[boundary_mask].mean()
        total = final_ce + dice_weight * dice + aux_weight * aux_ce + boundary_weight * boundary
        return {'loss': total, 'ce': final_ce, 'final_ce': final_ce, 'dice': dice, 'aux_ce': aux_ce,
                'boundary': boundary, 'valid_pixels': count}


def apply_semantic_fusion(p0: torch.Tensor, logits: torch.Tensor, valid: torch.Tensor, weight: float) -> torch.Tensor:
    """External optional mixture; exact baseline outside valid or at weight0."""
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('semantic mixture weight must be in [0,1]')
    if (p0.ndim != 4 or p0.shape[1] != NUM_CLASSES or p0.dtype != torch.float32
            or logits.shape != p0.shape or logits.device != p0.device
            or valid.shape != (p0.shape[0], 1, *p0.shape[-2:]) or valid.dtype != torch.bool
            or valid.device != p0.device):
        raise ValueError('mixture requires FP32 B24HW baseline/logits and bool B1HW valid')
    if not bool(torch.isfinite(p0).all()) or bool(((p0 < 0) | (p0 > 1)).any()) or float((p0.sum(1) - 1).abs().max()) > 2e-5:
        raise ValueError('baseline must be normalized finite probabilities')
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('semantic logits must be finite')
    if weight == 0:
        return p0
    with torch.autocast(device_type=p0.device.type, enabled=False):
        mixed = p0 * (1 - weight) + logits.float().softmax(1) * weight
        return torch.where(valid, mixed, p0)


loss_fn = semantic_loss
__all__ = ['SemanticAuxDecoder', 'FrozenCromaSAR', 'normalize_croma_sar', 'prepare_pan',
           'semantic_loss', 'loss_fn', 'boundary_targets', 'apply_semantic_fusion']
