"""Fresh D2/SAR supervised experts; no optical teacher or deployment fallback.

MS inputs are normalized B/G/R/NIR on the native 512 grid. SAR inputs are
VV/VH dB on the matching 166 grid. A zero-SAR control keeps the same masks,
initialization and trainable architecture; it is not a missing-SAR sample.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .dinov2_nir_model import DINOv2NIRSegmentation, DINOV2_VITB14_OFFICIAL_SHA256
from .dinov2_nir_optim import dinov2_nir_parameter_groups
from .native_sar_adapter import NativeCromaSAR, croma_to_padded_optical
from .semantic_aux_model import normalize_croma_sar


DECUR_COMMIT = 'f190e9a3895ef645c005c8c2fce287ffa5a937e3'
DECUR_HF_COMMIT = '3d856701da5dd847c52385181789a592511018fb'
DECUR_SHA256 = 'f0e69ba2dfecef0e16b42604358f58f84b9c80a5e32b9935dfa70b58faef7a0c'
DECUR_BYTES = 94333893
DECUR_URL = ('https://huggingface.co/wangyi111/DeCUR/resolve/' + DECUR_HF_COMMIT + '/'
             'rn50_ssl4eo-s12_sar_decur_ep100.pth')
DECUR_DATA_SOURCE_SHA256 = '1c99ec13ece094c6b7592dbf9399fd9275626ca4de8142201efd6a5db96b0b34'
DECUR_MEAN = (-12.54847273, -20.19237134)
DECUR_STD = (5.25697717, 5.91150917)


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            result.update(part)
    return result.hexdigest()


def _validate_sar(db: torch.Tensor, valid: torch.Tensor, aux_mode: str) -> None:
    if (db.ndim != 4 or db.shape[0] < 1 or tuple(db.shape[1:]) != (2, 166, 166)
            or not db.is_floating_point() or aux_mode not in ('real', 'zero')):
        raise ValueError('SAR requires floating Bx2x166x166 VV/VH and real/zero mode')
    if (valid.shape != (db.shape[0], 1, 166, 166) or valid.dtype != torch.bool
            or valid.device != db.device):
        raise ValueError('SAR native validity requires matching boolean Bx1x166x166')
    if aux_mode == 'real' and not bool(torch.isfinite(db[valid.expand_as(db)]).all()):
        raise ValueError('observed SAR dB must be finite')


def normalize_decur_sar(db: torch.Tensor, valid: torch.Tensor,
                        aux_mode: str = 'real') -> torch.Tensor:
    """Official SSL4EO VV/VH fixed mean +/-2std -> uint8 -> ToTensor.

    Constants/source pinned to DeCUR's SSL4EO dataset writer. No per-window
    quantile or normalization, no reversed channels, no ImageNet normalization.
    Invalid pixels are zero. Zero mode never reads real intensities/statistics.
    """
    _validate_sar(db, valid, aux_mode)
    if aux_mode == 'zero':
        return torch.zeros_like(db, dtype=torch.float32)
    with torch.autocast(device_type=db.device.type, enabled=False):
        mean = db.new_tensor(DECUR_MEAN, dtype=torch.float32)[None, :, None, None]
        std = db.new_tensor(DECUR_STD, dtype=torch.float32)[None, :, None, None]
        clean = torch.where(valid, db.float(), mean)
        value = ((clean - (mean - 2 * std)) / (4 * std) * 255).clamp(0, 255)
        value = value.floor() / 255.0
        return torch.where(valid, value, torch.zeros_like(value))


class JointCromaSAR(nn.Module):
    """Authenticated CROMA, frozen prefix, trainable final2 blocks and norm."""
    channels = (768, 768, 768, 768)
    geometry = 'croma_patch8'

    def __init__(self, checkpoint: str | Path, source: str | Path | None = None) -> None:
        super().__init__()
        self.loader = NativeCromaSAR(checkpoint, source, sar_size=166)
        encoder = self.loader.encoder
        if (len(encoder.transformer.layers) != 6 or encoder.patch_size != 8
                or encoder.dim != 768 or not isinstance(encoder.transformer.norm_out, nn.LayerNorm)):
            raise ValueError('official CROMA base architecture changed')
        for block in encoder.transformer.layers[4:]:
            block.requires_grad_(True)
        encoder.transformer.norm_out.requires_grad_(True)
        self.provenance = {**self.loader.provenance, 'frozen': False,
                           'frozen_prefix': 'patch_projection_and_blocks0_to3',
                           'trainable': 'blocks4_5_and_final_norm'}
        self.train(False)

    def train(self, mode: bool = True):
        super().train(mode)
        # The original loader recursively forces eval; selectively restore only
        # trainable tail mode afterwards. Do not call loader.forward: it detaches.
        self.loader.train(False)
        for block in self.loader.encoder.transformer.layers[4:]:
            block.train(mode)
        self.loader.encoder.transformer.norm_out.train(mode)
        return self

    @staticmethod
    def _block(block: nn.Module, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        attention, ffn = block
        x = attention(x, bias) + x
        return ffn(x) + x

    def forward(self, db: torch.Tensor, valid: torch.Tensor,
                aux_mode: str = 'real') -> tuple[torch.Tensor, ...]:
        _validate_sar(db, valid, aux_mode)
        encoder = self.loader.encoder
        if (self.loader.training or encoder.training or encoder.linear_input.training
                or any(b.training for b in encoder.transformer.layers[:4])):
            raise RuntimeError('permanently frozen CROMA prefix must remain eval')
        with torch.no_grad():
            value = (normalize_croma_sar(db, valid) if aux_mode == 'real'
                     else torch.zeros_like(db, dtype=torch.float32))
            value = F.pad(value, (1, 1, 1, 1), value=0)
            observed = F.avg_pool2d(F.pad(valid.float(), (1, 1, 1, 1)), 8, 8).flatten(1) > 0
            bias = self.loader.attention_bias + (~observed).float()[:, None, None, :] * -10000.
            b, c, h, w = value.shape
            value = value.reshape(b, c, h // 8, 8, w // 8, 8).permute(0, 2, 4, 1, 3, 5)
            value = encoder.linear_input(value.reshape(b, (h // 8) * (w // 8), c * 64))
            for block in encoder.transformer.layers[:4]:
                value = self._block(block, value, bias)
        # Intentionally outside no_grad: SAR tail learns from joint segmentation.
        for block in encoder.transformer.layers[4:]:
            value = self._block(block, value, bias)
        value = encoder.transformer.norm_out(value)
        value = torch.where(observed[..., None], value, torch.zeros_like(value))
        feature = value.transpose(1, 2).reshape(db.shape[0], 768, 21, 21)
        return (feature,) * 4


class JointDeCURSAR(nn.Module):
    """Official ordinary RN50-SAR, all backbone weights learn, BN stats fixed."""
    channels = (256, 512, 1024, 2048)
    strides = (4, 8, 16, 32)
    geometry = 'resnet_native'

    def __init__(self, checkpoint: str | Path, sha256: str = DECUR_SHA256) -> None:
        super().__init__()
        path = Path(checkpoint).resolve(strict=True)
        if (sha256 != DECUR_SHA256 or path.stat().st_size != DECUR_BYTES
                or _sha256(path) != DECUR_SHA256):
            raise ValueError('official DeCUR RN50-SAR checkpoint size/SHA mismatch')
        from torchvision.models import resnet50
        self.encoder = resnet50(weights=None)
        self.encoder.conv1 = nn.Conv2d(2, 64, 7, stride=2, padding=3, bias=False)
        self.encoder.fc = nn.Identity()
        state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if not isinstance(state, Mapping) or not state:
            raise ValueError('DeCUR backbone checkpoint must be a nonempty tensor mapping')
        # The linked backbone-only artifact is a plain ResNet state; wrappers,
        # projectors, missing layers and random replacements are not accepted.
        if any(not isinstance(v, torch.Tensor) or not bool(torch.isfinite(v).all())
               for v in state.values()):
            raise ValueError('DeCUR checkpoint contains non-tensor/nonfinite values')
        self.encoder.load_state_dict(state, strict=True)
        self.provenance = {'checkpoint': str(path), 'checkpoint_sha256': DECUR_SHA256,
                           'checkpoint_url': DECUR_URL, 'checkpoint_bytes': DECUR_BYTES,
                           'source_commit': DECUR_COMMIT, 'preprocessing_source_sha256': DECUR_DATA_SOURCE_SHA256,
                           'weights_commit': DECUR_HF_COMMIT,
                           'strict_load': True, 'architecture': 'ordinary_torchvision_RN50_SAR_no_RDA',
                           'normalization': 'official_SSL4EO_fixed_VV_VH_mean_plusminus2std_uint8_div255',
                           'normalization_mean': DECUR_MEAN, 'normalization_std': DECUR_STD,
                           'trainable': 'all_RN50_parameters_including_BN_affine',
                           'batchnorm': 'pretrained_running_statistics_permanently_eval'}
        self.train(False)

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, db: torch.Tensor, valid: torch.Tensor,
                aux_mode: str = 'real') -> tuple[torch.Tensor, ...]:
        value = normalize_decur_sar(db, valid, aux_mode)
        encoder = self.encoder
        if any(m.training for m in encoder.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)):
            raise RuntimeError('DeCUR BatchNorm running statistics must remain frozen')
        value = encoder.maxpool(encoder.relu(encoder.bn1(encoder.conv1(value))))
        outputs = []
        for block in (encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4):
            value = block(value)
            outputs.append(value)
        return tuple(outputs)


def resnet_to_padded_optical(feature: torch.Tensor, output_size: tuple[int, int],
                             stride: int) -> torch.Tensor:
    """Sample RN50 centre .5+stride*i onto actual DPT centres; no half-pixel drift."""
    if (feature.ndim != 4 or stride not in (4, 8, 16, 32)
            or feature.shape[-2:] != (math.ceil(166 / stride),) * 2):
        raise ValueError('ResNet SAR feature lattice does not match native166 stride')
    axes = []
    for n, source_n in zip(output_size, feature.shape[-2:]):
        indices = torch.arange(n, device=feature.device).float()
        centres = (2 * indices + .5) * 14 if n == 19 else (indices + .5) * 518 / n
        sar_centres = (centres - 3) / 512 * 166
        source_indices = (sar_centres - .5) / stride
        axes.append(2 * (source_indices + .5) / source_n - 1)
    yy, xx = torch.meshgrid(*axes, indexing='ij')
    grid = torch.stack((xx, yy), -1)[None].expand(feature.shape[0], -1, -1, -1)
    with torch.autocast(device_type=feature.device.type, enabled=False):
        return F.grid_sample(feature.float(), grid, mode='bilinear', padding_mode='border', align_corners=False)


class _FeatureInteraction(nn.Module):
    def __init__(self, optical_channels: int, sar_channels: int, width: int) -> None:
        super().__init__()
        groups = next(g for g in (8, 4, 2, 1) if width % g == 0)
        self.sar_projection = nn.Sequential(nn.Conv2d(sar_channels, width, 1, bias=False),
                                            nn.GroupNorm(groups, width), nn.GELU())
        self.optical_projection = nn.Sequential(nn.Conv2d(optical_channels, width, 1, bias=False),
                                                nn.GroupNorm(groups, width), nn.GELU())
        self.mix = nn.Sequential(nn.Conv2d(width * 2 + 1, width, 3, padding=1, bias=False),
                                 nn.GroupNorm(groups, width), nn.GELU())
        self.output = nn.Conv2d(width, optical_channels, 1)
        # One zero-initialized terminal projection, no second zero scalar gate.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, optical: torch.Tensor, sar: torch.Tensor, coverage: torch.Tensor) -> torch.Tensor:
        combined = torch.cat((self.optical_projection(optical), sar.to(optical.dtype),
                              coverage.to(optical.dtype)), 1)
        return optical + self.output(self.mix(combined)) * coverage.to(optical.dtype)


class JointSARModel(nn.Module):
    """Fresh DINOv2-NIR/DPT + intermediate SAR; all student MS weights learn."""
    def __init__(self, optical: nn.Module, sar_encoder: nn.Module, *,
                 aux_mode: str = 'real', width: int = 64, layer_decay: float = .9,
                 weight_decay: float = .01) -> None:
        super().__init__()
        if aux_mode not in ('real', 'zero') or width < 8 or width % 2:
            raise ValueError('invalid joint SAR mode/interaction width')
        self.optical, self.sar_encoder = optical, sar_encoder
        self.aux_mode, self.layer_decay, self.weight_decay = aux_mode, layer_decay, weight_decay
        channels = [layer.out_channels for layer in optical.decoder.scratch]
        if len(channels) != 4 or len(sar_encoder.channels) != 4:
            raise ValueError('joint interaction requires four native feature scales')
        self.interactions = nn.ModuleList(_FeatureInteraction(o, s, width)
                                         for o, s in zip(channels, sar_encoder.channels))
        self.provenance = {'architecture': 'fresh_D2_joint_SAR_multiscale_DPT',
                           'optical_initialization': 'DINOv2_B14_foundation_plus_fresh_DPT_NIR',
                           'optical_teacher': False, 'fallback': False,
                           'aux_mode': aux_mode, 'sar': dict(sar_encoder.provenance),
                           'mask_policy': 'equal_real_zero_masks_missing_disables_interaction',
                           'input': 'normalized_BGRN512_rawVVVH_dB166',
                           'interaction_initialization': 'single_zero_terminal_projection'}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> 'JointSARModel':
        optical_config, sar_config = config['optical'], config['sar']
        model_config = optical_config['model']
        if (model_config.get('pretrained') is not True or
                model_config.get('pretrained_file_sha256') != DINOV2_VITB14_OFFICIAL_SHA256):
            raise ValueError('joint model must start from authenticated DINOv2 foundation, not full120 D2')
        # Building optical first preserves its initialization across both SAR families.
        optical = DINOv2NIRSegmentation.from_config(optical_config)
        if sar_config['encoder'] == 'croma':
            sar_encoder = JointCromaSAR(sar_config['croma_path'], sar_config.get('croma_source'))
        elif sar_config['encoder'] == 'decur':
            sar_encoder = JointDeCURSAR(sar_config['decur_path'], sar_config['decur_sha256'])
        else:
            raise ValueError('only authenticated croma/decur SAR encoders are allowed')
        return cls(optical, sar_encoder, aux_mode=sar_config['aux_mode'],
                   width=int(sar_config.get('width', 64)),
                   layer_decay=float(optical_config['train'].get('layer_decay', .9)),
                   weight_decay=float(optical_config['train'].get('weight_decay', .01)))

    def forward(self, ms: torch.Tensor, sar: torch.Tensor,
                sar_native_valid: torch.Tensor, sar_valid: torch.Tensor) -> dict[str, torch.Tensor]:
        _validate_sar(sar, sar_native_valid, self.aux_mode)
        if (ms.ndim != 4 or ms.shape != (sar.shape[0], 4, 512, 512)
                or not ms.is_floating_point() or ms.device != sar.device
                or not bool(torch.isfinite(ms).all())):
            raise ValueError('joint MS input requires finite normalized Bx4x512x512')
        if (sar_valid.shape != (ms.shape[0], 1, 512, 512) or sar_valid.dtype != torch.bool
                or sar_valid.device != ms.device):
            raise ValueError('SAR optical-grid validity requires boolean Bx1x512x512')
        features = self.optical._backbone_features(*self.optical.prepare_inputs(ms))
        decoder = self.optical.decoder
        pyramid = tuple(scratch(reassemble(feature)) for feature, reassemble, scratch
                        in zip(features, decoder.reassemble, decoder.scratch))
        sar_features = self.sar_encoder(sar, sar_native_valid, self.aux_mode)
        effective_valid = sar_valid & sar_native_valid.flatten(1).any(1)[:, None, None, None]
        padded_valid = F.pad(effective_valid.float(), (3, 3, 3, 3))
        enhanced = []
        for index, (optical, native_sar, interaction) in enumerate(zip(pyramid, sar_features, self.interactions)):
            projected = interaction.sar_projection(native_sar)
            if self.sar_encoder.geometry == 'croma_patch8':
                sampled = croma_to_padded_optical(projected, optical.shape[-2:], sar_size=166)
            elif self.sar_encoder.geometry == 'resnet_native':
                sampled = resnet_to_padded_optical(projected, optical.shape[-2:], self.sar_encoder.strides[index])
            else:
                raise ValueError('unknown SAR spatial lattice')
            coverage = (F.avg_pool2d(F.avg_pool2d(padded_valid, 14, 14), 3, 2, 1)
                        if optical.shape[-2:] == (19, 19) else
                        F.interpolate(padded_valid, size=optical.shape[-2:], mode='area'))
            enhanced.append(interaction(optical, sampled, coverage))
        value = decoder.refine[3](enhanced[3])
        value = decoder.refine[2](value, enhanced[2])
        value = decoder.refine[1](value, enhanced[1])
        value = decoder.refine[0](value, enhanced[0])
        logits = F.interpolate(decoder.head(value), (518, 518), mode='bilinear', align_corners=False)
        logits = logits[..., 3:-3, 3:-3].float()
        if logits.shape != (ms.shape[0], 24, 512, 512) or not bool(torch.isfinite(logits).all()):
            raise RuntimeError('joint student produced invalid logits')
        return {'logits': logits}

    def parameter_groups(self, base_lr: float = 2e-4, backbone_lr: float = 2e-5,
                         sar_lr: float = 3e-6) -> list[dict[str, Any]]:
        if any(not math.isfinite(x) or x <= 0 for x in (base_lr, backbone_lr, sar_lr)):
            raise ValueError('joint learning rates must be finite and positive')
        cfg = {'train': {'learning_rate': base_lr, 'backbone_lr_multiplier': backbone_lr / base_lr,
                         'layer_decay': self.layer_decay, 'weight_decay': self.weight_decay}}
        groups = dinov2_nir_parameter_groups(self.optical, cfg)
        for group in groups:
            group['group_name'] = 'optical_' + group['group_name']
            group['parameter_names'] = ['optical.' + n for n in group['parameter_names']]
        for label, module, lr in (('sar_encoder', self.sar_encoder, sar_lr),
                                  ('interactions', self.interactions, base_lr)):
            for decay in (False, True):
                members = [(label + '.' + n, p) for n, p in module.named_parameters()
                           if p.requires_grad and (p.ndim > 1 and not n.endswith('.bias')) == decay]
                if members:
                    groups.append({'params': [p for _, p in members], 'lr': lr,
                                   'weight_decay': self.weight_decay if decay else 0.,
                                   'group_name': label + ('_decay' if decay else '_no_decay'),
                                   'parameter_names': [n for n, _ in members]})
        ids = [id(p) for g in groups for p in g['params']]
        expected = {id(p) for p in self.parameters() if p.requires_grad}
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise RuntimeError('joint optimizer groups duplicate or omit trainable parameters')
        for group in groups:
            group['initial_lr'] = group['lr']
        return groups

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        if strict is not True or assign is not False:
            raise ValueError('joint checkpoints require strict=True, assign=False')
        expected = self.state_dict()
        if not isinstance(state_dict, Mapping) or set(state_dict) != set(expected):
            raise ValueError('joint checkpoint must contain the complete matching architecture')
        for name, value in state_dict.items():
            target = expected[name]
            if (not isinstance(value, torch.Tensor) or value.shape != target.shape
                    or value.dtype != target.dtype or value.layout != torch.strided
                    or not bool(torch.isfinite(value).all())):
                raise ValueError('invalid joint checkpoint tensor: ' + name)
        return super().load_state_dict(state_dict, strict=True)


__all__ = ['JointSARModel', 'JointCromaSAR', 'JointDeCURSAR', 'normalize_decur_sar',
           'resnet_to_padded_optical', 'DECUR_SHA256', 'DECUR_URL']
