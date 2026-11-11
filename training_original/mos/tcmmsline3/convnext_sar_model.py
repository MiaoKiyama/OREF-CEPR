"""DINOv3 ConvNeXt-L optical/SAR expert with authenticated native RGB weights.

The optical path is RGB ConvNeXt + an independent four-band spectral pyramid.
SAR enters only at strides16/32. CROMA content and fixed-dB intensity values
share one zero-initialized terminal projection at each insertion. There is no
old optical probability fallback and no DINO probability input.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
from pathlib import Path
from types import ModuleType

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .joint_sar_model import JointCromaSAR, _validate_sar
from .ms75_sar_joint_model import align_croma_to_ms

CONVNEXT_SOURCE_SHA256 = "ab969325158df58f4b0957f81f77d543392df61611dc0c72e89ba172ecdf0fb9"
CONVNEXT_CHECKPOINT_SHA256 = "61fa432da5aa0e71bc8e49ce4179f4a4d1d95e4c084a81705c476316bfa3eb33"
CONVNEXT_CHECKPOINT_BYTES = 818663938
MODEL_INPUT_KEYS = ("ms",)
JOINT_INPUT_KEYS = ("ms", "sar_db", "sar_valid", "optical_valid")
IGNORED_VIEW_KEYS = frozenset(("ms_short", "ms_short_valid", "ms_global", "ms_global_valid",
                               "local_box_short", "local_box_global", "ms_valid"))


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def _groups(channels):
    return next(n for n in (8, 4, 2, 1) if channels % n == 0 and channels // n >= 2)


def _conv(cin, cout, kernel=3, stride=1, padding=None):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, stride,
                                   kernel // 2 if padding is None else padding, bias=False),
                         nn.GroupNorm(_groups(cout), cout), nn.GELU())


def _load_convnext(config, load_pretrained):
    source = Path(config.get("convnext_source", Path(__file__).resolve().parents[1]
                             / "vendor/dinov3_convnext/convnext.py"))
    source_bytes = source.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    digest = hashlib.sha256(source_bytes).hexdigest()
    if digest != CONVNEXT_SOURCE_SHA256 or config.get("convnext_source_sha256", digest) != digest:
        raise ValueError("authenticated DINOv3 ConvNeXt source differs")
    module = ModuleType("_authenticated_dinov3_convnext")
    module.__file__ = str(source)
    exec(compile(source_bytes, str(source), "exec"), module.__dict__)
    model = module.ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536],
                           patch_size=None, drop_path_rate=0.)
    if load_pretrained:
        path = Path(config["convnext_checkpoint"])
        expected = config.get("convnext_checkpoint_sha256", config.get("convnext_sha256"))
        if (expected != CONVNEXT_CHECKPOINT_SHA256 or path.stat().st_size != CONVNEXT_CHECKPOINT_BYTES
                or _sha(path) != CONVNEXT_CHECKPOINT_SHA256):
            raise ValueError("DINOv3 ConvNeXt-L checkpoint size/SHA256 differs")
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(state, Mapping):
            raise ValueError("ConvNeXt checkpoint must be its native state dictionary")
        # norm and norms.3 are the SAME official parameter objects. Strict load
        # alone cannot detect conflicting values for these duplicated keys.
        for suffix in ("weight", "bias"):
            if not torch.equal(state["norm." + suffix], state["norms.3." + suffix]):
                raise ValueError("official ConvNeXt final-norm aliases disagree")
        model.load_state_dict(state, strict=True)
    else:
        model.init_weights()
    return model


class OpticalBackbone(nn.Module):
    """Preserve native stage scales/norms; optionally checkpoint each block."""
    def __init__(self, native, gradient_checkpointing=True):
        super().__init__()
        self.native = native
        self.channels = tuple(int(c) for c in native.embed_dims)
        if len(self.channels) != 4 or len(native.downsample_layers) != 4 or len(native.norms) != 4:
            raise ValueError("ConvNeXt must have its original four native stages/norms")
        self.gradient_checkpointing = bool(gradient_checkpointing)

    def forward(self, rgb):
        result = []
        value = rgb
        active = self.training and torch.is_grad_enabled() and self.gradient_checkpointing
        for downsample, stage, norm in zip(self.native.downsample_layers, self.native.stages, self.native.norms):
            value = downsample(value)
            for block in stage:
                value = checkpoint(block, value, use_reentrant=False, preserve_rng_state=True) if active else block(value)
            b, c, h, w = value.shape
            normalized = norm(value.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(b, c, h, w)
            result.append(normalized)
        return tuple(result)


class SpectralPyramid(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = tuple(channels)
        layers, current = [], 4
        for channels in self.channels:
            # kernel2/stride2 centres agree with native ConvNeXt stride4/8/16/32.
            layers.append(nn.Sequential(_conv(current, channels, 2, 2, 0), _conv(channels, channels)))
            current = channels
        self.stages = nn.ModuleList(layers)

    def forward(self, ms):
        features = []
        for stage in self.stages:
            ms = stage(ms)
            features.append(ms)
        return tuple(features)


class PyramidDecoder(nn.Module):
    def __init__(self, optical_channels, spectral_channels, width):
        super().__init__()
        self.laterals = nn.ModuleList(_conv(a+b, width, 1, padding=0)
                                      for a, b in zip(optical_channels, spectral_channels[1:]))
        self.pool_sizes = (1, 2, 3, 6)
        self.pool_projections = nn.ModuleList(_conv(width, width // 4, 1, padding=0) for _ in self.pool_sizes)
        self.pool_fusion = _conv(width * 2, width)
        self.smooth = nn.ModuleList(_conv(width, width) for _ in range(3))
        self.fusion = _conv(width * 4, width)
        self.detail = _conv(width + spectral_channels[0], width)
        self.classifier = nn.Conv2d(width, 24, 1)

    def forward(self, optical, spectral):
        levels = [m(torch.cat((a, b), 1)) for m, a, b in zip(self.laterals, optical, spectral[1:])]
        coarse = levels[-1]
        pooled = [F.interpolate(m(F.adaptive_avg_pool2d(coarse, n)), coarse.shape[-2:],
                                mode="bilinear", align_corners=False)
                  for n, m in zip(self.pool_sizes, self.pool_projections)]
        levels[-1] = self.pool_fusion(torch.cat((coarse, *pooled), 1))
        for i in (2, 1, 0):
            levels[i] = self.smooth[i](levels[i] + F.interpolate(levels[i+1], levels[i].shape[-2:],
                                                               mode="bilinear", align_corners=False))
        size = levels[0].shape[-2:]
        fused = self.fusion(torch.cat([F.interpolate(x, size, mode="bilinear", align_corners=False)
                                      if x.shape[-2:] != size else x for x in levels], 1))
        detail = self.detail(torch.cat((F.interpolate(fused, spectral[0].shape[-2:],
                                                      mode="bilinear", align_corners=False), spectral[0]), 1))
        return F.interpolate(self.classifier(detail), (512, 512), mode="bilinear", align_corners=False)


def fixed_sar_intensity(db, valid, mode="real"):
    """Fixed physical dB scales, no per-crop/channel centering or statistics."""
    _validate_sar(db, valid, mode)
    with torch.autocast(db.device.type, enabled=False):
        if mode == "zero":
            return torch.zeros((len(db), 3, 166, 166), device=db.device, dtype=torch.float32)
        clean = torch.where(valid, db.float(), torch.zeros_like(db, dtype=torch.float32))
        vv_vh = (clean.clamp(-40, 10) + 15) / 25
        difference = ((clean[:, :1] - clean[:, 1:]).clamp(-30, 30)) / 30
        return torch.cat((vv_vh, difference), 1) * valid


class SARInteraction(nn.Module):
    def __init__(self, optical_channels, croma_channels, intensity_channels, width):
        super().__init__()
        self.content_value = nn.Conv2d(croma_channels, width, 1, bias=False)
        self.intensity_value = nn.Conv2d(intensity_channels, width, 1, bias=False)
        self.query = nn.Conv2d(optical_channels, width, 1)
        self.output = nn.Conv2d(width, optical_channels, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, optical, content, intensity, observed, optical_valid):
        size = tuple(optical.shape[-2:])
        value = self.content_value(content) + self.intensity_value(intensity)
        value = align_croma_to_ms(value, size).to(optical.dtype)
        support = F.adaptive_avg_pool2d(optical_valid.float(), size)
        support = support * (align_croma_to_ms(observed.float(), size) > 0)
        delta = self.output(value * self.query(optical).sigmoid()) * support.to(optical.dtype)
        return optical + delta, delta


class ConvNeXtSAR(nn.Module):
    num_classes = 24

    def __init__(self, config, load_pretrained=True, *, backbone=None, sar_encoder=None):
        super().__init__()
        cfg = dict(config)
        cfg.update(config.get("model", {}))
        spectral_channels = tuple(cfg.get("spectral_channels", (32, 48, 96, 128, 192)))
        width = int(cfg.get("decoder_channels", 128))
        interaction_width = int(cfg.get("sar_interaction_width", 64))
        intensity_channels = int(cfg.get("sar_intensity_channels", 32))
        if len(spectral_channels) != 5 or min(spectral_channels) < 2 or width < 8 or width % 4:
            raise ValueError("five spectral levels and decoder width divisible by4 required")
        native = _load_convnext(cfg, load_pretrained) if backbone is None else backbone
        self.backbone = OpticalBackbone(native, cfg.get("gradient_checkpointing", True))
        self.spectral = SpectralPyramid(spectral_channels)
        self.decoder = PyramidDecoder(self.backbone.channels, spectral_channels, width)
        self.optical_aux_head = nn.Conv2d(self.backbone.channels[2], 24, 1)
        self.spectral_aux_head = nn.Conv2d(spectral_channels[1], 24, 1)
        self.sar_encoder = sar_encoder if sar_encoder is not None else JointCromaSAR(cfg["croma_checkpoint"], cfg.get("croma_source"))
        croma_channels = self.sar_encoder.channels[-1]
        # No GroupNorm/InstanceNorm here: a uniform -25dB patch must remain
        # distinguishable from a uniform -5dB patch of identical geometry.
        self.sar_intensity_encoder = nn.Sequential(nn.Conv2d(3, intensity_channels, 3, padding=1), nn.GELU(),
                                                   nn.Conv2d(intensity_channels, intensity_channels, 3, padding=1), nn.GELU())
        self.interactions = nn.ModuleList(SARInteraction(c, croma_channels, intensity_channels, interaction_width)
                                           for c in self.backbone.channels[2:])
        self.sar_aux_head = nn.Conv2d(croma_channels, 24, 1)
        self.register_buffer("ms_mean", torch.tensor(cfg.get("ms_mean", (.363, .390, .371, .506))).view(1, 4, 1, 1))
        self.register_buffer("ms_std", torch.tensor(cfg.get("ms_std", (.231, .231, .244, .254))).view(1, 4, 1, 1))
        self.register_buffer("rgb_mean", torch.tensor((.485, .456, .406)).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor((.229, .224, .225)).view(1, 3, 1, 1))
        if not bool(torch.isfinite(self.ms_mean).all() & torch.isfinite(self.ms_std).all() & (self.ms_std > 0).all()):
            raise ValueError("finite four-band normalization and positive standard deviations required")
        self._backbone_ids = {id(p) for p in self.backbone.parameters()}
        self._sar_tail_ids = {id(p) for p in self.sar_encoder.parameters() if p.requires_grad}
        self._sar_new_ids = {id(p) for m in (self.interactions, self.sar_aux_head, self.sar_intensity_encoder) for p in m.parameters()}
        self._optical_new_ids = {id(p) for m in (self.spectral, self.decoder, self.optical_aux_head, self.spectral_aux_head) for p in m.parameters()}
        self._mutable_ids = self._backbone_ids | self._sar_tail_ids | self._sar_new_ids | self._optical_new_ids
        self.provenance = dict(schema="dinov3_convnext_large_croma_intensity_v1", backbone="official_native_RGB_4stage",
            checkpoint_sha256=CONVNEXT_CHECKPOINT_SHA256, source_sha256=CONVNEXT_SOURCE_SHA256,
            backbone_pretrained_loaded=bool(backbone is None and load_pretrained),
            test_injected_components=bool(backbone is not None or sar_encoder is not None),
            rgb="BGRN_denormalize_then_RGB_ImageNet_normalize", spectral="independent_BGRN_stride2_4_8_16_32",
            interaction_strides=[16, 32], croma=dict(self.sar_encoder.provenance),
            intensity="VV/VH_clip[-40,10]_affine;VV_minus_VH_clip[-30,30]/30;no_per_crop_statistics",
            warmup="optical_new_only;backbone_and_all_SAR_frozen", missing_sar="same_current_optical_path")
        self.set_stage("optical_warmup")

    def rgb_input(self, ms):
        with torch.autocast(ms.device.type, enabled=False):
            reflectance = ms.float() * self.ms_std + self.ms_mean
            return (reflectance[:, (2, 1, 0)] - self.rgb_mean) / self.rgb_std

    def set_stage(self, stage):
        if stage not in ("optical_warmup", "joint"):
            raise ValueError("expected optical_warmup or joint")
        self.stage = stage
        active = self._optical_new_ids if stage == "optical_warmup" else self._mutable_ids
        for p in self.parameters():
            p.requires_grad_(id(p) in active)
        self.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "stage"):
            self.backbone.train(mode and self.stage == "joint")
            sar_mode = mode and self.stage == "joint"
            for m in (self.sar_encoder, self.sar_intensity_encoder, self.interactions, self.sar_aux_head):
                m.train(sar_mode)
            for m in self.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    def mutable_keys(self):
        state = self.state_dict()
        # Include ALL aliases, especially official norm == norms.3; named_parameters()
        # defaults to removing duplicates and would silently omit those state keys.
        names = {n for n, p in self.named_parameters(remove_duplicate=False) if id(p) in self._mutable_ids}
        buffers = {n for n, _ in self.named_buffers(remove_duplicate=False)} & set(state)
        return tuple(sorted(names | buffers))

    def parameter_groups(self, optical_lr=1e-5, sar_lr=1e-4, tail_lr=3e-6, weight_decay=.01):
        if any(not math.isfinite(v) or v <= 0 for v in (optical_lr, sar_lr, tail_lr)) or not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("invalid learning rates/decay")
        bins = {}
        for name, p in self.named_parameters():
            if id(p) not in self._mutable_ids:
                continue
            label, rate = (("optical_backbone", optical_lr) if id(p) in self._backbone_ids else
                           ("sar_tail", tail_lr) if id(p) in self._sar_tail_ids else
                           ("sar_new", sar_lr) if id(p) in self._sar_new_ids else ("optical_new", sar_lr))
            decay = p.ndim > 1 and not name.endswith(".bias")
            bins.setdefault((label, rate, decay), []).append((name, p))
        groups = [dict(params=[p for _,p in rows], parameter_names=[n for n,_ in rows],
                       lr=rate, initial_lr=rate, weight_decay=weight_decay if decay else 0.,
                       group_name=label + ("_decay" if decay else "_no_decay"))
                  for (label, rate, decay), rows in sorted(bins.items())]
        ids = [id(p) for group in groups for p in group["params"]]
        if len(ids) != len(set(ids)) or set(ids) != self._mutable_ids:
            raise RuntimeError("optimizer stage-union inventory differs")
        return groups

    @torch.no_grad()
    def project_constrained_parameters_(self):
        return self

    def _optical_features(self, ms):
        if ms.ndim != 4 or tuple(ms.shape[1:]) != (4, 512, 512):
            raise ValueError("native Bx4x512x512 MS required")
        optical = self.backbone(self.rgb_input(ms))
        spectral = self.spectral(ms)
        if [tuple(x.shape[-2:]) for x in optical] != [(128,128),(64,64),(32,32),(16,16)]:
            raise ValueError("native ConvNeXt stride4/8/16/32 geometry changed")
        return optical, spectral

    def _decode(self, optical, spectral, original):
        return dict(logits=self.decoder(optical, spectral),
                    optical_aux_logits=F.interpolate(self.optical_aux_head(original[2]), (512,512), mode="bilinear", align_corners=False),
                    spectral_aux_logits=F.interpolate(self.spectral_aux_head(spectral[1]), (512,512), mode="bilinear", align_corners=False))

    def forward_optical(self, *, ms, **unused):
        optical, spectral = self._optical_features(ms)
        return self._decode(optical, spectral, optical)

    def _encode_pair(self, db, valid, mode):
        cpu_state = torch.random.get_rng_state()
        devices = [db.device.index if db.device.index is not None else torch.cuda.current_device()] if db.is_cuda else []
        device_state = torch.cuda.get_rng_state(devices[0]) if devices else None
        real = self.sar_encoder(db, valid, mode)[-1]
        with torch.random.fork_rng(devices=devices):
            torch.random.set_rng_state(cpu_state)
            if devices:
                torch.cuda.set_rng_state(device_state, devices[0])
            reference = self.sar_encoder(db, valid, "zero")[-1]
        return real, real - reference

    def forward(self, *, ms, sar_db, sar_valid, optical_valid, sar_mode="real", **unused):
        if set(unused) - IGNORED_VIEW_KEYS:
            raise ValueError("unexpected model inputs: " + ",".join(sorted(set(unused) - IGNORED_VIEW_KEYS)))
        _validate_sar(sar_db, sar_valid, sar_mode)
        if (len(ms) != len(sar_db) or ms.device != sar_db.device or optical_valid.shape != (len(ms),1,512,512)
                or optical_valid.dtype != torch.bool or optical_valid.device != ms.device):
            raise ValueError("matching native MS/SAR batch, device and boolean optical SAR support required")
        optical, spectral = self._optical_features(ms)
        present = sar_valid.flatten(1).any(1) & optical_valid.flatten(1).any(1)
        active = not (self.training and self.stage == "optical_warmup")
        if active:
            real, content = self._encode_pair(sar_db, sar_valid, sar_mode)
            observed = F.avg_pool2d(F.pad(sar_valid.float(), (1,1,1,1)), 8, 8) > 0
            intensity = self.sar_intensity_encoder(fixed_sar_intensity(sar_db, sar_valid, sar_mode))
            intensity = F.avg_pool2d(F.pad(intensity * sar_valid, (1,1,1,1)), 8, 8)
            intensity = intensity * observed * (1. if sar_mode == "real" else 0.)
            fused, deltas = list(optical), []
            for index, interaction in zip((2,3), self.interactions):
                fused[index], delta = interaction(optical[index], content, intensity, observed,
                                                  optical_valid & present[:,None,None,None])
                deltas.append(delta)
            result = self._decode(tuple(fused), spectral, optical)
            result.update(sar_aux_logits=self.sar_aux_head(real), sar_aux_valid=observed,
                          sar_content_difference=content, sar_intensity_feature=intensity,
                          sar_stride16_delta=deltas[0], sar_stride32_delta=deltas[1])
        else:
            result = self._decode(optical, spectral, optical)
            result.update(sar_aux_logits=ms.new_zeros((len(ms),24,21,21)),
                          sar_aux_valid=torch.zeros((len(ms),1,21,21),device=ms.device,dtype=torch.bool),
                          sar_stride16_delta=torch.zeros_like(optical[2]), sar_stride32_delta=torch.zeros_like(optical[3]))
        result["sar_present"] = present
        return result

    def forward_batch(self, batch):
        return self(**{key: batch[key] for key in JOINT_INPUT_KEYS})


__all__ = ["ConvNeXtSAR", "OpticalBackbone", "fixed_sar_intensity", "SARInteraction",
           "MODEL_INPUT_KEYS", "JOINT_INPUT_KEYS"]
