"""DINOv3 ViT-L/16 SAT-493M expert with a gated NIR patch adapter.

The official satellite checkpoint is loaded into its native three-channel
RGB backbone before the patch projection is wrapped.  The wrapper preserves
the pretrained RGB convolution byte-for-byte and adds a separately trainable
one-channel NIR convolution initialized from the RGB-kernel mean.  The FBP
reader supplies normalized B/G/R/NIR crops; visible reflectance is reordered
to RGB and normalized with the official SAT-493M statistics, while NIR keeps
the FBP normalization.

Unlike DINOv2-B/14, DINOv3 ViT-L is an EVA-style patch-16 model with RoPE and
five prefix tokens (CLS plus four storage/register tokens).  Intermediate
features are therefore obtained through timm's native ``forward_intermediates``
path so RoPE, prefix removal, and activation checkpointing remain correct.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .dinov2_nir_model import DINOv2NIRDPTDecoder


DINOV3_SAT_VITL16_SHA256 = (
    "eadcf0ffc02418b6c22a885ea1a7aaeeef84fbf0f5bb4d0b7d1d36e68a964f48"
)
DINOV3_SAT_RGB_MEAN = (0.430, 0.411, 0.296)
DINOV3_SAT_RGB_STD = (0.213, 0.156, 0.143)
DINOV3_SAT_BACKBONE_PARAMETERS = 303_079_424


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_tuple(
    values: Sequence[float], *, length: int, name: str, positive: bool = False
) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != length or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain {length} finite values")
    if positive and any(value <= 0.0 for value in result):
        raise ValueError(f"{name} values must be positive")
    return result


class GatedRGBNIRPatchProjection(nn.Module):
    """Native RGB patch projection plus an independently gated NIR branch."""

    def __init__(self, rgb_projection: nn.Conv2d, *, initial_gate: float) -> None:
        super().__init__()
        if not isinstance(rgb_projection, nn.Conv2d):
            raise TypeError("DINOv3 RGB patch projection must be Conv2d")
        if rgb_projection.in_channels != 3:
            raise ValueError("DINOv3 RGB projection must retain three input channels")
        if tuple(rgb_projection.kernel_size) != (16, 16):
            raise ValueError("DINOv3 SAT backbone must use 16x16 patches")
        if tuple(rgb_projection.stride) != (16, 16):
            raise ValueError("DINOv3 SAT patch stride must be 16x16")
        if not 0.0 < float(initial_gate) < 1.0:
            raise ValueError("initial NIR gate must lie in (0, 1)")
        self.rgb_projection = rgb_projection
        self.nir_projection = nn.Conv2d(
            1,
            rgb_projection.out_channels,
            kernel_size=rgb_projection.kernel_size,
            stride=rgb_projection.stride,
            padding=rgb_projection.padding,
            dilation=rgb_projection.dilation,
            groups=1,
            bias=rgb_projection.bias is not None,
            padding_mode=rgb_projection.padding_mode,
        )
        with torch.no_grad():
            self.nir_projection.weight.copy_(
                rgb_projection.weight.detach().mean(dim=1, keepdim=True)
            )
            if self.nir_projection.bias is not None:
                self.nir_projection.bias.zero_()
        initial_logit = math.log(float(initial_gate) / (1.0 - float(initial_gate)))
        self.nir_gate_logit = nn.Parameter(
            torch.tensor(initial_logit, dtype=torch.float32)
        )

    @property
    def nir_gate(self) -> torch.Tensor:
        return torch.sigmoid(self.nir_gate_logit)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != 4:
            raise ValueError("DINOv3 patch projection expects normalized RGB+NIR")
        rgb = self.rgb_projection(value[:, :3])
        nir = self.nir_projection(value[:, 3:4])
        return rgb + self.nir_gate.to(dtype=rgb.dtype) * nir


def _build_timm_backbone(
    *,
    model_name: str,
    image_size: int,
    pretrained: bool,
    pretrained_file: str | None,
    pretrained_file_sha256: str | None,
) -> nn.Module:
    try:
        import timm
    except ImportError as error:  # pragma: no cover - deployment dependency
        raise RuntimeError("DINOv3SATNIR requires timm") from error

    if str(model_name) != "vit_large_patch16_dinov3.sat493m":
        raise ValueError("DINOv3SATNIR requires the tagged SAT-493M ViT-L/16")
    if not bool(pretrained):
        raise ValueError("formal DINOv3SATNIR construction requires pretraining")
    if pretrained_file is None:
        raise ValueError("DINOv3SATNIR requires a pinned local pretrained file")
    path = Path(pretrained_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing DINOv3 SAT pretrained file: {path}")
    expected = str(pretrained_file_sha256 or "").lower()
    if expected != DINOV3_SAT_VITL16_SHA256:
        raise ValueError("DINOv3 SAT pretrained SHA256 is not the pinned identity")
    observed = _sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"DINOv3 SAT pretrained SHA256 mismatch: expected={expected}, "
            f"observed={observed}"
        )
    backbone = timm.create_model(
        str(model_name),
        pretrained=True,
        pretrained_cfg_overlay={"file": str(path)},
        img_size=int(image_size),
        num_classes=0,
    )
    parameter_count = sum(parameter.numel() for parameter in backbone.parameters())
    if parameter_count != DINOV3_SAT_BACKBONE_PARAMETERS:
        raise RuntimeError(
            "DINOv3 SAT backbone parameter identity changed: "
            f"expected={DINOV3_SAT_BACKBONE_PARAMETERS}, observed={parameter_count}"
        )
    return backbone


class DINOv3SATNIRSegmentation(nn.Module):
    """Satellite-pretrained DINOv3-L expert for 24-class pure-MS mapping."""

    def __init__(
        self,
        *,
        num_classes: int = 24,
        backbone_model: str = "vit_large_patch16_dinov3.sat493m",
        pretrained: bool = True,
        pretrained_file: str | None = None,
        pretrained_file_sha256: str | None = None,
        input_size: int = 512,
        patch_size: int = 16,
        feature_indices: Sequence[int] = (5, 11, 17, 23),
        reassemble_channels: Sequence[int] = (96, 192, 384, 768),
        decoder_channels: int = 256,
        initial_nir_gate: float = 0.1,
        ms_mean: Sequence[float] = (0.363, 0.390, 0.371, 0.506),
        ms_std: Sequence[float] = (0.231, 0.231, 0.244, 0.254),
        rgb_mean: Sequence[float] = DINOV3_SAT_RGB_MEAN,
        rgb_std: Sequence[float] = DINOV3_SAT_RGB_STD,
        gradient_checkpointing: bool = True,
        freeze_prefix_tokens: bool = True,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if int(num_classes) != 24:
            raise ValueError("DINOv3SATNIR is defined for exactly 24 classes")
        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        if self.input_size != 512 or self.patch_size != 16:
            raise ValueError("DINOv3SATNIR requires 512 crops and patch size 16")
        if self.input_size % self.patch_size:
            raise ValueError("DINOv3SATNIR input must divide evenly into patches")
        indices = tuple(int(value) for value in feature_indices)
        if indices != (5, 11, 17, 23):
            raise ValueError("DINOv3SATNIR features must be blocks 5,11,17,23")
        self.feature_indices = indices
        self.num_classes = int(num_classes)

        ms_mean_values = _finite_tuple(ms_mean, length=4, name="ms_mean")
        ms_std_values = _finite_tuple(ms_std, length=4, name="ms_std", positive=True)
        rgb_mean_values = _finite_tuple(rgb_mean, length=3, name="rgb_mean")
        rgb_std_values = _finite_tuple(rgb_std, length=3, name="rgb_std", positive=True)
        self.register_buffer(
            "ms_mean",
            torch.tensor(ms_mean_values, dtype=torch.float32).reshape(1, 4, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "ms_std",
            torch.tensor(ms_std_values, dtype=torch.float32).reshape(1, 4, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "rgb_mean",
            torch.tensor(rgb_mean_values, dtype=torch.float32).reshape(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor(rgb_std_values, dtype=torch.float32).reshape(1, 3, 1, 1),
            persistent=True,
        )

        self.backbone = backbone or _build_timm_backbone(
            model_name=str(backbone_model),
            image_size=self.input_size,
            pretrained=bool(pretrained),
            pretrained_file=pretrained_file,
            pretrained_file_sha256=pretrained_file_sha256,
        )
        patch_embed = getattr(self.backbone, "patch_embed", None)
        rgb_projection = getattr(patch_embed, "proj", None)
        if not isinstance(rgb_projection, nn.Conv2d):
            raise ValueError("DINOv3 backbone lacks its native RGB patch projection")
        if len(getattr(self.backbone, "blocks", ())) != 24:
            raise ValueError("DINOv3SATNIR requires exactly 24 transformer blocks")
        if int(getattr(self.backbone, "embed_dim", -1)) != 1024:
            raise ValueError("DINOv3SATNIR requires a 1024-dimensional backbone")
        if int(getattr(self.backbone, "num_prefix_tokens", -1)) != 5:
            raise ValueError("DINOv3SATNIR requires CLS plus four register tokens")
        if getattr(self.backbone, "rope", None) is None:
            raise ValueError("DINOv3SATNIR backbone must retain RoPE")

        patch_embed.proj = GatedRGBNIRPatchProjection(
            rgb_projection, initial_gate=float(initial_nir_gate)
        )
        if bool(freeze_prefix_tokens):
            for name in ("cls_token", "reg_token"):
                parameter = getattr(self.backbone, name, None)
                if isinstance(parameter, nn.Parameter):
                    parameter.requires_grad_(False)
        set_checkpointing = getattr(self.backbone, "set_grad_checkpointing", None)
        if not callable(set_checkpointing):
            raise ValueError("DINOv3 backbone lacks gradient-checkpointing support")
        set_checkpointing(bool(gradient_checkpointing))
        self.decoder = DINOv2NIRDPTDecoder(
            input_channels=1024,
            num_classes=self.num_classes,
            reassemble_channels=reassemble_channels,
            fusion_channels=int(decoder_channels),
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "DINOv3SATNIRSegmentation":
        model = config["model"]
        normalization = config["data"]["normalization"]["ms"]
        return cls(
            num_classes=int(model["num_classes"]),
            backbone_model=str(model["backbone_model"]),
            pretrained=bool(model["pretrained"]),
            pretrained_file=str(model["pretrained_file"]),
            pretrained_file_sha256=str(model["pretrained_file_sha256"]),
            input_size=int(model["input_size"]),
            patch_size=int(model["patch_size"]),
            feature_indices=model["feature_indices"],
            reassemble_channels=model["reassemble_channels"],
            decoder_channels=int(model["decoder_channels"]),
            initial_nir_gate=float(model["initial_nir_gate"]),
            ms_mean=normalization["mean"],
            ms_std=normalization["std"],
            rgb_mean=model["rgb_normalization"]["mean"],
            rgb_std=model["rgb_normalization"]["std"],
            gradient_checkpointing=bool(model["gradient_checkpointing"]),
            freeze_prefix_tokens=bool(model["freeze_prefix_tokens"]),
        )

    @property
    def patch_projection(self) -> GatedRGBNIRPatchProjection:
        projection = self.backbone.patch_embed.proj
        if not isinstance(projection, GatedRGBNIRPatchProjection):
            raise RuntimeError("DINOv3 gated patch projection was replaced")
        return projection

    @property
    def nir_gate(self) -> torch.Tensor:
        return self.patch_projection.nir_gate

    def prepare_inputs(self, ms: torch.Tensor) -> torch.Tensor:
        if ms.ndim != 4 or ms.shape[1] != 4:
            raise ValueError("DINOv3SATNIR input must have shape Bx4x512x512")
        if tuple(ms.shape[-2:]) != (self.input_size, self.input_size):
            raise ValueError("DINOv3SATNIR input crop must be exactly 512x512")
        if not ms.is_floating_point():
            raise TypeError("DINOv3SATNIR input must be floating point")
        ms_mean = self.ms_mean.to(device=ms.device, dtype=ms.dtype)
        ms_std = self.ms_std.to(device=ms.device, dtype=ms.dtype)
        reflectance = ms * ms_std + ms_mean
        rgb = reflectance[:, (2, 1, 0)]
        rgb_mean = self.rgb_mean.to(device=ms.device, dtype=ms.dtype)
        rgb_std = self.rgb_std.to(device=ms.device, dtype=ms.dtype)
        rgb = (rgb - rgb_mean) / rgb_std
        nir = (reflectance[:, 3:4] - ms_mean[:, 3:4]) / ms_std[:, 3:4]
        return torch.cat((rgb, nir), dim=1)

    def _backbone_features(self, normalized_rgb_nir: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self.backbone.forward_intermediates(
            normalized_rgb_nir,
            indices=list(self.feature_indices),
            norm=True,
            output_fmt="NCHW",
            intermediates_only=True,
        )
        if not isinstance(features, (list, tuple)) or len(features) != 4:
            raise RuntimeError("DINOv3 did not expose all four DPT features")
        grid = self.input_size // self.patch_size
        result = tuple(features)
        for feature in result:
            if tuple(feature.shape[1:]) != (1024, grid, grid):
                raise RuntimeError(
                    "DINOv3 intermediate feature geometry changed: "
                    f"observed={tuple(feature.shape)}"
                )
        return result

    def forward(
        self,
        ms: torch.Tensor,
        sar: torch.Tensor | None = None,
        sar_valid: torch.Tensor | None = None,
        time_gap_days: torch.Tensor | None = None,
        ms_context: torch.Tensor | None = None,
        ms_context_valid: torch.Tensor | None = None,
        **unused: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del sar, sar_valid, time_gap_days, ms_context, ms_context_valid, unused
        value = self.prepare_inputs(ms)
        features = self._backbone_features(value)
        logits, decoder_embedding = self.decoder(
            features, output_size=(self.input_size, self.input_size)
        )
        if tuple(logits.shape[-2:]) != (self.input_size, self.input_size):
            raise RuntimeError("DINOv3SATNIR output is not 512x512")
        return {
            "logits": logits,
            "ms_logits": logits,
            "flat_logits": logits,
            "ms_shared": decoder_embedding,
            "sar_gate": logits.new_zeros((logits.shape[0], 1, 1, 1)),
            "sar_scale": logits.new_zeros(1),
            "nir_gate": self.nir_gate.reshape(1),
        }

    def predict_proba(self, ms: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        return self.forward(ms, **kwargs)["logits"].softmax(dim=1)


__all__ = [
    "DINOV3_SAT_BACKBONE_PARAMETERS",
    "DINOV3_SAT_RGB_MEAN",
    "DINOV3_SAT_RGB_STD",
    "DINOV3_SAT_VITL16_SHA256",
    "DINOv3SATNIRSegmentation",
    "GatedRGBNIRPatchProjection",
]
