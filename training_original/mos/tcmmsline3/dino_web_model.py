"""Pinned web-pretrained DINO experts with independent NIR adapters and DPT.

All variants keep timm's official checkpoint filter and native intermediate
feature path. In particular, DINOv3's storage tokens and RoPE are not manually
reimplemented. The pretrained three-channel projection is loaded *before* it
is wrapped, so pretrained RGB tensors remain unchanged.

Web RGB normalization is the official ImageNet normalization (not SAT-493M):
https://github.com/facebookresearch/dinov3#image-transforms
https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py

Freezing/staged unfreezing is owned by the trainer. The wrapper deliberately
keeps the same RGB/NIR module names for all three variants.

All three new runs explicitly disable the autocast weight cache within model
forward, preserving the caller's dtype/enabled state. This is a shared policy
for the new checkpointed/staged experiments, not a claim of a historical bug
or a performance improvement.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .dinov2_nir_model import DINOv2NIRDPTDecoder


DINO_WEB_RGB_MEAN = (0.485, 0.456, 0.406)
DINO_WEB_RGB_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DinoWebVariantSpec:
    backbone_model: str
    filename: str
    sha256: str
    depth: int
    embed_dim: int
    patch_size: int
    padded_size: int
    feature_indices: tuple[int, int, int, int]
    num_prefix_tokens: int
    backbone_parameters: int


DINO_WEB_VARIANTS = MappingProxyType(
    {
        "v3b": DinoWebVariantSpec(
            "vit_base_patch16_dinov3.lvd1689m",
            "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
            "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c",
            12,
            768,
            16,
            512,
            (2, 5, 8, 11),
            5,
            85_641_216,
        ),
        "v3l": DinoWebVariantSpec(
            "vit_large_patch16_dinov3.lvd1689m",
            "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
            "8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035",
            24,
            1024,
            16,
            512,
            (5, 11, 17, 23),
            5,
            303_079_424,
        ),
        "v2l": DinoWebVariantSpec(
            "vit_large_patch14_dinov2.lvd142m",
            "dinov2_vitl14_pretrain.pth",
            "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428",
            24,
            1024,
            14,
            518,
            (5, 11, 17, 23),
            1,
            304_367_616,
        ),
    }
)


def _variant_spec(variant: str) -> DinoWebVariantSpec:
    if variant not in DINO_WEB_VARIANTS:
        raise ValueError(f"DINO web variant must be v3b, v3l, or v2l: {variant!r}")
    return DINO_WEB_VARIANTS[variant]


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
    if positive and any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive values")
    return result


def _current_autocast_settings(device_type: str) -> tuple[bool, torch.dtype]:
    """Read current AMP state, including the PyTorch 2.1 device-specific API."""
    get_dtype = getattr(torch, "get_autocast_dtype", None)
    if callable(get_dtype):
        return bool(torch.is_autocast_enabled(device_type)), get_dtype(device_type)
    if device_type == "cuda":  # PyTorch 2.1 compatibility
        return bool(torch.is_autocast_enabled()), torch.get_autocast_gpu_dtype()
    if device_type == "cpu":
        return bool(torch.is_autocast_cpu_enabled()), torch.get_autocast_cpu_dtype()
    raise ValueError(f"DINO web AMP state query does not support {device_type!r}")


def _build_timm_backbone(
    *,
    spec: DinoWebVariantSpec,
    pretrained: bool,
    pretrained_file: str | None,
    pretrained_file_sha256: str | None,
    load_pretrained: bool = True,
) -> nn.Module:
    # Fail closed before importing timm or invoking any of its weight resolvers.
    if load_pretrained and not pretrained:
        raise ValueError("DINO web construction requires pinned local pretraining")
    if load_pretrained and not pretrained_file:
        raise ValueError("DINO web requires a pinned local pretrained_file")
    path = None
    if load_pretrained:
        expected = str(pretrained_file_sha256 or "").lower()
        if expected != spec.sha256:
            raise ValueError(
                "DINO web pretrained SHA256 is not the pinned variant identity"
            )
        path = Path(pretrained_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing DINO web pretrained file: {path}")
        observed = _sha256_file(path)
        if observed != expected:
            raise ValueError(
                f"DINO web pretrained SHA256 mismatch: expected={expected}, observed={observed}"
            )
    try:
        import timm
    except ImportError as error:  # pragma: no cover - deployment dependency
        raise RuntimeError("DINO web requires timm with DINOv3 support") from error
    # Do not use checkpoint_path or load_state_dict(raw_official_weights): those
    # bypass timm's DINO filters (storage_tokens, layer scales, mask token, etc.).
    # Clear remote fallbacks as defense in depth; file is the only weight source.
    backbone = timm.create_model(
        spec.backbone_model,
        pretrained=bool(load_pretrained),
        pretrained_cfg_overlay={
            "file": str(path) if path is not None else "",
            "url": "",
            "hf_hub_id": "",
            "source": "timm",
        },
        img_size=spec.padded_size,
        num_classes=0,
    )
    count = sum(parameter.numel() for parameter in backbone.parameters())
    if count != spec.backbone_parameters:
        raise RuntimeError(
            f"DINO web backbone parameter identity changed: expected={spec.backbone_parameters}, observed={count}"
        )
    return backbone


class GatedRGBNIRPatchProjection(nn.Module):
    """Native RGB convolution plus separately trainable mean-initialized NIR."""

    def __init__(self, rgb_projection: nn.Conv2d, *, initial_gate: float) -> None:
        super().__init__()
        if not isinstance(rgb_projection, nn.Conv2d) or rgb_projection.in_channels != 3:
            raise ValueError("DINO web requires a native three-channel RGB Conv2d")
        if rgb_projection.groups != 1:
            raise ValueError("DINO web RGB patch projection must have groups=1")
        if not 0 < float(initial_gate) < 1:
            raise ValueError("initial NIR gate must lie in (0, 1)")
        self.rgb_projection = rgb_projection
        self.nir_projection = nn.Conv2d(
            1,
            rgb_projection.out_channels,
            kernel_size=rgb_projection.kernel_size,
            stride=rgb_projection.stride,
            padding=rgb_projection.padding,
            dilation=rgb_projection.dilation,
            bias=rgb_projection.bias is not None,
            padding_mode=rgb_projection.padding_mode,
            device=rgb_projection.weight.device,
            dtype=rgb_projection.weight.dtype,
        )
        with torch.no_grad():
            self.nir_projection.weight.copy_(
                rgb_projection.weight.detach().mean(dim=1, keepdim=True)
            )
            if self.nir_projection.bias is not None:
                self.nir_projection.bias.zero_()
        self.nir_gate_logit = nn.Parameter(
            torch.tensor(
                math.log(float(initial_gate) / (1 - float(initial_gate))),
                dtype=torch.float32,
                device=rgb_projection.weight.device,
            )
        )

    @property
    def nir_gate(self) -> torch.Tensor:
        return self.nir_gate_logit.sigmoid()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != 4:
            raise ValueError("DINO web patch projection expects normalized RGB+NIR")
        rgb = self.rgb_projection(value[:, :3])
        nir = self.nir_projection(value[:, 3:4])
        return rgb + self.nir_gate.to(dtype=rgb.dtype) * nir


class DinoWebSegmentation(nn.Module):
    """DINOv3-B/L or DINOv2-L with a shared 24-class NIR/DPT contract."""

    def __init__(
        self,
        *,
        variant: str,
        num_classes: int = 24,
        backbone_model: str | None = None,
        pretrained: bool = True,
        pretrained_file: str | None = None,
        pretrained_file_sha256: str | None = None,
        input_size: int = 512,
        padded_size: int | None = None,
        patch_size: int | None = None,
        feature_indices: Sequence[int] | None = None,
        reassemble_channels: Sequence[int] = (96, 192, 384, 768),
        decoder_channels: int = 256,
        initial_nir_gate: float = 0.1,
        ms_mean: Sequence[float] = (0.363, 0.390, 0.371, 0.506),
        ms_std: Sequence[float] = (0.231, 0.231, 0.244, 0.254),
        rgb_mean: Sequence[float] = DINO_WEB_RGB_MEAN,
        rgb_std: Sequence[float] = DINO_WEB_RGB_STD,
        gradient_checkpointing: bool = True,
        freeze_prefix_tokens: bool | None = None,
        backbone: nn.Module | None = None,
        load_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.variant = str(variant)
        spec = _variant_spec(self.variant)
        self.spec = spec
        if int(num_classes) != 24 or int(input_size) != 512:
            raise ValueError("DINO web requires exactly 24 classes and 512x512 crops")
        accepted_names = {spec.backbone_model}
        if self.variant == "v2l":
            accepted_names.add("vit_large_patch14_dinov2")
        if backbone_model is not None and backbone_model not in accepted_names:
            raise ValueError(
                "DINO web backbone_model does not match the pinned variant"
            )
        for name, supplied, expected in (
            ("padded_size", padded_size, spec.padded_size),
            ("patch_size", patch_size, spec.patch_size),
        ):
            if supplied is not None and int(supplied) != expected:
                raise ValueError(
                    f"DINO web {name} must be {expected} for {self.variant}"
                )
        if (
            feature_indices is not None
            and tuple(feature_indices) != spec.feature_indices
        ):
            raise ValueError(f"DINO web feature_indices must be {spec.feature_indices}")
        self.num_classes = int(num_classes)
        self.input_size = int(input_size)
        self.padded_size = spec.padded_size
        self.patch_size = spec.patch_size
        self.pad_pixels = (self.padded_size - self.input_size) // 2
        self.feature_indices = spec.feature_indices
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.autocast_cache_enabled = False
        means = _finite_tuple(rgb_mean, length=3, name="rgb_mean")
        stds = _finite_tuple(rgb_std, length=3, name="rgb_std", positive=True)
        if means != DINO_WEB_RGB_MEAN or stds != DINO_WEB_RGB_STD:
            raise ValueError(
                "DINO web RGB normalization must match official ImageNet statistics, not SAT"
            )
        for name, values in (
            ("ms_mean", _finite_tuple(ms_mean, length=4, name="ms_mean")),
            ("ms_std", _finite_tuple(ms_std, length=4, name="ms_std", positive=True)),
            ("rgb_mean", means),
            ("rgb_std", stds),
        ):
            self.register_buffer(
                name, torch.tensor(values, dtype=torch.float32).reshape(1, -1, 1, 1)
            )

        # Explicit backbone injection is a CPU-test seam. Production config
        # construction cannot bypass local checkpoint authentication.
        self.backbone = (
            backbone
            if backbone is not None
            else _build_timm_backbone(
                spec=spec,
                pretrained=bool(pretrained),
                pretrained_file=pretrained_file,
                pretrained_file_sha256=pretrained_file_sha256,
                load_pretrained=load_pretrained,
            )
        )
        patch_embed = getattr(self.backbone, "patch_embed", None)
        projection = getattr(patch_embed, "proj", None)
        if not isinstance(projection, nn.Conv2d) or projection.in_channels != 3:
            raise ValueError("DINO web backbone lacks its native RGB patch projection")
        if (
            projection.kernel_size != (spec.patch_size,) * 2
            or projection.stride != (spec.patch_size,) * 2
        ):
            raise ValueError("DINO web patch geometry does not match the variant")
        if len(getattr(self.backbone, "blocks", ())) != spec.depth:
            raise ValueError("DINO web backbone depth does not match the variant")
        if (
            int(getattr(self.backbone, "embed_dim", -1)) != spec.embed_dim
            or projection.out_channels != spec.embed_dim
        ):
            raise ValueError(
                "DINO web backbone embedding dimension does not match the variant"
            )
        if (
            int(getattr(self.backbone, "num_prefix_tokens", -1))
            != spec.num_prefix_tokens
        ):
            raise ValueError(
                "DINO web backbone prefix token identity does not match the variant"
            )
        if (
            self.variant.startswith("v3")
            and getattr(self.backbone, "rope", None) is None
        ):
            raise ValueError("DINOv3 web backbone must retain native RoPE")
        if not callable(getattr(self.backbone, "forward_intermediates", None)):
            raise ValueError("DINO web backbone requires native forward_intermediates")
        set_checkpointing = getattr(self.backbone, "set_grad_checkpointing", None)
        if not callable(set_checkpointing):
            raise ValueError(
                "DINO web backbone requires gradient-checkpointing support"
            )
        patch_embed.proj = GatedRGBNIRPatchProjection(
            projection, initial_gate=initial_nir_gate
        )
        self.freeze_prefix_tokens = (
            self.variant.startswith("v3")
            if freeze_prefix_tokens is None
            else bool(freeze_prefix_tokens)
        )
        if self.freeze_prefix_tokens:
            for name in ("cls_token", "reg_token"):
                parameter = getattr(self.backbone, name, None)
                if isinstance(parameter, nn.Parameter):
                    parameter.requires_grad_(False)
        set_checkpointing(self.gradient_checkpointing)
        self.decoder = DINOv2NIRDPTDecoder(
            input_channels=spec.embed_dim,
            num_classes=self.num_classes,
            reassemble_channels=reassemble_channels,
            fusion_channels=int(decoder_channels),
        )

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], *, load_pretrained: bool = True
    ) -> "DinoWebSegmentation":
        model = config["model"]
        if model.get("architecture") != "dino_web_nir_dpt":
            raise ValueError("DINO web requires architecture='dino_web_nir_dpt'")
        if model.get("fusion_mode", "ms_only") != "ms_only":
            raise ValueError("DINO web is a pure-MS expert")
        if tuple(model.get("input_band_order", ("B", "G", "R", "NIR"))) != (
            "B",
            "G",
            "R",
            "NIR",
        ):
            raise ValueError("DINO web input band order must be B/G/R/NIR")
        if (
            model.get("nir_initialization", "rgb_patch_weight_mean")
            != "rgb_patch_weight_mean"
        ):
            raise ValueError("DINO web requires RGB-kernel-mean NIR initialization")
        if model.get("autocast_cache_enabled", False) is not False:
            raise ValueError("DINO web requires model.autocast_cache_enabled=False")
        normalization = config["data"]["normalization"]["ms"]
        arguments = {
            name: model[name]
            for name in (
                "variant",
                "num_classes",
                "backbone_model",
                "pretrained",
                "pretrained_file",
                "pretrained_file_sha256",
                "input_size",
                "padded_size",
                "patch_size",
                "feature_indices",
                "reassemble_channels",
                "decoder_channels",
                "initial_nir_gate",
                "gradient_checkpointing",
                "freeze_prefix_tokens",
            )
            if name in model
        }
        rgb = model.get("rgb_normalization", {})
        return cls(
            **arguments,
            load_pretrained=load_pretrained,
            ms_mean=normalization["mean"],
            ms_std=normalization["std"],
            rgb_mean=rgb.get("mean", DINO_WEB_RGB_MEAN),
            rgb_std=rgb.get("std", DINO_WEB_RGB_STD),
        )

    @property
    def patch_projection(self) -> GatedRGBNIRPatchProjection:
        projection = self.backbone.patch_embed.proj
        if not isinstance(projection, GatedRGBNIRPatchProjection):
            raise RuntimeError("DINO web NIR patch adapter was replaced")
        return projection

    @property
    def nir_gate(self) -> torch.Tensor:
        return self.patch_projection.nir_gate

    def prepare_inputs(self, ms: torch.Tensor) -> torch.Tensor:
        if ms.ndim != 4 or ms.shape[1] != 4 or tuple(ms.shape[-2:]) != (512, 512):
            raise ValueError("DINO web input must have shape Bx4x512x512")
        if not ms.is_floating_point():
            raise TypeError("DINO web input must be floating point")
        reflectance = ms * self.ms_std.to(ms) + self.ms_mean.to(ms)
        rgb = (reflectance[:, (2, 1, 0)] - self.rgb_mean.to(ms)) / self.rgb_std.to(ms)
        # Retain the reader's independent FBP NIR normalization exactly.
        result = torch.cat((rgb, ms[:, 3:4]), dim=1)
        if self.pad_pixels:
            result = F.pad(result, (self.pad_pixels,) * 4, mode="reflect")
        return result

    def _backbone_features(self, rgb_nir: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self.backbone.forward_intermediates(
            rgb_nir,
            indices=list(self.feature_indices),
            norm=True,
            output_fmt="NCHW",
            intermediates_only=True,
        )
        if not isinstance(features, (tuple, list)) or len(features) != 4:
            raise RuntimeError("DINO web did not expose all four DPT features")
        grid = self.padded_size // self.patch_size
        for feature in features:
            if tuple(feature.shape) != (
                rgb_nir.shape[0],
                self.spec.embed_dim,
                grid,
                grid,
            ):
                raise RuntimeError(
                    f"DINO web intermediate feature geometry changed: {tuple(feature.shape)}"
                )
        return tuple(features)

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
        enabled, dtype = _current_autocast_settings(ms.device.type)
        with torch.autocast(
            device_type=ms.device.type,
            dtype=dtype,
            enabled=enabled,
            cache_enabled=False,
        ):
            return self._forward_ms(ms)

    def _forward_ms(self, ms: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self._backbone_features(self.prepare_inputs(ms))
        logits, embedding = self.decoder(features, output_size=(self.padded_size,) * 2)
        if self.pad_pixels:
            p = self.pad_pixels
            logits = logits[..., p:-p, p:-p]
        if tuple(logits.shape) != (ms.shape[0], 24, 512, 512):
            raise RuntimeError("DINO web logits must have shape Bx24x512x512")
        return {
            "logits": logits,
            "ms_logits": logits,
            "flat_logits": logits,
            "ms_shared": embedding,
            "sar_gate": logits.new_zeros((ms.shape[0], 1, 1, 1)),
            "sar_scale": logits.new_zeros(1),
            "nir_gate": self.nir_gate.reshape(1),
        }

    def predict_proba(self, ms: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        return self.forward(ms, **kwargs)["logits"].softmax(dim=1)


def build_dino_web_model(
    config: Mapping[str, Any], *, load_pretrained: bool = True
) -> DinoWebSegmentation:
    """Build a training expert, or an evaluation shell for strict full EMA load.

    ``load_pretrained=False`` is explicit: no checkpoint files are read and no
    network weight resolution occurs. The caller must strictly load a complete
    trained state dict before inference. A config cannot enable this bypass.
    """
    return DinoWebSegmentation.from_config(config, load_pretrained=load_pretrained)


def dino_web_parameter_groups(
    model: nn.Module,
    config: Mapping[str, Any],
    *,
    staged_parameters: Sequence[nn.Parameter] = (),
) -> list[dict[str, Any]]:
    """Partition trainable tensors once, using actual depth and a head-LR NIR.

    Explicit staged tensors remain optimizer-bound even if already frozen.
    The function never changes requires_grad; the trainer owns the schedule.
    Permanently frozen tensors are excluded and cannot also be staged.
    """
    train = config["train"]
    base_lr = float(train["learning_rate"])
    backbone_lr = base_lr * float(train["backbone_lr_multiplier"])
    layer_decay = float(train["layer_decay"])
    weight_decay = float(train["weight_decay"])
    if (
        not math.isfinite(base_lr)
        or base_lr <= 0
        or not math.isfinite(backbone_lr)
        or backbone_lr <= 0
    ):
        raise ValueError("DINO web learning rates must be positive and finite")
    if not 0 < layer_decay <= 1:
        raise ValueError("DINO web layer_decay must be in (0, 1]")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("DINO web weight_decay must be nonnegative and finite")
    depth = len(getattr(getattr(model, "backbone", None), "blocks", ()))
    if depth not in (12, 24):
        raise ValueError("DINO web layer decay requires a 12- or 24-block backbone")
    named_parameters = list(model.named_parameters())
    staged_ids = {id(parameter) for parameter in staged_parameters}
    model_ids = {id(parameter) for _, parameter in named_parameters}
    if not staged_ids.issubset(model_ids):
        raise ValueError("DINO web staged parameters must belong to this model")
    permanent_prefixes = tuple(str(name) for name in train.get("freeze_modules", ()))
    permanent_ids = {
        id(parameter)
        for name, parameter in named_parameters
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in permanent_prefixes
        )
    }
    if permanent_ids & staged_ids:
        raise ValueError("DINO web permanent freeze and staged parameters overlap")
    eligible_ids = {
        id(parameter)
        for _, parameter in named_parameters
        if (parameter.requires_grad or id(parameter) in staged_ids)
        and id(parameter) not in permanent_ids
    }
    grouped: defaultdict[tuple[bool, int, bool], list[tuple[str, nn.Parameter]]] = (
        defaultdict(list)
    )
    for name, parameter in named_parameters:
        if id(parameter) not in eligible_ids:
            continue
        is_backbone = name.startswith("backbone.") and not name.startswith(
            "backbone.patch_embed.proj.nir_"
        )
        layer = depth + 1
        if is_backbone:
            remainder = name[len("backbone.") :]
            if remainder.startswith("blocks."):
                try:
                    index = int(remainder.split(".")[1])
                except (IndexError, ValueError) as error:
                    raise ValueError(
                        f"malformed DINO web block parameter name: {name}"
                    ) from error
                if not 0 <= index < depth:
                    raise ValueError(
                        f"DINO web block index is outside depth {depth}: {name}"
                    )
                layer = index + 1
            else:
                layer = (
                    depth if remainder.startswith(("norm.", "fc_norm.", "head.")) else 0
                )
        decay = (
            parameter.ndim > 1
            and not name.endswith(".bias")
            and not any(
                marker in name.lower()
                for marker in (
                    "pos_embed",
                    "cls_token",
                    "mask_token",
                    "reg_token",
                    "register_tokens",
                    "storage_tokens",
                )
            )
        )
        grouped[(is_backbone, layer, decay)].append((name, parameter))
    if not grouped:
        raise ValueError("DINO web has no trainable parameters")
    result = []
    observed: set[int] = set()
    for (is_backbone, layer, decay), members in sorted(grouped.items()):
        parameters = [parameter for _, parameter in members]
        ids = {id(parameter) for parameter in parameters}
        if observed & ids:  # pragma: no cover - named_parameters deduplicates
            raise RuntimeError("DINO web optimizer parameter groups overlap")
        observed.update(ids)
        prefix = f"backbone_layer_{layer:02d}" if is_backbone else "new_modules"
        result.append(
            {
                "params": parameters,
                "lr": backbone_lr * layer_decay ** (depth - layer)
                if is_backbone
                else base_lr,
                "weight_decay": weight_decay if decay else 0.0,
                "group_name": f"{prefix}_{'decay' if decay else 'no_decay'}",
                "parameter_names": [name for name, _ in members],
            }
        )
    if observed != eligible_ids:
        raise RuntimeError("DINO web optimizer omitted trainable parameters")
    return result


__all__ = [
    "DINO_WEB_VARIANTS",
    "DINO_WEB_RGB_MEAN",
    "DINO_WEB_RGB_STD",
    "DinoWebVariantSpec",
    "DinoWebSegmentation",
    "GatedRGBNIRPatchProjection",
    "build_dino_web_model",
    "dino_web_parameter_groups",
]
