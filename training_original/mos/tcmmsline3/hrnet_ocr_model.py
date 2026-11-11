"""Four-band HRNet-W32 + OCR local geometry expert.

The backbone path intentionally consumes the native four parallel branches of
HRNet stage 4.  It does *not* use timm's ``features_only`` classification
pyramid, whose channels have already passed through the classification head.

ImageNet HRNet checkpoints are RGB.  The first convolution is therefore
adapted explicitly for the project's B/G/R/NIR input contract: B, G and R get
the corresponding pretrained RGB filters and NIR starts from their mean.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


HRNET_W32_BRANCH_CHANNELS = (32, 64, 128, 256)
HRNET_W32_MODEL_NAME = "hrnet_w32.ms_in1k"


def _groups(channels: int, requested: int) -> int:
    """Return the largest requested-or-smaller divisor for GroupNorm."""

    upper = min(int(channels), int(requested))
    for value in range(upper, 0, -1):
        if channels % value == 0:
            return value
    return 1  # pragma: no cover - every positive integer is divisible by one


class _ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        groups: int = 32,
        activation: bool = True,
    ) -> None:
        if int(in_channels) <= 0 or int(out_channels) <= 0:
            raise ValueError("convolution channels must be positive")
        if int(groups) <= 0:
            raise ValueError("group_norm_groups must be positive")
        padding = int(kernel_size) // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                int(in_channels),
                int(out_channels),
                int(kernel_size),
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_groups(int(out_channels), int(groups)), int(out_channels)),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class _DepthwiseSeparableNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        groups: int,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                int(in_channels),
                int(in_channels),
                3,
                padding=1,
                groups=int(in_channels),
                bias=False,
            ),
            nn.GroupNorm(
                _groups(int(in_channels), int(groups)), int(in_channels)
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels), int(out_channels), 1, bias=False),
            nn.GroupNorm(
                _groups(int(out_channels), int(groups)), int(out_channels)
            ),
            nn.ReLU(inplace=True),
        )


def adapt_rgb_stem_weight_to_bgr_nir(
    rgb_weight: torch.Tensor,
    *,
    nir_init: str = "rgb_mean",
) -> torch.Tensor:
    """Map an RGB stem tensor to the explicit B/G/R/NIR input order.

    Args:
        rgb_weight: Convolution weight with shape ``(out, 3, kh, kw)``.
        nir_init: ``"rgb_mean"`` (the production protocol) or ``"zeros"``.

    The returned tensor is newly allocated; the input is never mutated.
    """

    if rgb_weight.ndim != 4 or rgb_weight.shape[1] != 3:
        raise ValueError("RGB stem weight must have shape (out, 3, kh, kw)")
    strategy = str(nir_init).lower()
    if strategy not in {"rgb_mean", "zeros"}:
        raise ValueError("nir_init must be rgb_mean or zeros")

    adapted = rgb_weight.new_empty(
        (rgb_weight.shape[0], 4, rgb_weight.shape[2], rgb_weight.shape[3])
    )
    # ImageNet tensor order is R/G/B; the project input order is B/G/R/NIR.
    adapted[:, 0].copy_(rgb_weight[:, 2])
    adapted[:, 1].copy_(rgb_weight[:, 1])
    adapted[:, 2].copy_(rgb_weight[:, 0])
    if strategy == "rgb_mean":
        adapted[:, 3].copy_(rgb_weight.mean(dim=1))
    else:
        adapted[:, 3].zero_()
    return adapted


def _replace_rgb_stem(
    network: nn.Module,
    *,
    nir_init: str,
) -> None:
    old = getattr(network, "conv1", None)
    if not isinstance(old, nn.Conv2d) or old.in_channels != 3:
        raise RuntimeError("timm HRNet must expose a three-channel conv1 stem")
    if old.groups != 1:
        raise RuntimeError("HRNet conv1 must not be grouped")

    replacement = nn.Conv2d(
        4,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    with torch.no_grad():
        replacement.weight.copy_(
            adapt_rgb_stem_weight_to_bgr_nir(
                old.weight.detach(), nir_init=nir_init
            )
        )
        if old.bias is not None:
            assert replacement.bias is not None
            replacement.bias.copy_(old.bias)
    network.conv1 = replacement


def _validate_sha256(value: str | None, *, pretrained: bool) -> str | None:
    if value is None:
        return None
    digest = str(value).lower()
    if (
        not pretrained
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(
            "pretrained_file_sha256 requires pretrained=true and 64 hexadecimal "
            "characters"
        )
    return digest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_pretrained_file(value: str | os.PathLike[str]) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(os.fspath(value)))
    path = Path(expanded).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"HRNet pretrained_file does not exist: {path}")
    if path.suffix.lower() != ".safetensors":
        raise ValueError("HRNet pretrained_file must be a .safetensors artifact")
    return path


def _load_explicit_safetensors(
    network: nn.Module,
    pretrained_file: str | os.PathLike[str],
    *,
    expected_sha256: str,
) -> Path:
    path = _resolve_pretrained_file(pretrained_file)
    observed_sha256 = _sha256(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "HRNet pretrained checkpoint SHA256 differs from config: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )
    try:
        from safetensors.torch import load_file
    except ImportError as error:  # pragma: no cover - environment contract
        raise RuntimeError(
            "an explicit HRNet .safetensors file requires safetensors"
        ) from error

    source_state = load_file(str(path), device="cpu")
    target_state = network.state_dict()
    missing = sorted(set(target_state).difference(source_state))
    if missing:
        raise RuntimeError(
            "explicit HRNet checkpoint is missing backbone tensors: "
            + ", ".join(missing[:8])
        )
    mismatched = sorted(
        key
        for key, target in target_state.items()
        if source_state[key].shape != target.shape
    )
    if mismatched:
        raise RuntimeError(
            "explicit HRNet checkpoint tensor shapes differ: "
            + ", ".join(mismatched[:8])
        )
    # The official classification artifact also contains incre/head tensors.
    # The segmentation backbone deliberately loads only its exact key set.
    network.load_state_dict(
        {key: source_state[key] for key in target_state}, strict=True
    )
    return path


def _create_timm_hrnet(
    model_name: str,
    *,
    pretrained: bool,
    pretrained_file: str | os.PathLike[str] | None = None,
    pretrained_file_sha256: str | None = None,
) -> tuple[nn.Module, Path | None]:
    try:
        import timm
    except ImportError as error:  # pragma: no cover - environment contract
        raise RuntimeError("hrnet_w32_ocr requires timm") from error
    explicit = pretrained_file is not None
    if explicit and not pretrained:
        raise ValueError("pretrained_file requires pretrained=true")
    if explicit and pretrained_file_sha256 is None:
        raise ValueError("pretrained_file requires pretrained_file_sha256")
    if pretrained_file_sha256 is not None and not explicit:
        raise ValueError(
            "pretrained_file_sha256 requires an explicit pretrained_file"
        )
    network = timm.create_model(
        model_name,
        # With an explicit artifact we load and audit it below; letting timm
        # fetch by model name here could silently select a moving HF revision.
        pretrained=bool(pretrained) and not explicit,
        in_chans=3,
        head="",
    )
    resolved: Path | None = None
    if explicit:
        assert pretrained_file is not None
        assert pretrained_file_sha256 is not None
        resolved = _load_explicit_safetensors(
            network,
            pretrained_file,
            expected_sha256=pretrained_file_sha256,
        )
    return network, resolved


class HRNetW32NativeBackbone(nn.Module):
    """Expose HRNet-W32's native stage-4 parallel representations."""

    channels = HRNET_W32_BRANCH_CHANNELS

    def __init__(
        self,
        *,
        pretrained: bool = True,
        pretrained_model: str = HRNET_W32_MODEL_NAME,
        pretrained_file: str | os.PathLike[str] | None = None,
        pretrained_file_sha256: str | None = None,
        nir_init: str = "rgb_mean",
        freeze_batch_norm_stats: bool = True,
    ) -> None:
        super().__init__()
        model_name = str(pretrained_model)
        if model_name != HRNET_W32_MODEL_NAME:
            raise ValueError(
                f"pretrained_model must be exactly {HRNET_W32_MODEL_NAME}"
            )
        self.pretrained_file_sha256 = _validate_sha256(
            pretrained_file_sha256, pretrained=bool(pretrained)
        )
        self.pretrained_model = model_name
        self.nir_init = str(nir_init).lower()
        self.freeze_batch_norm_stats = bool(freeze_batch_norm_stats)

        network, resolved_pretrained_file = _create_timm_hrnet(
            model_name,
            pretrained=bool(pretrained),
            pretrained_file=pretrained_file,
            pretrained_file_sha256=self.pretrained_file_sha256,
        )
        self.pretrained_file = (
            str(resolved_pretrained_file)
            if resolved_pretrained_file is not None
            else None
        )
        stage4 = getattr(network, "stage4_cfg", None)
        stage4_channels = (
            tuple(int(value) for value in stage4.get("num_channels", ()))
            if isinstance(stage4, Mapping)
            else ()
        )
        if stage4_channels != self.channels:
            raise RuntimeError(
                "timm HRNet-W32 native stage4 channel contract changed: "
                f"expected {self.channels}, observed {stage4_channels}"
            )
        if not callable(getattr(network, "stages", None)):
            raise RuntimeError("timm HRNet no longer exposes its native stages method")
        _replace_rgb_stem(network, nir_init=self.nir_init)
        self.network = network
        if self.freeze_batch_norm_stats:
            self._freeze_batch_norm_stats()

    def _freeze_batch_norm_stats(self) -> None:
        for module in self.network.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def train(self, mode: bool = True) -> "HRNetW32NativeBackbone":
        super().train(mode)
        if mode and self.freeze_batch_norm_stats:
            self._freeze_batch_norm_stats()
        return self

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if inputs.ndim != 4 or inputs.shape[1] != 4:
            raise ValueError("HRNet-W32 MS input must have shape Bx4xHxW")
        network = self.network
        stem_stride2 = network.act1(network.bn1(network.conv1(inputs)))
        value = network.act2(network.bn2(network.conv2(stem_stride2)))
        branches = tuple(network.stages(value))
        if len(branches) != 4:
            raise RuntimeError(
                "HRNet-W32 stage4 must return exactly four parallel branches"
            )
        expected_sizes = tuple(
            (
                max(inputs.shape[-2] // (4 * (2**index)), 1),
                max(inputs.shape[-1] // (4 * (2**index)), 1),
            )
            for index in range(4)
        )
        for index, (branch, channels, spatial) in enumerate(
            zip(branches, self.channels, expected_sizes, strict=True)
        ):
            if branch.ndim != 4 or branch.shape[1] != channels:
                raise RuntimeError(
                    f"HRNet-W32 branch {index} must have {channels} channels"
                )
            if tuple(branch.shape[-2:]) != spatial:
                raise RuntimeError(
                    f"HRNet-W32 branch {index} has unexpected spatial shape "
                    f"{tuple(branch.shape[-2:])}; expected {spatial}"
                )
        return stem_stride2, branches


class SpatialGatherModule(nn.Module):
    """Aggregate pixel features into one soft object-region proxy per class."""

    def __init__(self, *, scale: float = 1.0) -> None:
        super().__init__()
        if not math.isfinite(float(scale)) or float(scale) <= 0:
            raise ValueError("OCR gather scale must be positive and finite")
        self.scale = float(scale)

    def forward(
        self, features: torch.Tensor, class_logits: torch.Tensor
    ) -> torch.Tensor:
        if features.ndim != 4 or class_logits.ndim != 4:
            raise ValueError("OCR gather inputs must be four-dimensional")
        if features.shape[0] != class_logits.shape[0] or (
            features.shape[-2:] != class_logits.shape[-2:]
        ):
            raise ValueError("OCR gather feature/logit shapes are incompatible")
        # With a 384 crop this softmax spans 9,216 locations.  Force the
        # reduction and accumulation to fp32 under AMP to avoid fp16
        # underflow erasing low-probability region evidence.
        with torch.autocast(device_type=features.device.type, enabled=False):
            probabilities = F.softmax(
                class_logits.float().flatten(2) * self.scale,
                dim=2,
            )
            pixels = features.float().flatten(2).transpose(1, 2)
            context = torch.bmm(probabilities, pixels)
        return context.transpose(1, 2).unsqueeze(-1).to(features.dtype)


class ObjectAttentionBlock(nn.Module):
    """Distribute the gathered class-region context back to every pixel."""

    def __init__(
        self,
        in_channels: int,
        key_channels: int,
        *,
        group_norm_groups: int,
    ) -> None:
        super().__init__()
        if int(key_channels) <= 0:
            raise ValueError("ocr_key_channels must be positive")
        self.key_channels = int(key_channels)
        common = {
            "kernel_size": 1,
            "groups": int(group_norm_groups),
        }
        self.query = nn.Sequential(
            _ConvNormAct(in_channels, key_channels, **common),
            _ConvNormAct(key_channels, key_channels, **common),
        )
        self.key = nn.Sequential(
            _ConvNormAct(in_channels, key_channels, **common),
            _ConvNormAct(key_channels, key_channels, **common),
        )
        self.value = _ConvNormAct(in_channels, key_channels, **common)
        self.project = _ConvNormAct(key_channels, in_channels, **common)

    def forward(
        self, features: torch.Tensor, object_proxy: torch.Tensor
    ) -> torch.Tensor:
        batch, _, height, width = features.shape
        query = self.query(features).flatten(2).transpose(1, 2)
        key = self.key(object_proxy).flatten(2)
        value = self.value(object_proxy).flatten(2).transpose(1, 2)
        with torch.autocast(device_type=features.device.type, enabled=False):
            similarity = torch.bmm(query.float(), key.float()) * (
                self.key_channels**-0.5
            )
            similarity = F.softmax(similarity, dim=-1)
            context = torch.bmm(similarity, value.float()).to(features.dtype)
        context = context.transpose(1, 2).reshape(
            batch, self.key_channels, height, width
        )
        return self.project(context)


class OCRHead(nn.Module):
    """Official-style gather/distribute OCR head over fused HRNet branches."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        ocr_channels: int,
        key_channels: int,
        group_norm_groups: int,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if int(ocr_channels) <= 0:
            raise ValueError("ocr_channels must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("ocr_dropout must be in [0, 1)")
        self.feature_projection = _ConvNormAct(
            int(in_channels),
            int(ocr_channels),
            groups=int(group_norm_groups),
        )
        self.aux_classifier = nn.Conv2d(int(ocr_channels), int(num_classes), 1)
        self.gather = SpatialGatherModule()
        self.distribute = ObjectAttentionBlock(
            int(ocr_channels),
            int(key_channels),
            group_norm_groups=int(group_norm_groups),
        )
        self.fuse = nn.Sequential(
            _ConvNormAct(
                2 * int(ocr_channels),
                int(ocr_channels),
                kernel_size=1,
                groups=int(group_norm_groups),
            ),
            nn.Dropout2d(float(dropout)),
        )
        self.classifier = nn.Conv2d(int(ocr_channels), int(num_classes), 1)

    def forward(
        self, fused_hrnet: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.feature_projection(fused_hrnet)
        auxiliary_logits = self.aux_classifier(features)
        object_proxy = self.gather(features, auxiliary_logits)
        context = self.distribute(features, object_proxy)
        embedding = self.fuse(torch.cat([features, context], dim=1))
        logits = self.classifier(embedding)
        return logits, auxiliary_logits, embedding, object_proxy


class HRNetW32OCRSegmentation(nn.Module):
    """Pure-MS high-resolution geometry expert for the 24-class protocol."""

    def __init__(
        self,
        *,
        num_classes: int = 24,
        pretrained: bool = True,
        pretrained_model: str = HRNET_W32_MODEL_NAME,
        pretrained_file: str | os.PathLike[str] | None = None,
        pretrained_file_sha256: str | None = None,
        stem_in_channels: int = 4,
        nir_init: str = "rgb_mean",
        ocr_channels: int = 256,
        ocr_key_channels: int = 128,
        boundary_channels: int = 64,
        norm_layer: str = "group_norm",
        group_norm_groups: int = 32,
        freeze_backbone_batch_norm_stats: bool = True,
        ocr_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if int(num_classes) != 24:
            raise ValueError("HRNet-W32+OCR is defined for exactly 24 classes")
        if int(stem_in_channels) != 4:
            raise ValueError("HRNet-W32+OCR requires stem_in_channels=4")
        if str(norm_layer).lower() != "group_norm":
            raise ValueError("HRNet-W32+OCR heads require norm_layer=group_norm")
        if int(boundary_channels) <= 0:
            raise ValueError("boundary_channels must be positive")
        if int(group_norm_groups) <= 0:
            raise ValueError("group_norm_groups must be positive")

        self.num_classes = int(num_classes)
        self.backbone = HRNetW32NativeBackbone(
            pretrained=bool(pretrained),
            pretrained_model=str(pretrained_model),
            pretrained_file=pretrained_file,
            pretrained_file_sha256=pretrained_file_sha256,
            nir_init=str(nir_init),
            freeze_batch_norm_stats=bool(freeze_backbone_batch_norm_stats),
        )
        fused_channels = sum(self.backbone.channels)
        self.ocr_head = OCRHead(
            fused_channels,
            self.num_classes,
            ocr_channels=int(ocr_channels),
            key_channels=int(ocr_key_channels),
            group_norm_groups=int(group_norm_groups),
            dropout=float(ocr_dropout),
        )

        # The first stem activation is stride 2.  Fusing it with OCR semantics
        # keeps this auxiliary head geometry-sensitive without making it a
        # second heavy decoder.
        self.boundary_stem_projection = _ConvNormAct(
            64,
            int(boundary_channels),
            kernel_size=1,
            groups=int(group_norm_groups),
        )
        self.boundary_semantic_projection = _ConvNormAct(
            int(ocr_channels),
            int(boundary_channels),
            kernel_size=1,
            groups=int(group_norm_groups),
        )
        self.boundary_fusion = _DepthwiseSeparableNormAct(
            2 * int(boundary_channels),
            int(boundary_channels),
            groups=int(group_norm_groups),
        )
        self.boundary_classifier = nn.Conv2d(int(boundary_channels), 1, 1)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "HRNetW32OCRSegmentation":
        model = config["model"]
        return cls(
            num_classes=int(model["num_classes"]),
            pretrained=bool(model.get("pretrained", True)),
            pretrained_model=str(
                model.get("pretrained_model", HRNET_W32_MODEL_NAME)
            ),
            pretrained_file=model.get("pretrained_file"),
            pretrained_file_sha256=model.get("pretrained_file_sha256"),
            stem_in_channels=int(model.get("stem_in_channels", 4)),
            nir_init=str(model.get("nir_init", "rgb_mean")),
            ocr_channels=int(model.get("ocr_channels", 256)),
            ocr_key_channels=int(model.get("ocr_key_channels", 128)),
            boundary_channels=int(model.get("boundary_channels", 64)),
            norm_layer=str(model.get("norm_layer", "group_norm")),
            group_norm_groups=int(model.get("group_norm_groups", 32)),
            freeze_backbone_batch_norm_stats=bool(
                model.get("freeze_backbone_batch_norm_stats", True)
            ),
            ocr_dropout=float(model.get("ocr_dropout", 0.05)),
        )

    @staticmethod
    def _validate_ms(ms: torch.Tensor) -> None:
        if ms.ndim != 4 or ms.shape[1] != 4:
            raise ValueError("HRNet-W32+OCR MS must have shape Bx4xHxW")
        if not ms.is_floating_point():
            raise TypeError("HRNet-W32+OCR MS must be floating point")
        if ms.shape[-2] % 32 or ms.shape[-1] % 32:
            raise ValueError("MS height and width must be divisible by 32")

    @staticmethod
    def _fuse_native_branches(
        branches: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if len(branches) != 4:
            raise ValueError("exactly four native HRNet branches are required")
        high_resolution_size = branches[0].shape[-2:]
        aligned = [branches[0]]
        aligned.extend(
            F.interpolate(
                branch,
                size=high_resolution_size,
                mode="bilinear",
                align_corners=False,
            )
            for branch in branches[1:]
        )
        return torch.cat(aligned, dim=1)

    def forward(
        self,
        ms: torch.Tensor,
        sar: torch.Tensor,
        sar_valid: torch.Tensor,
        time_gap_days: torch.Tensor | None = None,
        ms_context: torch.Tensor | None = None,
        ms_context_valid: torch.Tensor | None = None,
        **unused: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Shared callers pass SAR/context tensors to every architecture.  This
        # pure four-band local expert intentionally cannot consume them.
        del sar, sar_valid, time_gap_days, ms_context, ms_context_valid, unused
        self._validate_ms(ms)
        output_size = tuple(int(value) for value in ms.shape[-2:])

        stem_stride2, branches = self.backbone(ms)
        fused_hrnet = self._fuse_native_branches(branches)
        (
            logits_stride4,
            coarse_logits_stride4,
            ocr_embedding,
            object_proxy,
        ) = self.ocr_head(fused_hrnet)

        logits = F.interpolate(
            logits_stride4,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        coarse_logits = F.interpolate(
            coarse_logits_stride4,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        semantic_stride2 = F.interpolate(
            self.boundary_semantic_projection(ocr_embedding),
            size=stem_stride2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        boundary_embedding = self.boundary_fusion(
            torch.cat(
                [
                    self.boundary_stem_projection(stem_stride2),
                    semantic_stride2,
                ],
                dim=1,
            )
        )
        boundary_logits_stride2 = self.boundary_classifier(boundary_embedding)
        boundary_logits = F.interpolate(
            boundary_logits_stride2,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        zero_logits = torch.zeros_like(logits)
        zero_shared = torch.zeros_like(ocr_embedding)
        return {
            "logits": logits,
            "ms_logits": logits,
            "flat_logits": logits,
            "logits_stride4": logits_stride4,
            "coarse_logits": coarse_logits,
            "coarse_logits_stride4": coarse_logits_stride4,
            # Generic OCR auxiliary contract.  context_aux_logits is retained
            # as a compatibility alias for older training entry points.
            "aux_logits": coarse_logits,
            "context_aux_logits": coarse_logits,
            "boundary_logits": boundary_logits,
            "boundary_logits_stride2": boundary_logits_stride2,
            "ocr_context": object_proxy,
            "ms_shared": ocr_embedding,
            "sar_logits": zero_logits,
            "sar_native_logits": ms.new_zeros(
                (
                    ms.shape[0],
                    self.num_classes,
                    logits_stride4.shape[-2],
                    logits_stride4.shape[-1],
                )
            ),
            "sar_shared": zero_shared,
            "sar_private": zero_shared,
            "sar_gate": zero_shared[:, :1],
            "sar_scale": logits.new_zeros(1),
            "sar_present": torch.zeros(
                ms.shape[0], dtype=torch.bool, device=ms.device
            ),
        }


__all__ = [
    "HRNET_W32_BRANCH_CHANNELS",
    "HRNET_W32_MODEL_NAME",
    "HRNetW32NativeBackbone",
    "HRNetW32OCRSegmentation",
    "ObjectAttentionBlock",
    "OCRHead",
    "SpatialGatherModule",
    "adapt_rgb_stem_weight_to_bgr_nir",
]
