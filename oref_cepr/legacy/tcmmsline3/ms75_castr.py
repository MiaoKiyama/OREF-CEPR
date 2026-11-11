"""Exact CASTR structure/context/router building blocks for MS75.

This file adapts portions of ``fbponlyms.models.castr`` (Copyright (c)
2026 MiaoKiyama) under the MIT License.  The parameter/module names and
numerical operations are deliberately kept compatible with the archived
``baseline-ms-70.1564`` checkpoint.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


NUM_CLASSES = 24


def _groups(channels: int) -> int:
    for value in (32, 16, 8, 4, 2, 1):
        if channels % value == 0 and channels // value >= 2:
            return value
    return 1


class ConvNormAct(nn.Sequential):
    """The exact convolution/norm/activation primitive used by CASTR."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        dilation: int = 1,
        activation: bool = True,
    ) -> None:
        if padding is None:
            padding = dilation * (kernel_size // 2)
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(_groups(out_channels), out_channels),
        ]
        if activation:
            layers.append(nn.GELU())
        super().__init__(*layers)


@dataclass
class AnchorContextState:
    """Whole-scene evidence cached once for one or more local crops."""

    context_features: tuple[torch.Tensor, ...]
    context_deep: torch.Tensor
    structure_deep: torch.Tensor
    global_logits: torch.Tensor


class SparseStructureBlock(nn.Module):
    """Fixed sparse-grid message passing using multi-distance neighbours."""

    def __init__(
        self,
        channels: int,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                ConvNormAct(
                    channels,
                    channels,
                    kernel_size=3,
                    groups=channels,
                    dilation=dilation,
                )
                for dilation in dilations
            ]
        )
        self.merge = ConvNormAct(
            channels * len(dilations), channels, kernel_size=1, padding=0
        )
        self.refine = ConvNormAct(channels, channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        messages = self.merge(
            torch.cat([branch(x) for branch in self.branches], dim=1)
        )
        return x + self.refine(messages)


class CrossEvidenceFusion(nn.Module):
    """Let local deep tokens query a bounded whole-scene evidence grid."""

    def __init__(
        self,
        local_channels: int,
        evidence_channels: int,
        decoder_channels: int,
        attention_channels: int = 128,
        num_heads: int = 4,
        max_context_grid: int = 16,
    ) -> None:
        super().__init__()
        if attention_channels % num_heads:
            raise ValueError("attention_channels must be divisible by num_heads")
        if attention_channels % 4:
            raise ValueError(
                "attention_channels must be divisible by four for 2D encoding"
            )
        self.local_projection = ConvNormAct(
            local_channels, attention_channels, kernel_size=1, padding=0
        )
        self.evidence_projection = ConvNormAct(
            evidence_channels, attention_channels, kernel_size=1, padding=0
        )
        self.box_embedding = nn.Sequential(
            nn.Linear(4, attention_channels),
            nn.GELU(),
            nn.Linear(attention_channels, attention_channels),
        )
        self.attention = nn.MultiheadAttention(
            attention_channels, num_heads=num_heads, batch_first=True
        )
        frequency_count = attention_channels // 4
        self.register_buffer(
            "position_frequencies",
            torch.arange(1, frequency_count + 1, dtype=torch.float32) * math.pi,
            persistent=False,
        )
        self.position_scale = nn.Parameter(torch.ones(1))
        self.relative_position_scale = nn.Parameter(torch.ones(num_heads))
        self.num_heads = int(num_heads)
        self.norm1 = nn.LayerNorm(attention_channels)
        self.ffn = nn.Sequential(
            nn.Linear(attention_channels, attention_channels * 2),
            nn.GELU(),
            nn.Linear(attention_channels * 2, attention_channels),
        )
        self.norm2 = nn.LayerNorm(attention_channels)
        self.output = ConvNormAct(
            attention_channels, decoder_channels, kernel_size=1, padding=0
        )
        self.max_context_grid = int(max_context_grid)

    @staticmethod
    def _coordinate_grid(
        height: int,
        width: int,
        *,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        y = (
            torch.arange(height, device=device, dtype=torch.float32) + 0.5
        ) / height
        x = (
            torch.arange(width, device=device, dtype=torch.float32) + 0.5
        ) / width
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((grid_y, grid_x), dim=-1).reshape(
            1, height * width, 2
        )
        return coordinates.expand(batch, -1, -1)

    def _position_encoding(
        self, coordinates: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        angles = coordinates.unsqueeze(-1) * self.position_frequencies.view(
            1, 1, 1, -1
        )
        encoded = torch.cat(
            (
                angles[:, :, 0].sin(),
                angles[:, :, 0].cos(),
                angles[:, :, 1].sin(),
                angles[:, :, 1].cos(),
            ),
            dim=-1,
        )
        return encoded.to(dtype=dtype)

    def forward(
        self,
        local_deep: torch.Tensor,
        evidence: torch.Tensor,
        boxes: torch.Tensor,
        output_size: tuple[int, int],
        *,
        evidence_boxes: torch.Tensor | None = None,
        evidence_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        local = self.local_projection(local_deep)
        batch, channels, local_height, local_width = local.shape
        context_height = min(evidence.shape[-2], self.max_context_grid)
        context_width = min(evidence.shape[-1], self.max_context_grid)
        valid_mask: torch.Tensor | None = None
        if evidence_valid_mask is None:
            evidence = F.adaptive_avg_pool2d(
                evidence, (context_height, context_width)
            )
        else:
            if (
                evidence_valid_mask.ndim != 4
                or evidence_valid_mask.shape[:2] != (batch, 1)
            ):
                raise ValueError(
                    "evidence_valid_mask must be shaped [N, 1, H, W] "
                    "for the local batch"
                )
            if evidence_valid_mask.shape[-2:] != evidence.shape[-2:]:
                raise ValueError(
                    "evidence_valid_mask spatial size must match evidence"
                )
            mask_value = evidence_valid_mask.to(dtype=evidence.dtype)
            if evidence.shape[-2:] == (context_height, context_width):
                evidence = evidence * mask_value
                valid_mask = evidence_valid_mask.bool()
            else:
                pooled_mask = F.adaptive_avg_pool2d(
                    mask_value, (context_height, context_width)
                )
                evidence = F.adaptive_avg_pool2d(
                    evidence * mask_value, (context_height, context_width)
                ) / pooled_mask.clamp_min(1e-6)
                valid_mask = pooled_mask > 0
        evidence = self.evidence_projection(evidence)
        if valid_mask is not None:
            evidence = evidence * valid_mask.to(dtype=evidence.dtype)

        queries = local.flatten(2).transpose(1, 2)
        local_coordinates = self._coordinate_grid(
            local_height, local_width, batch=batch, device=local.device
        )
        box_start = boxes[:, :2].float().unsqueeze(1)
        box_extent = (boxes[:, 2:] - boxes[:, :2]).float().unsqueeze(1)
        query_coordinates = box_start + local_coordinates * box_extent
        base_key_coordinates = self._coordinate_grid(
            context_height, context_width, batch=batch, device=local.device
        )
        if evidence_boxes is None:
            key_coordinates = base_key_coordinates
        else:
            if evidence_boxes.shape != boxes.shape:
                raise ValueError(
                    "evidence_boxes must have the same [N, 4] shape as local boxes"
                )
            evidence_start = evidence_boxes[:, :2].float().unsqueeze(1)
            evidence_extent = (
                evidence_boxes[:, 2:] - evidence_boxes[:, :2]
            ).float().unsqueeze(1)
            key_coordinates = (
                evidence_start + base_key_coordinates * evidence_extent
            )
        position_scale = self.position_scale.to(dtype=queries.dtype)
        queries = (
            queries
            + self.box_embedding(boxes).unsqueeze(1)
            + position_scale
            * self._position_encoding(query_coordinates, queries.dtype)
        )
        values = evidence.flatten(2).transpose(1, 2)
        keys = values + position_scale * self._position_encoding(
            key_coordinates, values.dtype
        )

        squared_distance = (
            (query_coordinates.unsqueeze(2) - key_coordinates.unsqueeze(1))
            .square()
            .sum(dim=-1)
        )
        relative_scale = F.softplus(self.relative_position_scale).view(
            1, self.num_heads, 1, 1
        )
        attention_bias = -(relative_scale * squared_distance.unsqueeze(1))
        attention_bias = attention_bias.reshape(
            batch * self.num_heads, queries.shape[1], keys.shape[1]
        ).to(dtype=queries.dtype)
        if valid_mask is not None:
            valid_keys = valid_mask.flatten(2).squeeze(1)
            invalid_keys = (~valid_keys)[:, None, None, :].expand(
                -1, self.num_heads, queries.shape[1], -1
            )
            attention_bias = attention_bias.view(
                batch, self.num_heads, queries.shape[1], keys.shape[1]
            )
            attention_bias = attention_bias.masked_fill(
                invalid_keys, torch.finfo(attention_bias.dtype).min
            )
            attention_bias = attention_bias.reshape(
                batch * self.num_heads, queries.shape[1], keys.shape[1]
            )
        attended, _ = self.attention(
            queries, keys, values, attn_mask=attention_bias, need_weights=False
        )
        tokens = self.norm1(queries + attended)
        tokens = self.norm2(tokens + self.ffn(tokens))
        feature = tokens.transpose(1, 2).reshape(
            batch, channels, local_height, local_width
        )
        feature = self.output(feature)
        return F.interpolate(
            feature, output_size, mode="bilinear", align_corners=False
        )


class EvidenceRouter(nn.Module):
    """The archived CASTR learned three-expert router."""

    def __init__(
        self,
        channels: int,
        *,
        signal_mode: str = "local_entropy",
        detach_signals: bool = False,
    ) -> None:
        super().__init__()
        if signal_mode not in {"local_entropy", "expert_uncertainty"}:
            raise ValueError(f"Unknown router signal mode: {signal_mode}")
        self.signal_mode = signal_mode
        self.detach_signals = bool(detach_signals)
        signal_channels = 1 if signal_mode == "local_entropy" else 5
        self.network = nn.Sequential(
            ConvNormAct(channels * 3 + signal_channels, channels, kernel_size=3),
            nn.Conv2d(channels, 3, kernel_size=1),
        )

    @staticmethod
    def _normalized_entropy(logits: torch.Tensor) -> torch.Tensor:
        working_logits = (
            logits.float()
            if logits.dtype in {torch.float16, torch.bfloat16}
            else logits
        )
        probabilities = working_logits.softmax(dim=1)
        entropy = -(
            probabilities * probabilities.clamp_min(1e-8).log()
        ).sum(dim=1, keepdim=True)
        return entropy / torch.log(
            torch.tensor(
                float(NUM_CLASSES), device=entropy.device, dtype=entropy.dtype
            )
        )

    @staticmethod
    def _normalized_js(
        first_logits: torch.Tensor, second_logits: torch.Tensor
    ) -> torch.Tensor:
        first = first_logits.float().softmax(dim=1)
        second = second_logits.float().softmax(dim=1)
        mixture = (first + second) * 0.5
        first_kl = (
            first
            * (
                first.clamp_min(1e-8).log()
                - mixture.clamp_min(1e-8).log()
            )
        ).sum(dim=1, keepdim=True)
        second_kl = (
            second
            * (
                second.clamp_min(1e-8).log()
                - mixture.clamp_min(1e-8).log()
            )
        ).sum(dim=1, keepdim=True)
        return (first_kl + second_kl) * (0.5 / math.log(2.0))

    def _uncertainty_signals(
        self,
        local_logits: torch.Tensor,
        context_logits: torch.Tensor | None,
        third_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.signal_mode == "local_entropy":
            return self._normalized_entropy(local_logits)
        if context_logits is None or third_logits is None:
            raise ValueError(
                "expert_uncertainty routing requires logits from all three experts"
            )
        return torch.cat(
            (
                self._normalized_entropy(local_logits),
                self._normalized_entropy(context_logits),
                self._normalized_entropy(third_logits),
                self._normalized_js(local_logits, context_logits),
                self._normalized_js(local_logits, third_logits),
            ),
            dim=1,
        )

    def forward(
        self,
        local_feature: torch.Tensor,
        context_feature: torch.Tensor,
        third_feature: torch.Tensor,
        local_logits: torch.Tensor,
        context_logits: torch.Tensor | None = None,
        third_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.detach_signals:
            with torch.no_grad():
                signals = self._uncertainty_signals(
                    local_logits, context_logits, third_logits
                )
        else:
            signals = self._uncertainty_signals(
                local_logits, context_logits, third_logits
            )
        signals = F.interpolate(
            signals,
            local_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        signals = signals.to(dtype=local_feature.dtype)
        values = torch.cat(
            [local_feature, context_feature, third_feature, signals], dim=1
        )
        return self.network(values)


__all__ = [
    "AnchorContextState",
    "CrossEvidenceFusion",
    "EvidenceRouter",
    "SparseStructureBlock",
]
