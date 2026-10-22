"""ACDS-DETR: Spatially Adaptive Context Aggregation (SACA) and
Cross-Level Guided Shallow Fusion (CGSF).

This module depends only on PyTorch and the shared Conv/RepC3 primitives.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.conv import Conv
from ..modules.block import RepC3

__all__ = ('SACAStage', 'CGSF', 'ACDSSGEEncoder', 'ACDSCueRouter')


class LocationWiseReceptiveFieldWeighting(nn.Module):
    """Aggregate dilation {3, 5, 7} responses with channel-shared spatial weights."""

    def __init__(self, channels):
        super().__init__()
        self.branches = nn.ModuleList(Conv(channels, channels, 3, g=channels, d=d) for d in (3, 5, 7))
        self.score = nn.ModuleList(nn.Conv2d(channels, 1, 1) for _ in range(3))
        self.last_scale_weights = None

    def forward(self, x):
        features = [branch(x) for branch in self.branches]
        logits = [score(feature) for score, feature in zip(self.score, features)]
        weights = torch.softmax(torch.stack(logits, dim=1), dim=1)
        self.last_scale_weights = weights.detach()
        return (torch.stack(features, dim=1) * weights).sum(dim=1)


class SpatiallyGatedResidualRefinement(nn.Module):
    """Spatially gate and project a residual feature update, shared across channels."""

    def __init__(self, channels):
        super().__init__()
        self.expand = Conv(channels, channels * 2, 1)
        self.local = Conv(channels, channels, 3, g=channels)
        self.gate = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())
        self.project = Conv(channels, channels, 1)
        self.last_spatial_gate = None

    def forward(self, x):
        gate_features, residual = self.expand(x).chunk(2, dim=1)
        gate = self.gate(self.local(gate_features))
        self.last_spatial_gate = gate.detach()
        return self.project(residual * gate)


class SACABlock(nn.Module):
    """Location-wise context aggregation followed by spatially gated refinement."""

    def __init__(self, channels):
        super().__init__()
        self.context = LocationWiseReceptiveFieldWeighting(channels)
        self.spatial = SpatiallyGatedResidualRefinement(channels)

    def forward(self, x):
        selected = x + self.context(x)
        return selected + self.spatial(selected)


class SACAStage(nn.Module):
    """Spatially Adaptive Context Aggregation stage with refinement and bypass paths."""

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        hidden = max(1, int(c2 * e))
        self.cv1 = Conv(c1, hidden, 1)
        self.cv2 = Conv(c1, hidden, 1)
        self.m = nn.Sequential(*(SACABlock(hidden) for _ in range(n)))
        self.cv3 = Conv(hidden * 2, c2, 1)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SpaceToDepthAlignment(nn.Module):
    """S2D rearrangement followed by a 3x3 Conv-BN-SiLU projection."""

    def __init__(self, c1, c2):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, 3)

    def forward(self, x):
        x = torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2],
                       x[..., ::2, 1::2], x[..., 1::2, 1::2]), dim=1)
        return self.conv(x)


class CrossLevelSpatialGate(nn.Module):
    """Cross-level-conditioned spatial gating from shallow, lateral and top-down features."""

    def __init__(self, c_p2, c_p3, c_td, c_gate):
        super().__init__()
        self.p2_proj = Conv(c_p2, c_gate, 1)
        self.p3_proj = Conv(c_p3, c_gate, 1)
        self.td_proj = Conv(c_td, c_gate, 1)
        self.gate = nn.Sequential(Conv(c_gate * 3, c_gate, 3), nn.Conv2d(c_gate, 1, 1), nn.Sigmoid())
        self.last_gate = None

    def forward(self, shallow, lateral, top_down):
        context = torch.cat((self.p2_proj(shallow), self.p3_proj(lateral), self.td_proj(top_down)), dim=1)
        gate = self.gate(context)
        self.last_gate = gate.detach()
        return gate


class CGSF(nn.Module):
    """Cross-Level Guided Shallow Fusion: S2D -> shallow-only gate -> concat -> RepC3.

    Lateral and top-down features condition the gate but are not gated.
    The fused feature enters the three-level detection hierarchy.
    """

    def __init__(self, c_p2, c_p3, c_td, c_out=256, c_align=None, n=3, e=0.5):
        super().__init__()
        c_align = c_td if c_align is None else c_align
        self.align = SpaceToDepthAlignment(c_p2, c_align)
        self.reliability_gate = CrossLevelSpatialGate(c_align, c_p3, c_td, c_align)
        self.rep = RepC3(c_td + c_p3 + c_align, c_out, n=n, e=e)
        self.last_gate = None

    def forward(self, inputs):
        shallow, lateral, top_down = inputs
        shallow = self.align(shallow)
        if lateral.shape[-2:] != shallow.shape[-2:]:
            lateral = F.interpolate(lateral, size=shallow.shape[-2:], mode='nearest')
        if top_down.shape[-2:] != shallow.shape[-2:]:
            top_down = F.interpolate(top_down, size=shallow.shape[-2:], mode='nearest')
        gate = self.reliability_gate(shallow, lateral, top_down)
        self.last_gate = gate.detach()
        return self.rep(torch.cat((top_down, lateral, shallow * gate), dim=1))


# Import aliases preserve checkpoint and configuration compatibility.
# Registered module attribute names are unchanged to preserve state_dict keys.
SpatialScaleSelection = LocationWiseReceptiveFieldWeighting
SpatialEvidenceRefiner = SpatiallyGatedResidualRefinement
SpatialSGEBlock = SACABlock
ACDSSGEEncoder = SACAStage
ContextConditionedGate = CrossLevelSpatialGate
ACDSCueRouter = CGSF
