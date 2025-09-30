from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

@dataclass
class AttentionGateArgs:
    enable_building_ag: bool
    enable_damage_ag: bool

class AttentionGate2d(nn.Module):
    """
    Additive Attention Gate for skip connections (NCHW).
    x: high-res skip  [B, Cx, H, W]
    g: decoder gate   [B, Cg, Hg, Wg]  (upsampled to H x W if needed)
    Returns: x_gated = x * alpha(x,g), alpha ∈ [0,1]
    """
    def __init__(self, in_ch_x: int, in_ch_g: int, inter_ch: int = 64) -> None:
        super().__init__()
        self.theta_x = nn.Conv2d(in_ch_x, inter_ch, 1, bias=False)
        self.phi_g   = nn.Conv2d(in_ch_g, inter_ch, 1, bias=True)
        self.bn_x    = nn.BatchNorm2d(inter_ch)
        self.bn_g    = nn.BatchNorm2d(inter_ch)
        self.psi     = nn.Conv2d(inter_ch, 1, 1, bias=True)
        self.act     = nn.ReLU(inplace=True)
        self.sig     = nn.Sigmoid()

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        B, Cx, H, W = x.shape
        if g.shape[-2:] != (H, W):
            g = F.interpolate(g, size=(H, W), mode="bilinear", align_corners=False)
        qx = self.bn_x(self.theta_x(x))        # [B, inter, H, W]
        kg = self.bn_g(self.phi_g(g))          # [B, inter, H, W]
        a  = self.act(qx + kg)
        alpha = self.sig(self.psi(a))          # [B, 1, H, W]
        return x * alpha
