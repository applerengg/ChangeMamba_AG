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
    def __init__(self, in_ch_x: int, in_ch_g: int, inter_ch: int = 64, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.theta_x = nn.Conv2d(in_ch_x, inter_ch, 1, bias=False)
        self.phi_g   = nn.Conv2d(in_ch_g, inter_ch, 1, bias=True)
        self.bn_x    = nn.GroupNorm(8, inter_ch)
        self.bn_g    = nn.GroupNorm(8, inter_ch)

        # Add dropout for regularization
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.psi     = nn.Conv2d(inter_ch, 1, 1, bias=True)
        self.act     = nn.ReLU(inplace=True)
        self.sig     = nn.Sigmoid()

        # Initialize ψ bias negative: start closed → forces learning meaningful openings
        with torch.no_grad():
            self.psi.bias.fill_(-2.0)

        self.last_alpha: torch.Tensor | None = None  # for visualization
        self.register_buffer('alpha_mean', torch.tensor(0.0)) # running mean of alpha for monitoring
        self.register_buffer('alpha_std', torch.tensor(1.0)) # running mean of alpha for monitoring

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        B, Cx, H, W = x.shape
        if g.shape[-2:] != (H, W):
            g = F.interpolate(g, size=(H, W), mode="bilinear", align_corners=False)
        qx = self.bn_x(self.theta_x(x))        # [B, inter, H, W]
        kg = self.bn_g(self.phi_g(g))          # [B, inter, H, W]
        a  = self.act(qx + kg)
        a  = self.dropout(a)  # Regularize attention
        alpha = self.sig(self.psi(a))          # [B, 1, H, W]
        # Track statistics (no grad)
        with torch.no_grad():
            self.last_alpha: torch.Tensor = alpha
            self.alpha_mean: torch.Tensor = alpha.mean()
            self.alpha_std: torch.Tensor = alpha.std()
        return x * (0.5 + 0.5 * alpha) # residual gating to avoid collapse (final value is in range [0.5, 1])

    def get_gate_stats(self) -> dict:
        """For monitoring: check if gates are learning"""
        return {
            'mean': self.alpha_mean.item(),
            'std': self.alpha_std.item(),
            'active_ratio': (self.last_alpha > 0.5).float().mean().item() if self.last_alpha is not None else 0.0
        }
