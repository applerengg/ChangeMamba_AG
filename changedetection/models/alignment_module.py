from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class AlignmentArgs:
    enabled: bool
    stages: tuple[int, ...]
    mid_ch: int 

class AlignmentHead(nn.Module):
    """
    Predicts 2D flow (dx, dy) on feature maps and warps PRE -> POST.
    No supervision needed. Trains end-to-end via task losses.

    Input:  f_pre, f_post: [B, C, H, W], float32/16 ok
    Output: f_pre_warp:   [B, C, H, W], flow: [B, 2, H, W]  (pixels at this scale)
    """
    def __init__(self, in_ch: int, mid_ch: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch * 2, mid_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 2, 3, padding=1, bias=True),
        )

    @staticmethod
    def _make_base_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        gy = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        gx = torch.linspace(-1, 1, w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij") # [H,W]
        base = torch.stack((grid_x, grid_y), dim=-1) # [H,W,2]
        return base

    def forward(self, f_pre: torch.Tensor, f_post: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = f_pre.shape
        flow: torch.Tensor = self.net(torch.cat([f_pre, f_post], dim=1)) # [B,2,H,W]

        # normalize pixel flow to [-1,1] grid coordinates
        nx = flow[:, 0] / (W / 2.0) # [B,H,W]
        ny = flow[:, 1] / (H / 2.0) # [B,H,W]

        base = self._make_base_grid(H, W, f_pre.device, f_pre.dtype) # [H,W,2]
        base = base.unsqueeze(0).expand(B, H, W, 2) # [B,H,W,2]
        grid = torch.stack((base[..., 0] + nx, base[..., 1] + ny), dim=-1) # [B,H,W,2]

        f_pre_warp = F.grid_sample(
            f_pre, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        return f_pre_warp, flow


@torch.inference_mode()
def warp_image(img: torch.Tensor, flow: torch.Tensor, align_corners: bool = True):
    """
    :param pre_img: the image to be warped. In ChangeMamba_AG, pre-event features is warped to match post-event features. <br/>
    shape: [1, 3, H, W] (RGB channels, batch of single object) 
    :type pre_img: torch.Tensor

    :param flow_field: the flow field / offset map obtained from Alignment Module. shape: [1, 2, Hf, Wf] (2-channel, batch of single object)
    :type flow_field: torch.Tensor

    :param align_corners: align_corners
    :type align_corners: bool
    """
    device = img.device
    dtype = img.dtype
    _, _, H, W = img.shape
    _, _, Hf, Wf = flow.shape

    flow = flow.to(device=device, dtype=dtype)

    flow_upsampled = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=align_corners)

    # Scale displacement from feature pixels to image pixels
    scale_x = W / float(Wf)
    scale_y = H / float(Hf)
    flow_upsampled[:, 0] = flow_upsampled[:, 0] * scale_x
    flow_upsampled[:, 1] = flow_upsampled[:, 1] * scale_y

    nx = flow_upsampled[:, 0] / (W / 2.0) # [1, H, W]
    ny = flow_upsampled[:, 1] / (H / 2.0) # [1, H, W]

    base = AlignmentHead._make_base_grid(H, W, device, dtype).unsqueeze(0)  # [1, H, W, 2]
    grid = torch.stack((base[..., 0] + nx, base[..., 1] + ny), dim=-1)  # [1, H, W, 2]

    warped = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=align_corners) # [1, 3, H, W] 
    return warped





