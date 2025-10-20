import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionGate(nn.Module):
    """
    Cross-temporal attention gate for change detection.
    
    Uses post-disaster features to query pre-disaster features,
    creating change-aware attention maps.
    
    Key idea: "What in the PRE image corresponds to changed regions in POST?"
    
    Args:
        channels: Number of input channels
        reduction: Channel reduction factor for attention computation
        use_scale: Whether to use scaled dot-product attention
    """
    def __init__(self, channels: int, reduction: int = 8, use_scale: bool = True):
        super().__init__()
        
        self.channels = channels
        self.inter_channels = max(channels // reduction, 1)
        self.use_scale = use_scale
        
        # Query: from post-disaster (what to look for)
        self.query_conv = nn.Sequential(
            nn.Conv2d(channels, self.inter_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, self.inter_channels),
            nn.ReLU(inplace=True)
        )
        
        # Key: from pre-disaster (where it was)
        self.key_conv = nn.Sequential(
            nn.Conv2d(channels, self.inter_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, self.inter_channels),
            nn.ReLU(inplace=True)
        )
        
        # Value: direct from pre-disaster
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        
        # Output projection
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, channels)
        )
        
        # Learnable scaling factor (starts near 0 for residual connection)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # For monitoring
        self.last_attn: torch.Tensor | None = None
        self.register_buffer('attn_entropy', torch.tensor(0.0))
        
    def forward(self, pre_feat: torch.Tensor, post_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pre_feat: [B, C, H, W] - pre-disaster features
            post_feat: [B, C, H, W] - post-disaster features
            
        Returns:
            change_feat: [B, C, H, W] - change-enhanced pre-features
        """
        B, C, H, W = pre_feat.shape
        
        # Generate query from post (what changed?)
        query = self.query_conv(post_feat)  # [B, C', H, W]
        query = query.view(B, self.inter_channels, -1).permute(0, 2, 1)  # [B, HW, C']
        
        # Generate key from pre (original state)
        key = self.key_conv(pre_feat)  # [B, C', H, W]
        key = key.view(B, self.inter_channels, -1)  # [B, C', HW]
        
        # Generate value from pre
        value = self.value_conv(pre_feat)  # [B, C, H, W]
        value = value.view(B, C, -1)  # [B, C, HW]
        
        # Compute attention: query(post) × key(pre)
        attention = torch.bmm(query, key)  # [B, HW, HW]
        
        # Optional scaling for stability
        if self.use_scale:
            attention = attention / (self.inter_channels ** 0.5)
        
        attention = torch.softmax(attention, dim=-1)  # [B, HW, HW]
        
        # Track attention statistics
        with torch.no_grad():
            self.last_attn = attention
            # Compute entropy to monitor if attention is peaky or diffuse
            eps = 1e-8
            self.attn_entropy = -(attention * torch.log(attention + eps)).sum(dim=-1).mean()
        
        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C, HW]
        out = out.view(B, C, H, W)  # [B, C, H, W]
        
        # Project output
        out = self.out_conv(out)
        
        # Residual connection with learnable gate
        change_feat = pre_feat + self.gamma * out
        
        return change_feat
    
    def get_attention_stats(self) -> dict:
        """For debugging: check attention patterns"""
        return {
            'gamma': self.gamma.item(),
            'entropy': self.attn_entropy.item(),
            'max_attn': self.last_attn.max().item() if self.last_attn is not None else 0.0,
        }
    
