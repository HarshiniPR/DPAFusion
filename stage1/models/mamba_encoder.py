import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionMambaBlock(nn.Module):
    def __init__(self, dim=128, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.d_inner = dim * expand
        self.norm = nn.LayerNorm(dim)
        
        self.in_proj = nn.Linear(dim, self.d_inner * 2)
        
        # FIX: Explicit symmetric padding calculation for Conv1d
        # For d_conv=4, padding_left=1, padding_right=2 maintains sequence length 4096
        self.d_conv = d_conv
        self.pad_left = (d_conv - 1) // 2
        self.pad_right = d_conv - 1 - self.pad_left
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, 
            out_channels=self.d_inner,
            kernel_size=d_conv, 
            stride=1,
            padding=0, # Handled manually via F.pad
            groups=self.d_inner
        )
        
        self.x_proj = nn.Linear(self.d_inner, d_state * 2)
        self.dt_proj = nn.Linear(d_state, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2) # [B, 4096, 128]
        residual = x_flat
        x_norm = self.norm(x_flat)
        
        # Linear Expansion & Split
        xz = self.in_proj(x_norm)
        x_proj, z = xz.chunk(2, dim=-1) # Each [B, 4096, 256]
        
        # 1D Conv over sequence length with asymmetric padding
        x_conv = x_proj.transpose(1, 2) # [B, 256, 4096]
        x_conv = F.pad(x_conv, (self.pad_left, self.pad_right)) # Pad sequence dim
        x_conv = F.silu(self.conv1d(x_conv)).transpose(1, 2) # Back to [B, 4096, 256]
        
        # Selective Gating
        dt_x = self.x_proj(x_conv)
        dt = F.softplus(self.dt_proj(dt_x[:, :, :16]))
        
        # Tensor shapes now match: [B, 4096, 256] * [B, 4096, 256] * [B, 4096, 256]
        y = x_conv * torch.sigmoid(dt) * F.silu(z)
        
        out = self.dropout(self.out_proj(y)) + residual
        return out.transpose(1, 2).view(B, C, H, W)


class DualMambaEncoder(nn.Module):
    def __init__(self, depth=4, dim=128):
        super().__init__()
        self.rgb_mamba = nn.Sequential(*[VisionMambaBlock(dim=dim) for _ in range(depth)])
        self.ir_mamba = nn.Sequential(*[VisionMambaBlock(dim=dim) for _ in range(depth)])

    def forward(self, Fr0, Ft0):
        Fr = self.rgb_mamba(Fr0)
        Ft = self.ir_mamba(Ft0)
        return Fr, Ft