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
        
        self.d_conv = d_conv
        self.pad_left = (d_conv - 1) // 2
        self.pad_right = d_conv - 1 - self.pad_left
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, 
            out_channels=self.d_inner,
            kernel_size=d_conv, 
            stride=1,
            padding=0,
            groups=self.d_inner
        )
        
        self.x_proj = nn.Linear(self.d_inner, d_state * 2)
        self.dt_proj = nn.Linear(d_state, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        residual = x_flat
        x_norm = self.norm(x_flat)
        
        xz = self.in_proj(x_norm)
        x_proj, z = xz.chunk(2, dim=-1)
        
        x_conv = x_proj.transpose(1, 2)
        x_conv = F.pad(x_conv, (self.pad_left, self.pad_right))
        x_conv = F.silu(self.conv1d(x_conv)).transpose(1, 2)
        
        dt_x = self.x_proj(x_conv)
        dt = F.softplus(self.dt_proj(dt_x[:, :, :16]))
        
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