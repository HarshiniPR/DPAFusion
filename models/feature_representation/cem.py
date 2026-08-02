import torch
import torch.nn as nn

class ComplementarityEstimationModule(nn.Module):
    def __init__(self, in_channels=128, use_multiplication=False):
        super().__init__()
        self.use_multiplication = use_multiplication
        descriptor_dim = in_channels * 4 if use_multiplication else in_channels * 3
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(descriptor_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, in_channels),
            nn.Sigmoid()
        )

    def forward(self, Fr, Ft):
        zr = self.gap(Fr).squeeze(-1).squeeze(-1)
        zt = self.gap(Ft).squeeze(-1).squeeze(-1)
        
        if self.use_multiplication:
            Dc = torch.cat([zr, zt, torch.abs(zr - zt), zr * zt], dim=-1)
        else:
            Dc = torch.cat([zr, zt, torch.abs(zr - zt)], dim=-1)
            
        Sc = self.mlp(Dc)
        return Sc