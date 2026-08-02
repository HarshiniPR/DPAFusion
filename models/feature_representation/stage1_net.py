import torch
import torch.nn as nn
from .cnn_stem import ShallowCNNExtractor
from .mamba_encoder import DualMambaEncoder
from .cem import ComplementarityEstimationModule

class Stage1LHMRM(nn.Module):
    def __init__(self, use_multiplication_in_cem=False):
        super().__init__()
        self.cnn_rgb = ShallowCNNExtractor(in_channels=3)
        self.cnn_ir = ShallowCNNExtractor(in_channels=1)
        
        self.dual_mamba = DualMambaEncoder(depth=4, dim=128)
        self.cem = ComplementarityEstimationModule(in_channels=128, use_multiplication=use_multiplication_in_cem)
        
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, Ir, It):
        Fr0 = self.cnn_rgb(Ir)
        Ft0 = self.cnn_ir(It)
        
        Fr, Ft = self.dual_mamba(Fr0, Ft0)
        
        Sc = self.cem(Fr, Ft)
        Sc_broadcast = Sc.unsqueeze(-1).unsqueeze(-1)
        
        Fr_prime = Fr + self.alpha * (Sc_broadcast * Ft)
        Ft_prime = Ft + self.alpha * (Sc_broadcast * Fr)
        
        Fc = torch.cat([Fr_prime, Ft_prime], dim=1)
        Fu = self.fusion_conv(Fc)
        
        return Fu, Fr, Ft, Sc