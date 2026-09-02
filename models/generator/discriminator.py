import torch
import torch.nn as nn

class PatchDiscriminator(nn.Module):
    """
    Standard 70x70 PatchGAN Discriminator.
    Evaluates real vs. fake local patches across source modalities.
    """
    def __init__(self, in_channels=3, ndf=64):
        super().__init__()
        self.model = nn.Sequential(
            # Input: (B, in_channels, 256, 256)
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, ndf, 128, 128)
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, ndf * 2, 64, 64)
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, ndf * 4, 32, 32)
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Output patch predictions (B, 1, 31, 31)
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)


class DualDiscriminator(nn.Module):
    """
    Dual Discriminator System (TarDAL design):
        - D_rgb: Discriminates between I_fused and I_rgb (3 channels)
        - D_th:  Discriminates between I_fused and I_th (replicated to 3 channels)
    """
    def __init__(self):
        super().__init__()
        self.D_rgb = PatchDiscriminator(in_channels=3)
        self.D_th = PatchDiscriminator(in_channels=3)

    def forward_rgb(self, x):
        return self.D_rgb(x)

    def forward_th(self, x):
        return self.D_th(x)