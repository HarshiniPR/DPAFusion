import torch
import torch.nn as nn

class ShallowCNNExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> 64x128x128
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> 128x64x64
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.stem(x)