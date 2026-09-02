import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMBlock(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Residual Block.
    Modulates intermediate channel activations using Group C decisions: [r_lvl, d_pres].
    """
    def __init__(self, channels=128, cond_dim=2):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # Mapping network for FiLM affine parameters (gamma, beta)
        self.film_net = nn.Sequential(
            nn.Linear(cond_dim, channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channels * 2, channels * 2)
        )

    def forward(self, x, cond):
        # cond: [B, 2] -> gamma, beta: each [B, channels, 1, 1]
        film_params = self.film_net(cond)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        residual = x
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2, inplace=True)
        out = self.bn2(self.conv2(out))

        # Apply FiLM Affine Transformation: gamma * x + beta
        out = gamma * out + beta
        out = F.leaky_relu(out + residual, 0.2, inplace=True)
        return out


class FiLMGeneratorStage4(nn.Module):
    """
    Stage IV Generator:
    Takes F_fused (B, 128, 64, 64) and Group C decision vector [r_lvl, d_pres] (B, 2).
    Progressively upsamples to reconstruct I_fused (B, 3, 256, 256).
    """
    def __init__(self, in_channels=128, cond_dim=2, out_channels=3):
        super().__init__()
        
        # 1. Conditioning Residual Trunk (operates at 64x64)
        self.film_res1 = FiLMBlock(channels=in_channels, cond_dim=cond_dim)
        self.film_res2 = FiLMBlock(channels=in_channels, cond_dim=cond_dim)

        # 2. Upsampling Stage 1: 64x64 -> 128x128
        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels, 64 * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.film_res_up1 = FiLMBlock(channels=64, cond_dim=cond_dim)

        # 3. Upsampling Stage 2: 128x128 -> 256x256
        self.up2 = nn.Sequential(
            nn.Conv2d(64, 32 * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 4. Final Reconstruction Head (Output image in [-1, 1] or [0, 1])
        self.recon_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Produces normalized [-1, 1] output
        )

    def forward(self, F_fused, actions):
        """
        Args:
            F_fused: Feature map from Stage III (B, 128, 64, 64)
            actions: Dictionary containing Group C 'r_lvl' and 'd_pres'
        """
        r_lvl = actions['r_lvl']      # (B, 1)
        d_pres = actions['d_pres']    # (B, 1)
        cond = torch.cat([r_lvl, d_pres], dim=-1) # (B, 2)

        # FiLM-modulated feature refinement
        x = self.film_res1(F_fused, cond)
        x = self.film_res2(x, cond)

        # Progressive upsampling
        x = self.up1(x)
        x = self.film_res_up1(x, cond)
        x = self.up2(x)

        # Image generation
        I_fused = self.recon_head(x)
        # Shift Tanh [-1, 1] to normalized image space [0, 1]
        I_fused = (I_fused + 1.0) / 2.0
        return I_fused