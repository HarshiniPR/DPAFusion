import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """ Lightweight Depthwise Separable Convolution """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))

class FiLMBlock(nn.Module):
    """ Feature-wise Linear Modulation based on PPO D_rec [r_lvl, d_pres] """
    def __init__(self, channels, cond_dim=2):
        super().__init__()
        self.conv = DepthwiseSeparableConv(channels, channels)
        self.film_gen = nn.Sequential(
            nn.Linear(cond_dim, channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channels * 2, channels * 2)
        )

    def forward(self, x, d_rec):
        gamma_beta = self.film_gen(d_rec)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        out = self.conv(x)
        return (gamma * out) + beta

class DCLARNetGenerator(nn.Module):
    """
    Stage IV: Decision-Conditioned Lightweight Adversarial Reconstruction Network
    Input: F_fused (B, 128, 64, 64) and actions [r_lvl, d_pres]
    Output: I_fused (B, 1, 256, 256)
    """
    def __init__(self, in_channels=128, cond_dim=2):
        super().__init__()
        
        # 1. Bottleneck: 128 -> 64
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 2. Residual Refinement Module R_psi
        self.refine_conv = nn.Sequential(
            DepthwiseSeparableConv(64, 64),
            DepthwiseSeparableConv(64, 64)
        )

        # 3. FiLM Conditioning Generator
        self.film_base = FiLMBlock(64, cond_dim)
        
        # 4. Detail Path: Gradient Feature Fusion
        self.detail_proj = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            DepthwiseSeparableConv(64, 64)
        )

        # 5. Shared Progressive Upsampling Decoder
        # 64x64 -> 128x128
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparableConv(64, 32)
        )
        self.res1 = DepthwiseSeparableConv(32, 32)
        
        # 128x128 -> 256x256
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparableConv(32, 16)
        )
        
        # Final reconstruction head -> Single channel grayscale intensity
        self.head = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def extract_gradients(self, x):
        gx = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        gy = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        return gx + gy

    def forward(self, F_fused, actions):
        r_lvl = actions['r_lvl']      # (B, 1) in (0, 1)
        d_pres = actions['d_pres']    # (B, 1) in (-1, 1)
        d_rec = torch.cat([r_lvl, d_pres], dim=-1) # (B, 2)

        # Bottleneck to 64 channels
        F_0 = self.bottleneck(F_fused)

        # 4.3 Explicit Residual Reconstruction Control
        # F_refined = F_fused + r_lvl * R_psi
        R_psi = self.refine_conv(F_0)
        r_lvl_bc = r_lvl.unsqueeze(-1).unsqueeze(-1)
        F_refined = F_0 + (r_lvl_bc * R_psi)

        # 4.5 Path A: Base Reconstruction
        F_base = self.film_base(F_refined, d_rec)

        # 4.5 Path B: Detail Reconstruction (Frequency/Gradient Aware)
        G_feat = self.extract_gradients(F_refined)
        F_detail = self.detail_proj(torch.cat([F_refined, G_feat], dim=1))

        # 4.6 Explicit Detail-Preservation Control
        # omega_D = (d_pres + 1) / 2
        omega_D = (d_pres + 1.0) / 2.0
        omega_D_bc = omega_D.unsqueeze(-1).unsqueeze(-1)
        
        # Adaptive Merge
        F_merged = F_base + (omega_D_bc * F_detail)

        # Progressive Upsampling
        x = self.up1(F_merged)
        x = x + self.res1(x)
        x = self.up2(x)
        
        # Final image reconstruction
        I_fused = self.head(x) # [B, 1, 256, 256] in [0, 1]
        return I_fused


class LightweightPatchDiscriminator(nn.Module):
    """ Lightweight PatchGAN with spectral normalization """
    def __init__(self, in_channels=1, ndf=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.net(x)


class DCLARDualDiscriminator(nn.Module):
    """ Dual Discriminators: D_th and D_vis """
    def __init__(self):
        super().__init__()
        self.D_th = LightweightPatchDiscriminator(in_channels=1, ndf=32)
        self.D_vis = LightweightPatchDiscriminator(in_channels=1, ndf=32)

    def forward_th(self, x):
        return self.D_th(x)

    def forward_vis(self, x):
        return self.D_vis(x)