import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        # Use reflection padding to eliminate edge ringing
        self.pad = nn.ReflectionPad2d(padding)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.pointwise(self.depthwise(self.pad(x)))))

class FiLMBlock(nn.Module):
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
        gamma = torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = torch.tanh(beta).unsqueeze(-1).unsqueeze(-1)
        
        out = self.conv(x)
        return (1.0 + gamma) * out + beta

class DCLARNetGenerator(nn.Module):
    def __init__(self, in_channels=128, cond_dim=2):
        super().__init__()
        
        self.in_norm = nn.InstanceNorm2d(in_channels, affine=True)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.refine_conv = nn.Sequential(
            DepthwiseSeparableConv(64, 64),
            DepthwiseSeparableConv(64, 64)
        )

        self.film_base = FiLMBlock(64, cond_dim)
        
        self.detail_proj = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            DepthwiseSeparableConv(64, 64)
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparableConv(64, 32)
        )
        self.res1 = DepthwiseSeparableConv(32, 32)
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparableConv(32, 16)
        )
        
        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 16, kernel_size=3, bias=False),
            nn.InstanceNorm2d(16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Tanh()
        )

    def extract_gradients(self, x):
        gx = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        gy = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        return F.pad(gx, (0, 1, 0, 0), mode='replicate') + F.pad(gy, (0, 0, 0, 1), mode='replicate')

    def forward(self, F_fused, actions):
        r_lvl = actions['r_lvl']
        d_pres = actions['d_pres']
        d_rec = torch.cat([r_lvl, d_pres], dim=-1)

        F_norm = self.in_norm(F_fused)
        F_0 = self.bottleneck(F_norm)

        R_psi = self.refine_conv(F_0)
        r_lvl_bc = r_lvl.unsqueeze(-1).unsqueeze(-1)
        F_refined = F_0 + (r_lvl_bc * R_psi)

        F_base = self.film_base(F_refined, d_rec)
        G_feat = self.extract_gradients(F_refined)
        F_detail = self.detail_proj(torch.cat([F_refined, G_feat], dim=1))

        omega_D = (d_pres + 1.0) / 2.0
        omega_D_bc = omega_D.unsqueeze(-1).unsqueeze(-1)
        F_merged = F_base + (omega_D_bc * F_detail)

        x = self.up1(F_merged)
        x = x + self.res1(x)
        x = self.up2(x)
        
        I_out = self.head(x)
        I_fused = (I_out + 1.0) / 2.0
        return I_fused


class LightweightPatchDiscriminator(nn.Module):
    def __init__(self, in_channels=1, ndf=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.net(x)


class DCLARDualDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.D_th = LightweightPatchDiscriminator(in_channels=1, ndf=32)
        self.D_vis = LightweightPatchDiscriminator(in_channels=1, ndf=32)

    def forward_th(self, x):
        return self.D_th(x)

    def forward_vis(self, x):
        return self.D_vis(x)