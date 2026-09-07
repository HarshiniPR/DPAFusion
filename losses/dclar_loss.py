import torch
import torch.nn as nn
import torch.nn.functional as F

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.channel = 1
        self.register_buffer('window', self._create_window(window_size))

    def _gaussian(self, window_size, sigma=1.5):
        gauss = torch.exp(torch.tensor([-(x - window_size // 2) ** 2 / float(2 * sigma ** 2) for x in range(window_size)]))
        return gauss / gauss.sum()

    def _create_window(self, window_size):
        _1D_window = self._gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window

    def forward(self, img1, img2):
        window = self.window.to(img1.device)
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2)

        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2) - mu1_mu2

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1.0 - ssim_map.mean()


class DCLARLossSuite(nn.Module):
    def __init__(self, lambda_adv=0.01, lambda_th=1.0, lambda_vis=1.0, lambda_str=1.0, lambda_red=0.2):
        super().__init__()
        self.lambda_adv = lambda_adv
        self.lambda_th = lambda_th
        self.lambda_vis = lambda_vis
        self.lambda_str = lambda_str
        self.lambda_red = lambda_red
        self.ssim_loss = SSIMLoss()
        self.mse = nn.MSELoss()

    def gradient(self, img):
        gx = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
        gy = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
        return F.pad(gx, (0, 1, 0, 0)) + F.pad(gy, (0, 0, 0, 1))

    def normalize(self, x):
        b = x.size(0)
        x_flat = x.view(b, -1)
        x_min = x_flat.min(dim=-1, keepdim=True)[0].view(b, 1, 1, 1)
        x_max = x_flat.max(dim=-1, keepdim=True)[0].view(b, 1, 1, 1)
        return (x - x_min) / (x_max - x_min + 1e-6)

    # LSGAN (Least Squares GAN) Objective - Stable, bounded, non-saturating
    def discriminator_lsgan_loss(self, d_real, d_fake):
        real_target = torch.ones_like(d_real)
        fake_target = torch.zeros_like(d_fake)
        loss_real = self.mse(d_real, real_target)
        loss_fake = self.mse(d_fake, fake_target)
        return 0.5 * (loss_real + loss_fake)

    def generator_lsgan_loss(self, d_fake_th, d_fake_vis):
        target = torch.ones_like(d_fake_th)
        loss_g_th = self.mse(d_fake_th, target)
        loss_g_vis = self.mse(d_fake_vis, target)
        return loss_g_th + loss_g_vis

    def compute_generator_losses(self, I_fused, I_vis_gray, I_th, W_th, S_c, d_fake_th=None, d_fake_vis=None, warmup=False):
        # 1. Thermal Saliency Loss
        G_th = self.gradient(I_th)
        S_th = self.normalize(I_th + 0.5 * G_th)
        L_th_sal = torch.mean(S_th * torch.abs(I_fused - I_th))

        # 2. Visible Detail Loss
        G_vis = self.gradient(I_vis_gray)
        G_fused = self.gradient(I_fused)
        S_vis = self.normalize(G_vis)
        L_grad = torch.mean(S_vis * torch.abs(G_fused - G_vis))

        # 3. Structural Loss
        W_th_up = F.interpolate(W_th, size=I_fused.shape[2:], mode='bilinear', align_corners=False)
        W_vis_up = 1.0 - W_th_up
        I_ref = (W_th_up * I_th) + (W_vis_up * I_vis_gray)
        L_ssim = self.ssim_loss(I_fused, I_ref)

        # 4. Redundancy Suppression
        S_c_mean = S_c.mean(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        R_c = 1.0 - S_c_mean
        L_red = torch.mean(R_c * torch.abs(I_fused - I_ref))

        # 5. Adversarial Loss (LSGAN)
        if warmup or d_fake_th is None or d_fake_vis is None:
            L_adv = torch.tensor(0.0, device=I_fused.device)
            total_loss = (self.lambda_th * L_th_sal) + (self.lambda_vis * L_grad) + \
                         (self.lambda_str * L_ssim) + (self.lambda_red * L_red)
        else:
            L_adv = self.generator_lsgan_loss(d_fake_th, d_fake_vis)
            total_loss = (self.lambda_adv * L_adv) + (self.lambda_th * L_th_sal) + \
                         (self.lambda_vis * L_grad) + (self.lambda_str * L_ssim) + \
                         (self.lambda_red * L_red)

        loss_dict = {
            'total_loss': total_loss,
            'L_adv': L_adv.item() if isinstance(L_adv, torch.Tensor) else L_adv,
            'L_th_sal': L_th_sal.item(),
            'L_grad': L_grad.item(),
            'L_ssim': L_ssim.item(),
            'L_red': L_red.item()
        }
        return total_loss, loss_dict