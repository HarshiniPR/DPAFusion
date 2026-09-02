import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelGrad(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('weight_x', sobel_x)
        self.register_buffer('weight_y', sobel_y)

    def forward(self, img):
        if img.shape[1] == 3:
            img = 0.2989 * img[:, 0:1] + 0.5870 * img[:, 1:2] + 0.1140 * img[:, 2:3]
        grad_x = F.conv2d(img, self.weight_x, padding=1)
        grad_y = F.conv2d(img, self.weight_y, padding=1)
        return torch.abs(grad_x) + torch.abs(grad_y)


class Stage4LossSuite(nn.Module):
    """
    Stabilized GAN Loss Suite for Multi-Modal Fusion:
    L_G = lambda_adv * L_adv + lambda_grad * L_grad + lambda_pixel * L_pixel
    """
    def __init__(self, lambda_adv=0.05, lambda_grad=20.0, lambda_pixel=5.0):
        super().__init__()
        self.lambda_adv = lambda_adv
        self.lambda_grad = lambda_grad
        self.lambda_pixel = lambda_pixel
        self.sobel = SobelGrad()
        self.bce = nn.BCEWithLogitsLoss()

    def d_loss(self, d_real_out, d_fake_out):
        # One-sided label smoothing (0.9 instead of 1.0) prevents discriminator saturation
        real_labels = torch.full_like(d_real_out, 0.9)
        fake_labels = torch.zeros_like(d_fake_out)
        loss_real = self.bce(d_real_out, real_labels)
        loss_fake = self.bce(d_fake_out, fake_labels)
        return 0.5 * (loss_real + loss_fake)

    def g_adversarial_loss(self, d_fake_rgb, d_fake_th):
        real_labels_rgb = torch.ones_like(d_fake_rgb)
        real_labels_th = torch.ones_like(d_fake_th)
        raw_adv = self.bce(d_fake_rgb, real_labels_rgb) + self.bce(d_fake_th, real_labels_th)
        return self.lambda_adv * raw_adv, raw_adv

    def g_content_loss(self, I_fused, Ir, It):
        if It.shape[1] == 1:
            It_3c = It.repeat(1, 3, 1, 1)
        else:
            It_3c = It

        target_intensity = torch.max(Ir, It_3c)
        l_pixel = F.l1_loss(I_fused, target_intensity)

        grad_fused = self.sobel(I_fused)
        grad_r = self.sobel(Ir)
        grad_t = self.sobel(It_3c)
        target_grad = torch.max(grad_r, grad_t)
        l_grad = F.l1_loss(grad_fused, target_grad)

        total_content = (self.lambda_pixel * l_pixel) + (self.lambda_grad * l_grad)
        return total_content, l_pixel, l_grad