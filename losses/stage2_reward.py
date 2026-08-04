import torch
import torch.nn as nn
import torch.nn.functional as F

class DecisionAwareReward(nn.Module):
    """
    Computes surrogate task reward evaluating feature map quality for downstream detection:
    R = w1 * TargetContrast + w2 * StructuralGradients + w3 * FeatureEnergy
    """
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_x', kernel_x)
        self.register_buffer('sobel_y', kernel_y)

    def compute_gradient_map(self, x):
        x_mean = torch.mean(x, dim=1, keepdim=True)
        grad_x = F.conv2d(x_mean, self.sobel_x, padding=1)
        grad_y = F.conv2d(x_mean, self.sobel_y, padding=1)
        return torch.abs(grad_x) + torch.abs(grad_y)

    def forward(self, F_fused, Fr, Ft):
        # 1. Target Contrast (Standard deviation across spatial dimensions)
        contrast_score = torch.std(F_fused, dim=[-2, -1]).mean(dim=-1)
        
        # 2. Structural Edge Quality vs Modality Max Gradients
        grad_fused = self.compute_gradient_map(F_fused)
        grad_target = torch.max(self.compute_gradient_map(Fr), self.compute_gradient_map(Ft))
        structure_score = 1.0 / (1.0 + F.l1_loss(grad_fused, grad_target, reduction='none').mean(dim=[-3, -2, -1]))
        
        # 3. Feature Energy (L2 Norm)
        energy_score = torch.norm(F_fused, p=2, dim=[-2, -1]).mean(dim=-1) / 1000.0
        
        total_reward = (2.0 * contrast_score) + (1.5 * structure_score) + (0.5 * energy_score)
        return total_reward.unsqueeze(-1)