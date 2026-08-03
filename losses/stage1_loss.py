import torch
import torch.nn as nn
import torch.nn.functional as F

class Stage1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('sobel_x', kernel_x)
        self.register_buffer('sobel_y', kernel_y)

    def compute_gradient(self, x):
        x_mean = torch.mean(x, dim=1, keepdim=True)
        grad_x = F.conv2d(x_mean, self.sobel_x, padding=1)
        grad_y = F.conv2d(x_mean, self.sobel_y, padding=1)
        return torch.abs(grad_x) + torch.abs(grad_y)

    def forward(self, Fu, Fr, Ft, Sc):
        # 1. Feature Decorrelation
        z_rgb = F.normalize(Fr.mean(dim=[-2, -1]), dim=-1)
        z_ir = F.normalize(Ft.mean(dim=[-2, -1]), dim=-1)
        cos_sim = torch.sum(z_rgb * z_ir, dim=-1)
        loss_decorr = torch.mean(F.relu(cos_sim - 0.3))

        # 2. Stable CEM Activity Loss (Prevents saturation to hard 0 or 1)
        # Keeps average gating around 0.5 and prevents zero-variance collapse
        sc_mean_penalty = torch.abs(torch.mean(Sc) - 0.5)
        sc_variance_penalty = F.relu(0.05 - torch.std(Sc))  # Ensures dynamic spread
        loss_comp = sc_mean_penalty + sc_variance_penalty

        # 3. Structure Preservation Loss
        grad_rgb = self.compute_gradient(Fr)
        grad_ir = self.compute_gradient(Ft)
        grad_fu = self.compute_gradient(Fu)
        grad_target = torch.max(grad_rgb, grad_ir)
        loss_structure = F.l1_loss(grad_fu, grad_target)

        # 4. Feature Information Preservation
        loss_info = F.l1_loss(Fu, (Fr + Ft) / 2.0)

        total_loss = (1.0 * loss_structure) + (0.5 * loss_decorr) + (0.2 * loss_comp) + (1.0 * loss_info)
        
        return total_loss, {
            'loss_decorr': loss_decorr.item(),
            'loss_comp': loss_comp.item(),
            'loss_structure': loss_structure.item(),
            'loss_info': loss_info.item()
        }