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
        # 1. Cross-Modal Contrastive / Cosine Distance Loss (Bounded in [0, 2])
        z_rgb = F.normalize(Fr.mean(dim=[-2, -1]), dim=-1)
        z_ir = F.normalize(Ft.mean(dim=[-2, -1]), dim=-1)
        # 1.0 - mean cosine similarity keeps it non-negative
        loss_contrastive = 1.0 - torch.mean(torch.sum(z_rgb * z_ir, dim=-1))

        # 2. CEM Loss (Encourages active channel gating, bounded in [0, 0.5])
        loss_comp = torch.mean(torch.abs(Sc - 0.5)) 

        # 3. Spatial Feature Cosine Loss
        loss_cosine = 1.0 - F.cosine_similarity(Fr, Ft, dim=1).mean()

        # 4. Structure Preservation Loss (L1 gradient loss, bounded >= 0)
        grad_rgb = self.compute_gradient(Fr)
        grad_ir = self.compute_gradient(Ft)
        grad_fu = self.compute_gradient(Fu)
        grad_target = torch.max(grad_rgb, grad_ir)
        loss_structure = F.l1_loss(grad_fu, grad_target)

        # Total Loss is strictly positive
        total_loss = (1.0 * loss_contrastive) + (0.5 * loss_comp) + (0.5 * loss_cosine) + (2.0 * loss_structure)
        
        return total_loss, {
            'loss_contrastive': loss_contrastive.item(),
            'loss_comp': loss_comp.item(),
            'loss_cosine': loss_cosine.item(),
            'loss_structure': loss_structure.item()
        }