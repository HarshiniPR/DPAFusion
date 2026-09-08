import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveStage2Reward(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Ir, It, F_fused, W_th, actions):
        """
        Calculates context-dependent reward to break mode collapse.
        """
        vis_gray = 0.2989 * Ir[:, 0:1] + 0.5870 * Ir[:, 1:2] + 0.1140 * Ir[:, 2:3]
        
        # 1. Thermal Target Alignment vs. Background Leakage
        # Pedestrians are typically high-intensity (>0.45) in normalized thermal
        target_mask = (It > 0.45).float()
        bg_mask = (It <= 0.45).float()
        
        # Reward high W_th on pedestrians; penalize W_th on cold background
        th_target_overlap = (W_th * target_mask).sum(dim=[1, 2, 3]) / (target_mask.sum(dim=[1, 2, 3]) + 1e-5)
        th_bg_leakage = (W_th * bg_mask).sum(dim=[1, 2, 3]) / (bg_mask.sum(dim=[1, 2, 3]) + 1e-5)
        R_saliency = 2.5 * th_target_overlap - 1.2 * th_bg_leakage

        # 2. Edge & Gradient Information Density
        gx = torch.abs(F_fused[:, :, :, :-1] - F_fused[:, :, :, 1:]).mean(dim=[1, 2, 3])
        gy = torch.abs(F_fused[:, :, :-1, :] - F_fused[:, :, 1:, :]).mean(dim=[1, 2, 3])
        R_detail = 1.8 * torch.tanh((gx + gy) * 6.0)

        # 3. Illumination-Modality Matching
        # In darkness, penalize selecting purely visible-dominated operators
        mean_vis = vis_gray.mean(dim=[1, 2, 3])
        op_idx = actions['op_idx']
        
        is_dark = mean_vis < 0.22
        # Op 0: Base sum, Op 1: Thermal priority, Op 2: Difference enhance, Op 3: Visible priority
        dark_reward = torch.where((op_idx == 1) | (op_idx == 2), 1.0, -0.6)
        bright_reward = torch.where((op_idx == 0) | (op_idx == 3), 0.8, -0.3)
        R_context = torch.where(is_dark, dark_reward, bright_reward)

        # 4. Anti-saturation penalty on continuous detail preservation
        d_pres = actions['d_pres'].squeeze(-1)
        R_penalty = torch.relu(torch.abs(d_pres) - 0.80) * 3.0

        total_reward = R_saliency + R_detail + R_context - R_penalty
        return total_reward.detach()