import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveFusionReward(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Ir, It, F_fused, W_th, actions):
        """
        Rewards actions that adapt dynamically to illumination and thermal presence.
        """
        b = Ir.size(0)
        vis_gray = 0.2989 * Ir[:, 0:1] + 0.5870 * Ir[:, 1:2] + 0.1140 * Ir[:, 2:3]
        
        # 1. Thermal Salient Foreground Contrast Reward
        # High score when W_th activates on actual heat targets
        target_mask = (It > 0.45).float()
        th_overlap = (W_th * target_mask).sum(dim=[1, 2, 3]) / (target_mask.sum(dim=[1, 2, 3]) + 1e-5)
        bg_mask = (It <= 0.45).float()
        th_leakage = (W_th * bg_mask).sum(dim=[1, 2, 3]) / (bg_mask.sum(dim=[1, 2, 3]) + 1e-5)
        R_thermal = 2.0 * (th_overlap - 0.5 * th_leakage)

        # 2. Detail and Gradient Richness (Spatial Frequency)
        gx = torch.abs(F_fused[:, :, :, :-1] - F_fused[:, :, :, 1:]).mean(dim=[1, 2, 3])
        gy = torch.abs(F_fused[:, :, :-1, :] - F_fused[:, :, 1:, :]).mean(dim=[1, 2, 3])
        sf_score = torch.tanh((gx + gy) * 5.0)
        R_detail = 1.5 * sf_score

        # 3. Illumination Context Bonus
        mean_vis = vis_gray.mean(dim=[1, 2, 3])
        op_idx = actions['op_idx']
        # If dark (<0.20), reward selecting thermal-dominant operators (e.g. Op 1 or 2)
        # If bright (>0.50), reward balanced/visible operators (Op 0 or 3)
        R_context = torch.where(
            mean_vis < 0.20,
            torch.where((op_idx == 1) | (op_idx == 2), 1.0, -0.5),
            torch.where((op_idx == 0) | (op_idx == 3), 0.8, -0.2)
        )

        # 4. Anti-saturation Penalty on d_pres
        d_pres = actions['d_pres'].squeeze(-1)
        P_sat = torch.relu(torch.abs(d_pres) - 0.85) * 2.0

        total_reward = R_thermal + R_detail + R_context - P_sat
        return total_reward.detach()