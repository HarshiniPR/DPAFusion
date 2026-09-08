import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveStage2Reward(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Ir, It, F_fused, W_th, actions):
        vis_gray = 0.2989 * Ir[:, 0:1] + 0.5870 * Ir[:, 1:2] + 0.1140 * Ir[:, 2:3]
        
        # 1. Thermal Saliency Alignment
        target_mask = (It > 0.45).float()
        bg_mask = (It <= 0.45).float()
        
        th_target_overlap = (W_th * target_mask).sum(dim=[1, 2, 3]) / (target_mask.sum(dim=[1, 2, 3]) + 1e-5)
        th_bg_leakage = (W_th * bg_mask).sum(dim=[1, 2, 3]) / (bg_mask.sum(dim=[1, 2, 3]) + 1e-5)
        R_saliency = 2.5 * th_target_overlap - 1.2 * th_bg_leakage

        # 2. Detail and Gradient Preservation
        gx = torch.abs(F_fused[:, :, :, :-1] - F_fused[:, :, :, 1:]).mean(dim=[1, 2, 3])
        gy = torch.abs(F_fused[:, :, :-1, :] - F_fused[:, :, 1:, :]).mean(dim=[1, 2, 3])
        R_detail = 1.8 * torch.tanh((gx + gy) * 6.0)

        # 3. Contextual Balance
        mean_vis = vis_gray.mean(dim=[1, 2, 3])
        c_th = actions['c_th'].squeeze(-1)
        c_rgb = actions['c_rgb'].squeeze(-1)
        
        # Reward higher thermal ratio in dark scenes; reward higher visible ratio in bright scenes
        is_dark = mean_vis < 0.22
        R_balance = torch.where(is_dark, c_th - c_rgb, c_rgb - c_th) * 0.8

        # 4. Anti-saturation Penalty on d_pres
        d_pres = actions['d_pres'].squeeze(-1)
        R_penalty = torch.relu(torch.abs(d_pres) - 0.80) * 3.0

        total_reward = R_saliency + R_detail + R_balance - R_penalty
        return total_reward.detach()