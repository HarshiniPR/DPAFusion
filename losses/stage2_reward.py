import torch
import torch.nn as nn
import torch.nn.functional as F

class FullMultiObjectiveReward(nn.Module):
    """
    Computes Section 7 Full Reward Function:
    R_t = lambda1*DeltaDet + lambda2*Q_img + lambda3*ComplUtil - lambda4*Penalty - lambda5*ComputeCost
    """
    def __init__(self, lambda1=1.0, lambda2=0.3, lambda3=0.2, lambda4=0.1, lambda5=0.1):
        super().__init__()
        self.l1, self.l2, self.l3, self.l4, self.l5 = lambda1, lambda2, lambda3, lambda4, lambda5
        
        # Fixed precomputed FLOP costs for K=4 primitive operators (normalized [0, 1])
        # [0: weighted-sum (cheap), 1: max-select, 2: conv-fusion, 3: attention-gated (expensive)]
        self.register_buffer('op_costs', torch.tensor([0.1, 0.2, 0.5, 1.0]))

    def compute_image_quality(self, F_fused):
        # Q_img proxy (Contrast + StdDev)
        contrast = torch.std(F_fused, dim=[-2, -1]).mean(dim=-1)
        return contrast

    def compute_compl_util(self, W_rgb, Sc):
        # Measure alignment between spatial weight direction and Sc map
        W_centered = W_rgb - 0.5
        Sc_centered = Sc - 0.5
        cos_sim = F.cosine_similarity(W_centered.mean(dim=[-2, -1]), Sc_centered, dim=-1)
        return cos_sim

    def compute_penalties(self, W_rgb):
        # Spatial Total Variation (TV) penalty
        tv_h = torch.abs(W_rgb[:, :, 1:, :] - W_rgb[:, :, :-1, :]).mean()
        tv_w = torch.abs(W_rgb[:, :, :, 1:] - W_rgb[:, :, :, :-1]).mean()
        tv_penalty = tv_h + tv_w
        
        # Modality collapse penalty
        mean_w = torch.abs(W_rgb.mean() - 0.5)
        collapse_penalty = F.relu(mean_w - 0.3)
        
        return 0.05 * tv_penalty + 0.05 * collapse_penalty

    def forward(self, F_fused, W_rgb, Sc, alpha_op):
        # 1. Perception/Contrast surrogate
        q_img = self.compute_image_quality(F_fused)
        
        # 2. Complementarity Utilization
        compl_util = self.compute_compl_util(W_rgb, Sc)
        
        # 3. Penalties
        penalty = self.compute_penalties(W_rgb)
        
        # 4. Operator Compute Cost Penalty
        compute_cost = torch.sum(alpha_op * self.op_costs, dim=-1)

        total_reward = (self.l2 * q_img) + (self.l3 * compl_util) - (self.l4 * penalty) - (self.l5 * compute_cost)
        return total_reward.unsqueeze(-1)