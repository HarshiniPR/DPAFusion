import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveSpatialFusionStage3(nn.Module):
    """
    Stage III: Adaptive Spatial Feature Fusion & Operator Execution Engine
    Consumes:
        - Stage 1 Features: Fr, Ft, Fu, Sc
        - Stage 2 Decisions: c_rgb, c_th, c_comp, z_w, alpha_op, g_int
    Outputs:
        - F_fused: Adaptive fused feature map (B, 128, 64, 64)
        - W_rgb, W_th: Continuous spatial region weighting maps
    """
    def __init__(self, in_channels=128, num_operators=4):
        super().__init__()
        self.in_channels = in_channels
        self.num_operators = num_operators

        # Operator Primitive 3: Convolutional Cross-Fusion
        self.op_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Operator Primitive 4: Attention-Gated Fusion
        self.op_att_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def upsample_zw_to_grid(self, z_w, height=64, width=64):
        """ Expands latent z_w (B, 12) deterministically to spatial grid (B, 1, H, W) """
        B = z_w.shape[0]
        grid_raw = z_w.view(B, 1, 3, 4)
        return F.interpolate(grid_raw, size=(height, width), mode='bilinear', align_corners=False)

    def execute_primitive_operators(self, Fr, Ft, W_rgb, W_th):
        """ Evaluates candidate fusion primitives """
        # Op 1: Weighted Sum Fusion
        F_op1 = (W_rgb * Fr) + (W_th * Ft)
        # Op 2: Element-wise Max Selection
        F_op2 = torch.max(Fr, Ft)
        # Op 3: Convolutional Cross-Fusion
        F_op3 = self.op_conv(torch.cat([Fr, Ft], dim=1))
        # Op 4: Attention-Gated Fusion
        F_op4 = (self.op_att_gate(Fr) * Ft) + Fr

        return torch.stack([F_op1, F_op2, F_op3, F_op4], dim=1)

    def forward(self, Fr, Ft, Fu, Sc, actions):
        B, C, H, W = Fr.shape

        # Step A: Generate Continuous Spatial Weights
        c_rgb = actions['c_rgb'].unsqueeze(-1).unsqueeze(-1)
        c_th = actions['c_th'].unsqueeze(-1).unsqueeze(-1)
        c_comp = actions['c_comp'].unsqueeze(-1).unsqueeze(-1)
        g_int = actions['g_int'].unsqueeze(-1).unsqueeze(-1)

        bias = torch.logit(c_rgb.clamp(1e-4, 1.0 - 1e-4)) - torch.logit(c_th.clamp(1e-4, 1.0 - 1e-4))
        z_w_expanded = self.upsample_zw_to_grid(actions['z_w'], height=H, width=W)
        
        W_rgb = torch.sigmoid(bias + z_w_expanded)
        W_th = 1.0 - W_rgb

        # Step B: Primitive Operator Mixture
        op_stack = self.execute_primitive_operators(Fr, Ft, W_rgb, W_th)
        alpha_op = actions['alpha_op'].view(B, self.num_operators, 1, 1, 1)
        F_base = torch.sum(alpha_op * op_stack, dim=1)

        # Step C: Gated Complementarity Injection
        Sc_broadcast = Sc.unsqueeze(-1).unsqueeze(-1)
        F_comp = g_int * c_comp * (Sc_broadcast * Fu)

        # Step D: Final Fusion Output
        F_fused = F_base + F_comp

        return F_fused, W_rgb, W_th