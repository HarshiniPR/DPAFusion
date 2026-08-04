import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Normal, Categorical

class StateEncoder(nn.Module):
    """
    Encodes Stage 1 features and scene metrics into state vector s_t (dim=322):
    s_t = [GAP(Fu), GMP(Fu), CEM_pooled, phi_scene]
    """
    def __init__(self, in_channels=128, state_dim=322, hidden_dim=128):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.cem_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )

    def compute_scene_descriptors(self, Ir, It):
        B = Ir.shape[0]
        illum_rgb = torch.mean(Ir, dim=[-3, -2, -1], keepdim=True).view(B, 1)
        contrast_th = torch.std(It, dim=[-3, -2, -1], keepdim=True).view(B, 1)
        return torch.cat([illum_rgb, contrast_th], dim=-1)

    def forward(self, Fu, Sc, Ir, It):
        B = Fu.shape[0]
        
        Fu_det = Fu.detach()
        Sc_det = Sc.detach()
        
        gap_feat = self.gap(Fu_det).view(B, -1)                      # (B, 128)
        gmp_feat = self.gmp(Fu_det).view(B, -1)                      # (B, 128)
        
        Sc_grid = Sc_det.view(B, 1, 8, 16)
        cem_pooled = self.cem_pool(Sc_grid).view(B, -1)              # (B, 64)
        
        phi_scene = self.compute_scene_descriptors(Ir, It)           # (B, 2)
        
        s_t = torch.cat([gap_feat, gmp_feat, cem_pooled, phi_scene], dim=-1) # (B, 322)
        return self.fc(s_t)


class FullActorCritic(nn.Module):
    def __init__(self, state_dim=322, hidden_dim=128, num_operators=4, latent_w_dim=12):
        super().__init__()
        self.encoder = StateEncoder(state_dim=state_dim, hidden_dim=hidden_dim)
        
        # --- GROUP A HEADS ---
        self.groupA_alpha = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Softplus())
        self.groupA_beta = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Softplus())

        # --- GROUP B HEADS ---
        self.groupB_zw_mu = nn.Linear(hidden_dim, latent_w_dim)
        self.groupB_zw_logstd = nn.Linear(hidden_dim, latent_w_dim)
        self.groupB_op_logits = nn.Linear(hidden_dim, num_operators)
        self.groupB_gint_alpha = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
        self.groupB_gint_beta = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())

        # --- GROUP C HEADS ---
        self.groupC_rlvl_alpha = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
        self.groupC_rlvl_beta = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
        self.groupC_dpres_mu = nn.Linear(hidden_dim, 1)
        self.groupC_dpres_logstd = nn.Linear(hidden_dim, 1)

        # --- CRITIC HEAD ---
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, Fu, Sc, Ir, It):
        h_t = self.encoder(Fu, Sc, Ir, It)
        value = self.critic(h_t)
        return h_t, value

    def get_action(self, Fu, Sc, Ir, It, deterministic=False, tau=1.0):
        h_t, value = self.forward(Fu, Sc, Ir, It)
        
        # Group A
        a_A = self.groupA_alpha(h_t) + 1.0
        b_A = self.groupA_beta(h_t) + 1.0
        dist_A = Beta(a_A, b_A)
        c_A = a_A / (a_A + b_A) if deterministic else dist_A.sample()
        
        # Group B1
        mu_zw = self.groupB_zw_mu(h_t)
        std_zw = torch.exp(torch.clamp(self.groupB_zw_logstd(h_t), -20, 2))
        dist_B1 = Normal(mu_zw, std_zw)
        z_w = mu_zw if deterministic else dist_B1.sample()
        
        # Group B2
        op_logits = self.groupB_op_logits(h_t)
        if deterministic:
            alpha_op = F.softmax(op_logits, dim=-1)
        else:
            alpha_op = F.gumbel_softmax(op_logits, tau=tau, hard=False)
        dist_B2 = Categorical(logits=op_logits)
        
        # Group B3
        a_g = self.groupB_gint_alpha(h_t) + 1.0
        b_g = self.groupB_gint_beta(h_t) + 1.0
        dist_B3 = Beta(a_g, b_g)
        g_int = a_g / (a_g + b_g) if deterministic else dist_B3.sample()
        
        # Group C1
        a_r = self.groupC_rlvl_alpha(h_t) + 1.0
        b_r = self.groupC_rlvl_beta(h_t) + 1.0
        dist_C1 = Beta(a_r, b_r)
        r_lvl = a_r / (a_r + b_r) if deterministic else dist_C1.sample()
        
        # Group C2
        mu_dp = self.groupC_dpres_mu(h_t)
        std_dp = torch.exp(torch.clamp(self.groupC_dpres_logstd(h_t), -20, 2))
        dist_C2 = Normal(mu_dp, std_dp)
        d_pres_raw = mu_dp if deterministic else dist_C2.sample()
        d_pres = torch.tanh(d_pres_raw)

        log_prob = (
            dist_A.log_prob(c_A).sum(dim=-1, keepdim=True) +
            dist_B1.log_prob(z_w).sum(dim=-1, keepdim=True) +
            dist_B2.log_prob(torch.argmax(alpha_op, dim=-1)).unsqueeze(-1) +
            dist_B3.log_prob(g_int) +
            dist_C1.log_prob(r_lvl) +
            dist_C2.log_prob(d_pres_raw).sum(dim=-1, keepdim=True)
        )
        
        entropy = (
            dist_A.entropy().sum(dim=-1, keepdim=True) +
            dist_B1.entropy().sum(dim=-1, keepdim=True) +
            dist_B2.entropy().unsqueeze(-1) +
            dist_B3.entropy() +
            dist_C1.entropy() +
            dist_C2.entropy().sum(dim=-1, keepdim=True)
        )

        action_dict = {
            'c_rgb': c_A[:, 0:1],
            'c_th': c_A[:, 1:2],
            'c_comp': c_A[:, 2:3],
            'z_w': z_w,
            'alpha_op': alpha_op,
            'g_int': g_int,
            'r_lvl': r_lvl,
            'd_pres': d_pres,
            'd_pres_raw': d_pres_raw
        }

        return action_dict, log_prob, entropy, value

    def evaluate_actions(self, Fu, Sc, Ir, It, action_dict):
        h_t, value = self.forward(Fu, Sc, Ir, It)
        
        # Re-eval Distributions
        a_A = self.groupA_alpha(h_t) + 1.0
        b_A = self.groupA_beta(h_t) + 1.0
        dist_A = Beta(a_A, b_A)
        c_A = torch.cat([action_dict['c_rgb'], action_dict['c_th'], action_dict['c_comp']], dim=-1)

        mu_zw = self.groupB_zw_mu(h_t)
        std_zw = torch.exp(torch.clamp(self.groupB_zw_logstd(h_t), -20, 2))
        dist_B1 = Normal(mu_zw, std_zw)

        op_logits = self.groupB_op_logits(h_t)
        dist_B2 = Categorical(logits=op_logits)

        a_g = self.groupB_gint_alpha(h_t) + 1.0
        b_g = self.groupB_gint_beta(h_t) + 1.0
        dist_B3 = Beta(a_g, b_g)

        a_r = self.groupC_rlvl_alpha(h_t) + 1.0
        b_r = self.groupC_rlvl_beta(h_t) + 1.0
        dist_C1 = Beta(a_r, b_r)

        mu_dp = self.groupC_dpres_mu(h_t)
        std_dp = torch.exp(torch.clamp(self.groupC_dpres_logstd(h_t), -20, 2))
        dist_C2 = Normal(mu_dp, std_dp)

        log_prob = (
            dist_A.log_prob(c_A).sum(dim=-1, keepdim=True) +
            dist_B1.log_prob(action_dict['z_w']).sum(dim=-1, keepdim=True) +
            dist_B2.log_prob(torch.argmax(action_dict['alpha_op'], dim=-1)).unsqueeze(-1) +
            dist_B3.log_prob(action_dict['g_int']) +
            dist_C1.log_prob(action_dict['r_lvl']) +
            dist_C2.log_prob(action_dict['d_pres_raw']).sum(dim=-1, keepdim=True)
        )
        
        entropy = (
            dist_A.entropy().sum(dim=-1, keepdim=True) +
            dist_B1.entropy().sum(dim=-1, keepdim=True) +
            dist_B2.entropy().unsqueeze(-1) +
            dist_B3.entropy() +
            dist_C1.entropy() +
            dist_C2.entropy().sum(dim=-1, keepdim=True)
        )

        return log_prob, entropy, value