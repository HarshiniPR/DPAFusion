import torch
import torch.nn as nn
from torch.distributions import Beta

class StateEncoder(nn.Module):
    """
    Encodes Stage 1 feature map Fu (B, 128, H, W) into a compact State Representation Vector (B, 256).
    """
    def __init__(self, in_channels=128, state_dim=256):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, state_dim),
            nn.LayerNorm(state_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, Fu):
        avg_feat = self.gap(Fu).squeeze(-1).squeeze(-1)
        max_feat = self.gmp(Fu).squeeze(-1).squeeze(-1)
        combined = torch.cat([avg_feat, max_feat], dim=-1)
        return self.fc(combined)


class GroupAActorCritic(nn.Module):
    """
    Actor-Critic network outputting continuous Beta parameters (alpha, beta > 1.0) 
    for Group A actions: c_rgb, c_th, c_comp in (0, 1).
    """
    def __init__(self, state_dim=256, num_actions=3):
        super().__init__()
        self.state_encoder = StateEncoder(in_channels=128, state_dim=state_dim)
        
        # Shared Feature Backbone
        self.shared_mlp = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(inplace=True)
        )
        
        # Actor Head (Outputs Alpha and Beta for Beta Distribution)
        self.alpha_head = nn.Sequential(
            nn.Linear(128, num_actions),
            nn.Softplus()
        )
        self.beta_head = nn.Sequential(
            nn.Linear(128, num_actions),
            nn.Softplus()
        )
        
        # Critic Head (Outputs State Value Estimate V(s))
        self.critic_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, Fu):
        state_vec = self.state_encoder(Fu)
        shared_feat = self.shared_mlp(state_vec)
        
        # Add 1.0 to ensure Alpha and Beta parameters stay > 1 for unimodal Beta distribution
        alpha = self.alpha_head(shared_feat) + 1.0
        beta = self.beta_head(shared_feat) + 1.0
        value = self.critic_head(shared_feat)
        
        return alpha, beta, value

    def get_action(self, Fu, deterministic=False):
        alpha, beta, value = self.forward(Fu)
        dist = Beta(alpha, beta)
        
        if deterministic:
            action = alpha / (alpha + beta) # Mode/Mean of Beta Distribution
        else:
            action = dist.sample()
            
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return action, log_prob, entropy, value

    def evaluate_actions(self, Fu, actions):
        alpha, beta, value = self.forward(Fu)
        dist = Beta(alpha, beta)
        
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy, value