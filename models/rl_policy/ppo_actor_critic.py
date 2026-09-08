import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

class FullActorCritic(nn.Module):
    def __init__(self, state_dim=322, hidden_dim=128, num_ops=4):
        super().__init__()
        
        # 1. Feature Representation Trunk
        self.actor_trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 2. Discrete Head (Operator Selection: 4 Primitives)
        self.actor_op = nn.Linear(hidden_dim, num_ops)
        
        # 3. Continuous Head: 3 parameters -> [c_rgb, r_lvl, d_pres]
        self.actor_cont_mean = nn.Linear(hidden_dim, 3)
        self.actor_cont_log_std = nn.Parameter(torch.ones(1, 3) * -0.30)
        
        # 4. Critic Trunk & Value Head
        self.critic_trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor_op.weight, gain=0.01)
        nn.init.orthogonal_(self.actor_cont_mean.weight, gain=0.01)

    def extract_state(self, Fu, Sc, Ir, It):
        b = Fu.size(0)
        
        # A. Fu representation (4D: [B, C, H, W])
        fu_flat = F.adaptive_avg_pool2d(Fu, (1, 1)).view(b, -1)     # (B, 128)
        fu_max = F.adaptive_max_pool2d(Fu, (1, 1)).view(b, -1)      # (B, 128)
        
        # B. Sc representation (Robust to 2D [B, 128] or 4D [B, 128, 1, 1])
        if Sc.dim() == 4:
            sc_flat = F.adaptive_avg_pool2d(Sc, (1, 1)).view(b, -1)
        elif Sc.dim() == 2:
            sc_flat = Sc.view(b, -1)
        else:
            sc_flat = Sc.flatten(start_dim=1)
            
        # C. Low-level Environmental Context
        vis_gray = 0.2989 * Ir[:, 0:1] + 0.5870 * Ir[:, 1:2] + 0.1140 * Ir[:, 2:3]
        stats = torch.cat([
            vis_gray.mean(dim=[1, 2, 3], keepdim=True).view(b, 1),
            vis_gray.std(dim=[1, 2, 3], keepdim=True).view(b, 1),
            It.mean(dim=[1, 2, 3], keepdim=True).view(b, 1),
            It.std(dim=[1, 2, 3], keepdim=True).view(b, 1)
        ], dim=-1)  # (B, 4)
        
        # Total state: 128 + 128 + 62 + 4 = 322 dims
        state = torch.cat([fu_flat, sc_flat, fu_max[:, :62], stats], dim=-1)
        return state

    def get_action(self, Fu, Sc, Ir, It, deterministic=False):
        state = self.extract_state(Fu, Sc, Ir, It)
        features = self.actor_trunk(state)
        
        # Discrete branch
        op_logits = self.actor_op(features)
        dist_op = Categorical(logits=op_logits)
        
        # Continuous branch (dim=3)
        cont_mean = torch.tanh(self.actor_cont_mean(features))
        cont_std = torch.exp(self.actor_cont_log_std).expand_as(cont_mean)
        dist_cont = Normal(cont_mean, cont_std)
        
        if deterministic:
            op_idx = torch.argmax(op_logits, dim=-1)
            cont_action = cont_mean
        else:
            op_idx = dist_op.sample()
            cont_action = dist_cont.sample()
            
        cont_action = torch.clamp(cont_action, -0.99, 0.99)
        
        # Map actions
        alpha_op = F.one_hot(op_idx, num_classes=4).float()
        c_rgb = (cont_action[:, 0:1] + 1.0) / 2.0   # Scale [-1, 1] -> [0, 1]
        r_lvl = (cont_action[:, 1:2] + 1.0) / 2.0   # Scale [-1, 1] -> [0, 1]
        d_pres = cont_action[:, 2:3]                # Bounded in [-1, 1]
        
        log_prob_op = dist_op.log_prob(op_idx)
        log_prob_cont = dist_cont.log_prob(cont_action).sum(dim=-1)
        total_log_prob = log_prob_op + log_prob_cont
        
        actions = {
            'alpha_op': alpha_op,
            'op_idx': op_idx,
            'c_rgb': c_rgb,
            'r_lvl': r_lvl,
            'd_pres': d_pres,
            'cont_action': cont_action
        }
        values = self.critic_trunk(state)
        return actions, total_log_prob, state, values

    def evaluate_actions(self, states, op_indices, cont_actions):
        features = self.actor_trunk(states)
        
        # Discrete
        op_logits = self.actor_op(features)
        dist_op = Categorical(logits=op_logits)
        
        # Continuous
        cont_mean = torch.tanh(self.actor_cont_mean(features))
        cont_std = torch.exp(self.actor_cont_log_std).expand_as(cont_mean)
        dist_cont = Normal(cont_mean, cont_std)
        
        log_prob_op = dist_op.log_prob(op_indices)
        log_prob_cont = dist_cont.log_prob(cont_actions).sum(dim=-1)
        total_log_prob = log_prob_op + log_prob_cont
        
        entropy_op = dist_op.entropy().mean()
        entropy_cont = dist_cont.entropy().sum(dim=-1).mean()
        
        values = self.critic_trunk(states)
        return total_log_prob, values, entropy_op, entropy_cont