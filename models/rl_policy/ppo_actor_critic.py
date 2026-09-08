import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

class FullActorCritic(nn.Module):
    def __init__(self, state_dim=322, hidden_dim=128, num_ops=4):
        super().__init__()
        
        # Shared / Actor feature representation
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 1. Discrete Head: Fusion Operator Alpha (4 primitives)
        self.actor_op = nn.Linear(hidden_dim, num_ops)
        
        # 2. Continuous Head: r_lvl (Reconstruction Level) & d_pres (Detail Preservation)
        self.actor_cont_mean = nn.Linear(hidden_dim, 2)
        # Trainable log_std initialized higher to encourage exploration
        self.actor_cont_log_std = nn.Parameter(torch.ones(1, 2) * -0.5)  # std ~ 0.60
        
        # 3. Critic Value Head
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
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
        # Small gain on policy heads prevents extreme early logits
        nn.init.orthogonal_(self.actor_op.weight, gain=0.01)
        nn.init.orthogonal_(self.actor_cont_mean.weight, gain=0.01)

    def extract_state(self, Fu, Sc, Ir, It):
        # Flatten and concatenate multi-modal features and global descriptors
        b = Fu.size(0)
        fu_flat = F.adaptive_avg_pool2d(Fu, (1, 1)).view(b, -1)     # (B, 128)
        sc_flat = F.adaptive_avg_pool2d(Sc, (1, 1)).view(b, -1)     # (B, 128)
        
        # Global illuminance, thermal contrast, and gradient stats
        vis_gray = 0.2989 * Ir[:, 0:1] + 0.5870 * Ir[:, 1:2] + 0.1140 * Ir[:, 2:3]
        stats = torch.cat([
            vis_gray.mean(dim=[1, 2, 3], keepdim=True).view(b, 1),
            vis_gray.std(dim=[1, 2, 3], keepdim=True).view(b, 1),
            It.mean(dim=[1, 2, 3], keepdim=True).view(b, 1),
            It.std(dim=[1, 2, 3], keepdim=True).view(b, 1),
        ], dim=-1) # (B, 4)
        
        # Additional channel-wise statistics to match state_dim=322
        fu_max = F.adaptive_max_pool2d(Fu, (1, 1)).view(b, -1)
        state = torch.cat([fu_flat, sc_flat, fu_max[:, :62], stats], dim=-1)
        return state

    def get_action(self, Fu, Sc, Ir, It, deterministic=False):
        state = self.extract_state(Fu, Sc, Ir, It)
        features = self.shared_net(state)
        
        # Discrete Operator Distribution
        op_logits = self.actor_op(features)
        op_dist = Categorical(logits=op_logits)
        
        # Continuous (r_lvl, d_pres) Distribution
        cont_mean = torch.tanh(self.actor_cont_mean(features))
        cont_std = torch.exp(self.actor_cont_log_std).expand_as(cont_mean)
        cont_dist = Normal(cont_mean, cont_std)
        
        if deterministic:
            op_idx = torch.argmax(op_logits, dim=-1)
            cont_action = cont_mean
        else:
            op_idx = op_dist.sample()
            cont_action = cont_dist.sample()
            
        cont_action = torch.clamp(cont_action, -0.99, 0.99)
        
        # Map actions to dictionary
        alpha_op = F.one_hot(op_idx, num_classes=4).float()
        r_lvl = (cont_action[:, 0:1] + 1.0) / 2.0  # scale to [0, 1]
        d_pres = cont_action[:, 1:2]               # scale in [-1, 1]
        
        log_prob_op = op_dist.log_prob(op_idx)
        log_prob_cont = cont_dist.log_prob(cont_action).sum(dim=-1)
        total_log_prob = log_prob_op + log_prob_cont
        
        actions = {
            'alpha_op': alpha_op,
            'op_idx': op_idx,
            'r_lvl': r_lvl,
            'd_pres': d_pres,
            'cont_action': cont_action
        }
        return actions, total_log_prob, state, self.critic(state)

    def evaluate_actions(self, states, op_indices, cont_actions):
        features = self.shared_net(states)
        
        op_logits = self.actor_op(features)
        op_dist = Categorical(logits=op_logits)
        
        cont_mean = torch.tanh(self.actor_cont_mean(features))
        cont_std = torch.exp(self.actor_cont_log_std).expand_as(cont_mean)
        cont_dist = Normal(cont_mean, cont_std)
        
        log_prob_op = op_dist.log_prob(op_indices)
        log_prob_cont = cont_dist.log_prob(cont_actions).sum(dim=-1)
        total_log_prob = log_prob_op + log_prob_cont
        
        # Explicit Entropies
        entropy_op = op_dist.entropy()
        entropy_cont = cont_dist.entropy().sum(dim=-1)
        total_entropy = entropy_op + entropy_cont
        
        values = self.critic(states)
        return total_log_prob, values, total_entropy