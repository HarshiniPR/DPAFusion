import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.dataset import LLVIPDataset
from models.feature_representation.stage1_net import Stage1LHMRM
from models.rl_policy.ppo_actor_critic import FullActorCritic
from models.adaptive_fusion.stage3_fusion import AdaptiveSpatialFusionStage3
from losses.stage2_reward import AdaptiveFusionReward

def run_stage2_training(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    print(f"=== Training Stage II: PPO Policy on {device} ===")

    # 1. Dataset & Loaders
    train_dataset = LLVIPDataset(root_dir=config.data_dir, split='train', img_size=config.img_size)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    # 2. Frozen Stage 1
    stage1 = Stage1LHMRM(use_multiplication_in_cem=config.use_cem_multiplication).to(device)
    stage1.load_state_dict(torch.load(config.stage1_ckpt_path, map_location=device)['model_state_dict'])
    stage1.eval()
    for p in stage1.parameters(): p.requires_grad = False

    # 3. Stage 2 Policy & Stage 3 Execution Layer
    actor_critic = FullActorCritic(state_dim=322, hidden_dim=128).to(device)
    stage3 = AdaptiveSpatialFusionStage3(in_channels=128, num_operators=4).to(device)
    stage3.eval()

    reward_fn = AdaptiveFusionReward().to(device)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=1e-4, eps=1e-5)

    # Tuned PPO Hyperparameters
    entropy_coef = 0.06   # High entropy to prevent mode collapse
    clip_eps = 0.15
    ppo_epochs = 4

    for epoch in range(1, config.epochs + 1):
        actor_critic.train()
        epoch_rewards = []

        for step, batch in enumerate(train_loader):
            Ir = batch['rgb'].to(device)
            It = batch['ir'].to(device)

            with torch.no_grad():
                Fu, Fr, Ft, Sc = stage1(Ir, It)

            # Rollout
            actions, old_log_prob, states, values = actor_critic.get_action(Fu, Sc, Ir, It, deterministic=False)
            
            with torch.no_grad():
                F_fused, W_rgb, W_th = stage3(Fr, Ft, Fu, Sc, actions)
                rewards = reward_fn(Ir, It, F_fused, W_th, actions)

            epoch_rewards.append(rewards.mean().item())

            # Advantage estimation (Batch-level)
            advantages = rewards.unsqueeze(1) - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO Updates
            for _ in range(ppo_epochs):
                new_log_prob, new_values, entropy = actor_critic.evaluate_actions(
                    states, actions['op_idx'], actions['cont_action']
                )

                ratio = torch.exp(new_log_prob - old_log_prob.detach())
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * F.mse_loss(new_values, rewards.unsqueeze(1))
                entropy_loss = -entropy_coef * entropy.mean()

                loss = actor_loss + critic_loss + entropy_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), max_norm=0.5)
                optimizer.step()

        avg_reward = sum(epoch_rewards) / len(epoch_rewards)
        print(f"Epoch [{epoch}/{config.epochs}] | Mean Reward: {avg_reward:.4f}")

        # Save Best Model
        ckpt_path = os.path.join(config.checkpoint_dir, 'stage2_best.pth')
        torch.save({'epoch': epoch, 'agent_state_dict': actor_critic.state_dict()}, ckpt_path)