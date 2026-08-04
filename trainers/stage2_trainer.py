import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.dataset import LLVIPDataset
from models.feature_representation.stage1_net import Stage1LHMRM
from models.rl_policy.ppo_actor_critic import FullActorCritic
from losses.stage2_reward import FullMultiObjectiveReward

def upsample_zw_to_grid(z_w, height=64, width=64):
    """ Deterministic expansion of latent z_w (B, 12) to spatial grid (B, 1, 64, 64) """
    B = z_w.shape[0]
    grid_raw = z_w.view(B, 1, 3, 4)
    return F.interpolate(grid_raw, size=(height, width), mode='bilinear', align_corners=False)

def run_stage2_training(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print(f"=== Training Stage 2 (Full Hybrid PPO Policy - Groups A, B, C) on {device} ===")

    # 1. Dataset
    train_dataset = LLVIPDataset(root_dir=config.data_dir, split='train', img_size=config.img_size)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    # 2. Load Frozen Stage 1 Backbone
    stage1_model = Stage1LHMRM(use_multiplication_in_cem=config.use_cem_multiplication).to(device)
    if os.path.exists(config.stage1_ckpt_path):
        ckpt = torch.load(config.stage1_ckpt_path, map_location=device)
        stage1_model.load_state_dict(ckpt['model_state_dict'])
        print(f"-> Successfully loaded frozen Stage 1 weights from: {config.stage1_ckpt_path}")
    else:
        raise FileNotFoundError(f"Stage 1 checkpoint not found at: {config.stage1_ckpt_path}")
    
    stage1_model.eval()
    for p in stage1_model.parameters():
        p.requires_grad = False

    # 3. Policy Agent & Reward Engine
    agent = FullActorCritic(state_dim=322, hidden_dim=128).to(device)
    reward_fn = FullMultiObjectiveReward().to(device)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=config.lr_actor, weight_decay=1e-4)

    best_reward = -float('inf')

    for epoch in range(1, config.epochs + 1):
        agent.train()
        running_reward = 0.0

        for step, batch in enumerate(train_loader):
            Ir = batch['rgb'].to(device)
            It = batch['ir'].to(device)

            with torch.no_grad():
                Fu, Fr, Ft, Sc = stage1_model(Ir, It)

            # Sample Action (Group A, B, C)
            actions, log_prob_old, _, values = agent.get_action(Fu, Sc, Ir, It, deterministic=False)

            c_rgb = actions['c_rgb'].unsqueeze(-1).unsqueeze(-1)
            c_th = actions['c_th'].unsqueeze(-1).unsqueeze(-1)
            c_comp = actions['c_comp'].unsqueeze(-1).unsqueeze(-1)
            
            # Compute Spatial Region Weights (Group B)
            bias = torch.logit(c_rgb.clamp(1e-4, 1-1e-4)) - torch.logit(c_th.clamp(1e-4, 1-1e-4))
            W_rgb = torch.sigmoid(bias + upsample_zw_to_grid(actions['z_w']))
            W_th = 1.0 - W_rgb

            Sc_broadcast = Sc.unsqueeze(-1).unsqueeze(-1)
            F_fused = (W_rgb * Fr) + (W_th * Ft) + (c_comp * (Sc_broadcast * Fu))

            # Compute Reward & Advantage (Detached)
            rewards = reward_fn(F_fused, W_rgb, Sc, actions['alpha_op']).detach()
            advantages = (rewards - values.detach())

            # PPO Epoch Updates
            for _ in range(config.ppo_epochs):
                log_prob_new, entropy, new_values = agent.evaluate_actions(Fu, Sc, Ir, It, actions)
                
                ratio = torch.exp(log_prob_new - log_prob_old.detach())
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * (rewards - new_values).pow(2).mean()
                entropy_loss = -0.01 * entropy.mean()

                total_loss = actor_loss + critic_loss + entropy_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            running_reward += rewards.mean().item()

            if (step + 1) % 20 == 0 or (step + 1) == len(train_loader):
                top_op = torch.argmax(actions['alpha_op'], dim=-1).mode().item()
                print(
                    f"Epoch [{epoch}/{config.epochs}] Step [{step+1}/{len(train_loader)}] | "
                    f"Reward: {rewards.mean().item():.4f} | "
                    f"c_rgb: {actions['c_rgb'].mean().item():.2f} | "
                    f"c_th: {actions['c_th'].mean().item():.2f} | "
                    f"c_comp: {actions['c_comp'].mean().item():.2f} | "
                    f"g_int: {actions['g_int'].mean().item():.2f} | "
                    f"r_lvl: {actions['r_lvl'].mean().item():.2f} | "
                    f"d_pres: {actions['d_pres'].mean().item():.2f} | "
                    f"Op: {top_op}"
                )

        avg_reward = running_reward / len(train_loader)
        print(f"\n=================================================================")
        print(f"FULL STAGE 2 - EPOCH {epoch} SUMMARY | Average Reward: {avg_reward:.4f}")
        print(f"=================================================================\n")

        # Save Epoch Checkpoint
        ckpt_path = os.path.join(config.checkpoint_dir, f'stage2_epoch_{epoch}.pth')
        checkpoint_data = {
            'epoch': epoch,
            'agent_state_dict': agent.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'reward': avg_reward
        }
        torch.save(checkpoint_data, ckpt_path)

        # Save Best Checkpoint
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_ckpt_path = os.path.join(config.checkpoint_dir, 'stage2_best.pth')
            torch.save(checkpoint_data, best_ckpt_path)
            print(f"--> [NEW BEST] Saved best Stage 2 RL policy to {best_ckpt_path} (Reward: {best_reward:.4f})\n")