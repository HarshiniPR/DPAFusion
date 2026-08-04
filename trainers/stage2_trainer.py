import os
import torch
from torch.utils.data import DataLoader

from datasets.dataset import LLVIPDataset
from models.feature_representation.stage1_net import Stage1LHMRM
from models.rl_policy.ppo_actor_critic import GroupAActorCritic
from losses.stage2_reward import DecisionAwareReward

def run_stage2_training(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print(f"=== Training Stage 2 (PPO Decision-Aware Policy) on {device} ===")

    # 1. Load Data
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
    for param in stage1_model.parameters():
        param.requires_grad = False  # Freeze Stage 1

    # 3. Initialize Policy Agent & Reward Engine
    agent = GroupAActorCritic(state_dim=256, num_actions=3).to(device)
    reward_fn = DecisionAwareReward().to(device)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=config.lr_actor, weight_decay=1e-4)

    best_reward = -float('inf')

    for epoch in range(1, config.epochs + 1):
        agent.train()
        running_reward = 0.0
        running_c_rgb, running_c_th, running_c_comp = 0.0, 0.0, 0.0

        for step, batch in enumerate(train_loader):
            Ir = batch['rgb'].to(device)
            It = batch['ir'].to(device)

            # Step A: Extract Stage 1 Features (No Gradients)
            with torch.no_grad():
                Fu, Fr, Ft, Sc = stage1_model(Ir, It)

            # Step B: Get Action (Group A Confidences)
            actions, log_probs_old, _, values = agent.get_action(Fu, deterministic=False)
            
            c_rgb = actions[:, 0:1].unsqueeze(-1).unsqueeze(-1)
            c_th = actions[:, 1:2].unsqueeze(-1).unsqueeze(-1)
            c_comp = actions[:, 2:3].unsqueeze(-1).unsqueeze(-1)

            # Step C: Compute Weighted Fused Features
            Sc_broadcast = Sc.unsqueeze(-1).unsqueeze(-1)
            F_fused = (c_rgb * Fr) + (c_th * Ft) + (c_comp * (Sc_broadcast * Fu))

            # Step D: Compute Reward & Advantage
            rewards = reward_fn(F_fused, Fr, Ft)
            advantages = (rewards - values.detach())

            # Step E: PPO Update Steps
            for _ in range(config.ppo_epochs):
                log_probs_new, entropy, new_values = agent.evaluate_actions(Fu, actions)
                
                ratio = torch.exp(log_probs_new - log_probs_old.detach())
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * (rewards - new_values).pow(2).mean()
                entropy_loss = -0.01 * entropy.mean()

                total_ppo_loss = actor_loss + critic_loss + entropy_loss

                optimizer.zero_grad()
                total_ppo_loss.backward()
                optimizer.step()

            running_reward += rewards.mean().item()
            running_c_rgb += actions[:, 0].mean().item()
            running_c_th += actions[:, 1].mean().item()
            running_c_comp += actions[:, 2].mean().item()

            if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
                print(
                    f"Epoch [{epoch}/{config.epochs}] Step [{step+1}/{len(train_loader)}] | "
                    f"Reward: {rewards.mean().item():.4f} | "
                    f"c_rgb: {actions[:, 0].mean().item():.3f} | "
                    f"c_th: {actions[:, 1].mean().item():.3f} | "
                    f"c_comp: {actions[:, 2].mean().item():.3f}"
                )

        num_batches = len(train_loader)
        avg_reward = running_reward / num_batches

        # Save Checkpoints
        ckpt_path = os.path.join(config.checkpoint_dir, f'stage2_epoch_{epoch}.pth')
        checkpoint_data = {
            'epoch': epoch,
            'agent_state_dict': agent.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'reward': avg_reward
        }
        torch.save(checkpoint_data, ckpt_path)

        if avg_reward > best_reward:
            best_reward = avg_reward
            best_ckpt_path = os.path.join(config.checkpoint_dir, 'stage2_best.pth')
            torch.save(checkpoint_data, best_ckpt_path)
            print(f"--> [NEW BEST] Saved best RL policy to {best_ckpt_path} (Reward: {best_reward:.4f})")

        print("\n" + "="*65)
        print(f"STAGE 2 - EPOCH {epoch} SUMMARY:")
        print(f" -> Average Reward       : {avg_reward:.4f}")
        print(f" -> Avg c_rgb Confidence  : {running_c_rgb / num_batches:.4f}")
        print(f" -> Avg c_th Confidence   : {running_c_th / num_batches:.4f}")
        print(f" -> Avg c_comp Confidence : {running_c_comp / num_batches:.4f}")
        print(f" -> Checkpoint Saved      : {ckpt_path}")
        print("="*65 + "\n")