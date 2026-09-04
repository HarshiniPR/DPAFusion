import os
import torch
from torch.utils.data import DataLoader

from datasets.dataset import LLVIPDataset
from models.feature_representation.stage1_net import Stage1LHMRM
from models.rl_policy.ppo_actor_critic import FullActorCritic
from models.adaptive_fusion.stage3_fusion import AdaptiveSpatialFusionStage3
from models.generator.dclar_net import DCLARNetGenerator, DCLARDualDiscriminator
from losses.dclar_loss import DCLARLossSuite

def run_stage4_training(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print(f"=== Training Stage IV: DCLAR-Net on {device} ===")

    # 1. Dataloader
    train_dataset = LLVIPDataset(root_dir=config.data_dir, split='train', img_size=config.img_size)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    # 2. Frozen Stage 1 Backbone
    stage1 = Stage1LHMRM(use_multiplication_in_cem=config.use_cem_multiplication).to(device)
    ckpt1 = torch.load(config.stage1_ckpt_path, map_location=device)
    stage1.load_state_dict(ckpt1['model_state_dict'])
    stage1.eval()
    for p in stage1.parameters(): p.requires_grad = False
    print(f"-> Frozen Stage 1 loaded: {config.stage1_ckpt_path}")

    # 3. Frozen Stage 2 PPO Policy
    stage2 = FullActorCritic(state_dim=322, hidden_dim=128).to(device)
    ckpt2 = torch.load(config.stage2_ckpt_path, map_location=device)
    stage2.load_state_dict(ckpt2['agent_state_dict'])
    stage2.eval()
    for p in stage2.parameters(): p.requires_grad = False
    print(f"-> Frozen Stage 2 loaded: {config.stage2_ckpt_path}")

    # 4. Frozen Stage 3 Deterministic Execution Layer
    stage3 = AdaptiveSpatialFusionStage3(in_channels=config.in_channels, num_operators=4).to(device)
    stage3.eval()

    # 5. Stage IV Generator & Dual Discriminators
    net_G = DCLARNetGenerator(in_channels=config.in_channels, cond_dim=config.cond_dim).to(device)
    net_D = DCLARDualDiscriminator().to(device)

    # 6. Optimizers & Losses
    opt_G = torch.optim.Adam(net_G.parameters(), lr=config.lr_g, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(net_D.parameters(), lr=config.lr_d, betas=(0.5, 0.999))
    loss_suite = DCLARLossSuite().to(device)

    best_loss = float('inf')

    for epoch in range(1, config.epochs + 1):
        net_G.train()
        net_D.train()
        is_warmup = (epoch <= 3) # Phase 1: Warmup for Epochs 1-3
        
        running_loss = 0.0
        print(f"--- Epoch [{epoch}/{config.epochs}] {'[PHASE I: CONTENT WARMUP]' if is_warmup else '[PHASE II: JOINT HINGE GAN]'} ---")

        for step, batch in enumerate(train_loader):
            Ir = batch['rgb'].to(device)
            It = batch['ir'].to(device)
            
            # Grayscale luminance for visible edge calculations
            I_vis_gray = 0.2989 * Ir[:, 0:1] + 0.5870 * Ir[:, 1:2] + 0.1140 * Ir[:, 2:3]

            # Frozen Inference: Stages I -> II -> III
            with torch.no_grad():
                Fu, Fr, Ft, Sc = stage1(Ir, It)
                actions, _, _, _ = stage2.get_action(Fu, Sc, Ir, It, deterministic=True)
                F_fused, W_rgb, W_th = stage3(Fr, Ft, Fu, Sc, actions)

            # -----------------------------------------------
            # (1) Update Dual Discriminators (Phase II Only)
            # -----------------------------------------------
            loss_D_val = 0.0
            if not is_warmup:
                opt_D.zero_grad()
                with torch.no_grad():
                    I_fused_det = net_G(F_fused, actions).detach()

                d_real_th = net_D.forward_th(It)
                d_fake_th = net_D.forward_th(I_fused_det)
                loss_D_th = loss_suite.discriminator_hinge_loss(d_real_th, d_fake_th)

                d_real_vis = net_D.forward_vis(I_vis_gray)
                d_fake_vis = net_D.forward_vis(I_fused_det)
                loss_D_vis = loss_suite.discriminator_hinge_loss(d_real_vis, d_fake_vis)

                loss_D = loss_D_th + loss_D_vis
                loss_D.backward()
                opt_D.step()
                loss_D_val = loss_D.item()

            # -----------------------------------------------
            # (2) Update DCLAR-Net Generator
            # -----------------------------------------------
            opt_G.zero_grad()
            I_fused = net_G(F_fused, actions)

            if not is_warmup:
                d_fake_th_G = net_D.forward_th(I_fused)
                d_fake_vis_G = net_D.forward_vis(I_fused)
                loss_G, loss_dict = loss_suite.compute_generator_losses(
                    I_fused, I_vis_gray, It, W_th, Sc, d_fake_th_G, d_fake_vis_G, warmup=False
                )
            else:
                loss_G, loss_dict = loss_suite.compute_generator_losses(
                    I_fused, I_vis_gray, It, W_th, Sc, warmup=True
                )

            loss_G.backward()
            opt_G.step()

            running_loss += loss_G.item()

            if (step + 1) % 20 == 0 or (step + 1) == len(train_loader):
                print(
                    f"Step [{step+1}/{len(train_loader)}] | "
                    f"Loss_G: {loss_G.item():.4f} | "
                    f"Th_Sal: {loss_dict['L_th_sal']:.4f} | "
                    f"Grad: {loss_dict['L_grad']:.4f} | "
                    f"SSIM: {loss_dict['L_ssim']:.4f} | "
                    f"Red: {loss_dict['L_red']:.4f} | "
                    f"Adv: {loss_dict['L_adv']:.4f} | "
                    f"Loss_D: {loss_D_val:.4f}"
                )

        avg_loss = running_loss / len(train_loader)
        print(f"\nEPOCH {epoch} SUMMARY | Avg Generator Loss: {avg_loss:.4f}\n")

        # Save local epoch checkpoint
        ckpt_path = os.path.join(config.checkpoint_dir, f'stage4_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'net_G_state_dict': net_G.state_dict(),
            'net_D_state_dict': net_D.state_dict(),
            'loss_G': avg_loss
        }, ckpt_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt = os.path.join(config.checkpoint_dir, 'stage4_best.pth')
            torch.save({
                'epoch': epoch,
                'net_G_state_dict': net_G.state_dict(),
                'net_D_state_dict': net_D.state_dict(),
                'loss_G': avg_loss
            }, best_ckpt)
            print(f"--> [BEST SAVED] Checkpoint saved to {best_ckpt} (Loss: {best_loss:.4f})\n")