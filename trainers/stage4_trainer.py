import os
import torch
from torch.utils.data import DataLoader

from datasets.dataset import LLVIPDataset
from models.feature_representation.stage1_net import Stage1LHMRM
from models.rl_policy.ppo_actor_critic import FullActorCritic
from models.adaptive_fusion.stage3_fusion import AdaptiveSpatialFusionStage3
from models.generator.film_generator import FiLMGeneratorStage4
from models.generator.discriminator import DualDiscriminator
from losses.gan_loss import Stage4LossSuite

def run_stage4_training(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print(f"=== Training Stage 4 (FiLM Generator & Dual Discriminator) on {device} ===")

    # 1. Dataloader
    train_dataset = LLVIPDataset(root_dir=config.data_dir, split='train', img_size=config.img_size)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    # 2. Load Frozen Stage 1
    stage1 = Stage1LHMRM(use_multiplication_in_cem=config.use_cem_multiplication).to(device)
    if not os.path.exists(config.stage1_ckpt_path):
        raise FileNotFoundError(f"Stage 1 checkpoint not found at: {config.stage1_ckpt_path}")
    ckpt1 = torch.load(config.stage1_ckpt_path, map_location=device)
    stage1.load_state_dict(ckpt1['model_state_dict'])
    stage1.eval()
    for p in stage1.parameters():
        p.requires_grad = False
    print(f"-> Successfully loaded frozen Stage 1 weights from: {config.stage1_ckpt_path}")

    # 3. Load Frozen Stage 2 Policy
    stage2 = FullActorCritic(state_dim=322, hidden_dim=128).to(device)
    stage2_ckpt_path = getattr(config, 'stage2_ckpt_path', None)
    if stage2_ckpt_path is None or not os.path.exists(stage2_ckpt_path):
        local_s2 = os.path.join(config.checkpoint_dir, 'stage2_best.pth')
        drive_s2 = '/content/drive/MyDrive/DPAFusion_Checkpoints/Stage2/stage2_best.pth'
        stage2_ckpt_path = local_s2 if os.path.exists(local_s2) else drive_s2

    if not os.path.exists(stage2_ckpt_path):
        raise FileNotFoundError(f"Stage 2 checkpoint not found at: {stage2_ckpt_path}")
    ckpt2 = torch.load(stage2_ckpt_path, map_location=device)
    stage2.load_state_dict(ckpt2['agent_state_dict'])
    stage2.eval()
    for p in stage2.parameters():
        p.requires_grad = False
    print(f"-> Successfully loaded frozen Stage 2 weights from: {stage2_ckpt_path}")

    # 4. Initialize Stage 3 Execution Layer
    stage3 = AdaptiveSpatialFusionStage3(in_channels=config.in_channels, num_operators=4).to(device)
    stage3.eval()

    # 5. Initialize Stage 4 Generator & Dual Discriminator
    net_G = FiLMGeneratorStage4(in_channels=config.in_channels, cond_dim=config.cond_dim, out_channels=config.out_channels).to(device)
    net_D = DualDiscriminator().to(device)

    # 6. Optimizers (Two-Time-Scale Update Rule) & Losses
    opt_G = torch.optim.Adam(net_G.parameters(), lr=config.lr_g, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(net_D.parameters(), lr=2e-5, betas=(0.5, 0.999))
    loss_suite = Stage4LossSuite(lambda_adv=0.05, lambda_grad=20.0, lambda_pixel=5.0).to(device)

    best_g_loss = float('inf')

    for epoch in range(1, config.epochs + 1):
        net_G.train()
        net_D.train()
        running_g_loss = 0.0
        loss_D_val = 1.0  # Initialized per epoch so it is always defined

        for step, batch in enumerate(train_loader):
            Ir = batch['rgb'].to(device)
            It = batch['ir'].to(device)
            It_3c = It.repeat(1, 3, 1, 1) if It.shape[1] == 1 else It

            # Frozen Inference: Stages 1 -> 2 -> 3
            with torch.no_grad():
                Fu, Fr, Ft, Sc = stage1(Ir, It)
                actions, _, _, _ = stage2.get_action(Fu, Sc, Ir, It, deterministic=True)
                F_fused, _, _ = stage3(Fr, Ft, Fu, Sc, actions)
                I_fused_detached = net_G(F_fused, actions).detach()

            # (1) Train Dual Discriminator only when not overpowered (loss_D > 0.15)
            if loss_D_val > 0.15 or step == 0:
                opt_D.zero_grad()
                d_real_rgb = net_D.forward_rgb(Ir)
                d_fake_rgb = net_D.forward_rgb(I_fused_detached)
                loss_D_rgb = loss_suite.d_loss(d_real_rgb, d_fake_rgb)

                d_real_th = net_D.forward_th(It_3c)
                d_fake_th = net_D.forward_th(I_fused_detached)
                loss_D_th = loss_suite.d_loss(d_real_th, d_fake_th)

                loss_D = loss_D_rgb + loss_D_th
                loss_D.backward()
                opt_D.step()
                loss_D_val = loss_D.item()

            # (2) Train Generator (G)
            opt_G.zero_grad()
            I_fused = net_G(F_fused, actions)

            d_fake_rgb_for_G = net_D.forward_rgb(I_fused)
            d_fake_th_for_G = net_D.forward_th(I_fused)
            l_adv_scaled, l_adv_raw = loss_suite.g_adversarial_loss(d_fake_rgb_for_G, d_fake_th_for_G)

            l_content, l_pixel, l_grad = loss_suite.g_content_loss(I_fused, Ir, It)

            loss_G = l_adv_scaled + l_content
            loss_G.backward()
            opt_G.step()

            running_g_loss += loss_G.item()

            if (step + 1) % 20 == 0 or (step + 1) == len(train_loader):
                print(
                    f"Epoch [{epoch}/{config.epochs}] Step [{step+1}/{len(train_loader)}] | "
                    f"Loss_G: {loss_G.item():.4f} | "
                    f"L_Adv: {l_adv_raw.item():.4f} | "
                    f"L_Pix: {l_pixel.item():.4f} | "
                    f"L_Grad: {l_grad.item():.4f} | "
                    f"Loss_D: {loss_D_val:.4f}"
                )

        avg_g_loss = running_g_loss / len(train_loader)
        print(f"\n=================================================================")
        print(f"STAGE 4 - EPOCH {epoch} SUMMARY | Avg Generator Loss: {avg_g_loss:.4f}")
        print(f"=================================================================\n")

        # Save Epoch Checkpoint locally
        ckpt_path = os.path.join(config.checkpoint_dir, f'stage4_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'net_G_state_dict': net_G.state_dict(),
            'net_D_state_dict': net_D.state_dict(),
            'loss_G': avg_g_loss
        }, ckpt_path)

        # Save Best Checkpoint locally
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            best_ckpt = os.path.join(config.checkpoint_dir, 'stage4_best.pth')
            torch.save({
                'epoch': epoch,
                'net_G_state_dict': net_G.state_dict(),
                'net_D_state_dict': net_D.state_dict(),
                'loss_G': avg_g_loss
            }, best_ckpt)
            print(f"--> [NEW BEST] Saved Stage 4 checkpoint to {best_ckpt} (Loss_G: {best_g_loss:.4f})\n")