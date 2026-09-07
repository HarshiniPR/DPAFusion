import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from datasets.dataset import LLVIPDataset
from models.feature_representation.stage1_net import Stage1LHMRM
from models.rl_policy.ppo_actor_critic import FullActorCritic
from models.adaptive_fusion.stage3_fusion import AdaptiveSpatialFusionStage3
from models.generator.dclar_net import DCLARNetGenerator

def resolve_checkpoint(local_dir, drive_dir, stage_name):
    """Find checkpoint in local checkpoints/ first, then fall back to Google Drive."""
    local_path = os.path.join(local_dir, f"{stage_name}_best.pth")
    if os.path.isfile(local_path):
        return local_path
    
    # Check Drive subfolder then flat Drive folder
    drive_sub = os.path.join(drive_dir, stage_name.capitalize(), f"{stage_name}_best.pth")
    if os.path.isfile(drive_sub):
        return drive_sub
        
    drive_flat = os.path.join(drive_dir, f"{stage_name}_best.pth")
    if os.path.isfile(drive_flat):
        return drive_flat

    raise FileNotFoundError(
        f"Could not find checkpoint for {stage_name}. Looked in:\n"
        f" - {local_path}\n"
        f" - {drive_sub}\n"
        f" - {drive_flat}"
    )

def main():
    parser = argparse.ArgumentParser(description="DPAFusion Visual Inference Pipeline")
    parser.add_argument('--samples', type=int, default=5, help="Number of test pairs to visualize")
    parser.add_argument('--output_dir', type=str, default='inference_results', help="Directory to save output images")
    parser.add_argument('--drive_ckpt_dir', type=str, default='/content/drive/MyDrive/DPAFusion_Checkpoints',
                        help="Google Drive checkpoint directory")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    local_ckpt_dir = os.path.join(project_root, 'checkpoints')
    save_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== DPAFusion Inference Pipeline on {device} ===")

    # 1. Resolve Checkpoint Paths
    s1_path = resolve_checkpoint(local_ckpt_dir, args.drive_ckpt_dir, 'stage1')
    s2_path = resolve_checkpoint(local_ckpt_dir, args.drive_ckpt_dir, 'stage2')
    s4_path = resolve_checkpoint(local_ckpt_dir, args.drive_ckpt_dir, 'stage4')

    print(f"-> S1 Checkpoint: {s1_path}")
    print(f"-> S2 Checkpoint: {s2_path}")
    print(f"-> S4 Checkpoint: {s4_path}")

    # 2. Instantiate and Load Models
    print("\nLoading models...")
    stage1 = Stage1LHMRM(use_multiplication_in_cem=False).to(device)
    stage1.load_state_dict(torch.load(s1_path, map_location=device)['model_state_dict'])
    stage1.eval()

    stage2 = FullActorCritic(state_dim=322, hidden_dim=128).to(device)
    stage2.load_state_dict(torch.load(s2_path, map_location=device)['agent_state_dict'])
    stage2.eval()

    stage3 = AdaptiveSpatialFusionStage3(in_channels=128, num_operators=4).to(device)
    stage3.eval()

    stage4 = DCLARNetGenerator(in_channels=128, cond_dim=2).to(device)
    ckpt4 = torch.load(s4_path, map_location=device)
    state_dict_4 = ckpt4['net_G_state_dict'] if 'net_G_state_dict' in ckpt4 else ckpt4
    stage4.load_state_dict(state_dict_4)
    stage4.eval()
    print("All four stages initialized and set to evaluation mode.")

    # 3. Load Dataset
    data_dir = os.path.join(project_root, 'LLVIP')
    if not os.path.exists(data_dir):
        data_dir = '/content/LLVIP'
        
    dataset = LLVIPDataset(root_dir=data_dir, split='test', img_size=(256, 256))
    if len(dataset) == 0:
        dataset = LLVIPDataset(root_dir=data_dir, split='train', img_size=(256, 256))

    indices = np.linspace(0, len(dataset) - 1, args.samples, dtype=int)

    # 4. Generate Comparisons
    fig, axes = plt.subplots(args.samples, 3, figsize=(12, 3.5 * args.samples))
    if args.samples == 1:
        axes = np.expand_dims(axes, axis=0)

    print(f"\nProcessing {args.samples} paired frames...")
    for row_idx, data_idx in enumerate(indices):
        sample = dataset[data_idx]
        Ir = sample['rgb'].unsqueeze(0).to(device)
        It = sample['ir'].unsqueeze(0).to(device)

        with torch.no_grad():
            Fu, Fr, Ft, Sc = stage1(Ir, It)
            actions, _, _, _ = stage2.get_action(Fu, Sc, Ir, It, deterministic=True)
            F_fused, W_rgb, W_th = stage3(Fr, Ft, Fu, Sc, actions)
            I_fused = stage4(F_fused, actions)

        vis_disp = np.clip(Ir.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0.0, 1.0)
        ir_disp = np.clip(It.squeeze(0).squeeze(0).cpu().numpy(), 0.0, 1.0)
        fused_disp = np.clip(I_fused.squeeze(0).squeeze(0).cpu().numpy(), 0.0, 1.0)

        # Plot Visible
        axes[row_idx, 0].imshow(vis_disp)
        axes[row_idx, 0].set_title(f"Visible (Sample {data_idx})", fontsize=11)
        axes[row_idx, 0].axis('off')

        # Plot Infrared
        axes[row_idx, 1].imshow(ir_disp, cmap='gray')
        axes[row_idx, 1].set_title("Infrared (Thermal)", fontsize=11)
        axes[row_idx, 1].axis('off')

        # Plot Fused
        axes[row_idx, 2].imshow(fused_disp, cmap='gray')
        op_selected = torch.argmax(actions['alpha_op'], dim=-1).item()
        axes[row_idx, 2].set_title(f"DPAFusion Fused (Op: {op_selected})", fontsize=11, fontweight='bold', color='darkgreen')
        axes[row_idx, 2].axis('off')

        # Save individual row composite
        composite = np.concatenate([
            (np.tile(vis_disp.mean(axis=-1, keepdims=True), (1, 1, 3)) * 255).astype(np.uint8),
            (np.tile(ir_disp[..., None], (1, 1, 3)) * 255).astype(np.uint8),
            (np.tile(fused_disp[..., None], (1, 1, 3)) * 255).astype(np.uint8)
        ], axis=1)
        Image.fromarray(composite).save(os.path.join(save_dir, f'sample_{data_idx}.png'))

    plt.tight_layout()
    overview_path = os.path.join(save_dir, 'side_by_side_comparison.png')
    plt.savefig(overview_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\nCompleted! Saved individual samples and montage to: {save_dir}")
    print(f"Montage file: {overview_path}")

if __name__ == '__main__':
    main()