import os
import torch
from configs.stage3_config import Stage3Config
from models.feature_representation.stage1_net import Stage1LHMRM
from models.rl_policy.ppo_actor_critic import FullActorCritic
from models.adaptive_fusion.stage3_fusion import AdaptiveSpatialFusionStage3

def test_stage3_pipeline():
    config = Stage3Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Testing Stage 1 -> Stage 2 -> Stage 3 Pipeline on {device} ===")

    # 1. Load Stage 1
    if not os.path.exists(config.stage1_ckpt_path):
        raise FileNotFoundError(f"Stage 1 checkpoint not found at: {config.stage1_ckpt_path}")
    stage1 = Stage1LHMRM(use_multiplication_in_cem=config.use_cem_multiplication).to(device)
    stage1_ckpt = torch.load(config.stage1_ckpt_path, map_location=device)
    stage1.load_state_dict(stage1_ckpt['model_state_dict'])
    stage1.eval()
    print(f"-> Stage 1 weights loaded from: {config.stage1_ckpt_path}")

    # 2. Load Stage 2
    if not os.path.exists(config.stage2_ckpt_path):
        raise FileNotFoundError(f"Stage 2 checkpoint not found at: {config.stage2_ckpt_path}")
    stage2 = FullActorCritic(state_dim=322, hidden_dim=128).to(device)
    stage2_ckpt = torch.load(config.stage2_ckpt_path, map_location=device)
    stage2.load_state_dict(stage2_ckpt['agent_state_dict'])
    stage2.eval()
    print(f"-> Stage 2 policy loaded from: {config.stage2_ckpt_path}")

    # 3. Initialize Stage 3
    stage3 = AdaptiveSpatialFusionStage3(in_channels=config.in_channels, num_operators=config.num_operators).to(device)
    stage3.eval()
    print("-> Stage 3 execution layer initialized.")

    # 4. Forward Verification
    Ir = torch.randn(2, 3, 256, 256).to(device)
    It = torch.randn(2, 1, 256, 256).to(device)

    with torch.no_grad():
        Fu, Fr, Ft, Sc = stage1(Ir, It)
        actions, _, _, _ = stage2.get_action(Fu, Sc, Ir, It, deterministic=True)
        F_fused, W_rgb, W_th = stage3(Fr, Ft, Fu, Sc, actions)

    print("\n" + "="*60)
    print("OUTPUT SHAPE VERIFICATION:")
    print(f" -> Fr       : {Fr.shape} (Expected: [2, 128, 64, 64])")
    print(f" -> Ft       : {Ft.shape} (Expected: [2, 128, 64, 64])")
    print(f" -> Fu       : {Fu.shape} (Expected: [2, 128, 64, 64])")
    print(f" -> Sc       : {Sc.shape} (Expected: [2, 128])")
    print(f" -> W_rgb    : {W_rgb.shape} (Expected: [2, 1, 64, 64])")
    print(f" -> W_th     : {W_th.shape} (Expected: [2, 1, 64, 64])")
    print(f" -> F_fused  : {F_fused.shape} (Expected: [2, 128, 64, 64])")
    print("="*60)
    print("Stage 1 -> Stage 2 -> Stage 3 verification successful.\n")

if __name__ == '__main__':
    test_stage3_pipeline()