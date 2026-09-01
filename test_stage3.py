import os
import torch
from datasets.dataset import LLVIPDataset
from models.feature_representation.stage1_net import Stage1LHMRM
from models.rl_policy.ppo_actor_critic import FullActorCritic
from models.adaptive_fusion.stage3_fusion import AdaptiveSpatialFusionStage3

def test_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Testing Stage 1 -> Stage 2 -> Stage 3 Pipeline on {device} ===")

    # 1. Paths
    stage1_path = '/content/drive/MyDrive/DPAFusion_Checkpoints/Stage1/stage1_best.pth'
    stage2_path = '/content/drive/MyDrive/DPAFusion_Checkpoints/Stage2/stage2_best.pth'

    if not os.path.exists(stage1_path):
        stage1_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'stage1_best.pth')
    if not os.path.exists(stage2_path):
        stage2_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'stage2_best.pth')

    # 2. Instantiate & Load Stage 1
    stage1 = Stage1LHMRM(use_multiplication_in_cem=False).to(device)
    stage1_ckpt = torch.load(stage1_path, map_location=device)
    stage1.load_state_dict(stage1_ckpt['model_state_dict'])
    stage1.eval()
    print("-> Stage 1 weights loaded successfully.")

    # 3. Instantiate & Load Stage 2
    stage2 = FullActorCritic(state_dim=322, hidden_dim=128).to(device)
    stage2_ckpt = torch.load(stage2_path, map_location=device)
    stage2.load_state_dict(stage2_ckpt['agent_state_dict'])
    stage2.eval()
    print("-> Stage 2 policy loaded successfully.")

    # 4. Instantiate Stage 3
    stage3 = AdaptiveSpatialFusionStage3(in_channels=128, num_operators=4).to(device)
    stage3.eval()
    print("-> Stage 3 execution engine initialized.")

    # 5. Synthetic batch test
    Ir = torch.randn(2, 3, 256, 256).to(device)
    It = torch.randn(2, 1, 256, 256).to(device)

    with torch.no_grad():
        # Pass through Stage 1
        Fu, Fr, Ft, Sc = stage1(Ir, It)
        # Pass through Stage 2 (Deterministic Mode)
        actions, _, _, _ = stage2.get_action(Fu, Sc, Ir, It, deterministic=True)
        # Pass through Stage 3
        F_fused, W_rgb, W_th = stage3(Fr, Ft, Fu, Sc, actions)

    print("\n" + "="*55)
    print("PIPELINE TEST OUTPUT VERIFICATION:")
    print(f" -> Fr Shape      : {Fr.shape} (Expected: [2, 128, 64, 64])")
    print(f" -> Ft Shape      : {Ft.shape} (Expected: [2, 128, 64, 64])")
    print(f" -> Sc Shape      : {Sc.shape} (Expected: [2, 128])")
    print(f" -> W_rgb Shape   : {W_rgb.shape} (Expected: [2, 1, 64, 64])")
    print(f" -> W_th Shape    : {W_th.shape} (Expected: [2, 1, 64, 64])")
    print(f" -> F_fused Shape : {F_fused.shape} (Expected: [2, 128, 64, 64])")
    print("="*55)
    print("Stage 1 -> Stage 2 -> Stage 3 integration is functional.")

if __name__ == '__main__':
    test_pipeline()