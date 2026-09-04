import os

class Stage4Config:
    def __init__(self):
        # Base project path
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Paths
        self.data_dir = os.path.join(self.project_root, 'LLVIP')
        self.checkpoint_dir = os.path.join(self.project_root, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Frozen Stage 1 Weights (Local first, fallback to Google Drive)
        local_s1 = os.path.join(self.checkpoint_dir, 'stage1_best.pth')
        drive_s1 = '/content/drive/MyDrive/DPAFusion_Checkpoints/Stage1/stage1_best.pth'
        self.stage1_ckpt_path = local_s1 if os.path.exists(local_s1) else drive_s1

        # Frozen Stage 2 Weights (Local first, fallback to Google Drive)
        local_s2 = os.path.join(self.checkpoint_dir, 'stage2_best.pth')
        drive_s2 = '/content/drive/MyDrive/DPAFusion_Checkpoints/Stage2/stage2_best.pth'
        self.stage2_ckpt_path = local_s2 if os.path.exists(local_s2) else drive_s2

        # Architecture Parameters for DCLAR-Net
        self.use_cem_multiplication = False
        self.in_channels = 128
        self.cond_dim = 2             # [r_lvl, d_pres]
        self.out_channels = 1          # Grayscale luminance output I_fused

        # Training Hyperparameters
        self.img_size = (256, 256)
        self.batch_size = 8
        self.epochs = 20
        self.warmup_epochs = 3         # Phase 1: generator content warmup without discriminators
        self.lr_g = 1e-4              # Generator learning rate
        self.lr_d = 2e-5              # TTUR: Lower rate for lightweight PatchGAN discriminators
        self.num_workers = 2

        # DCLAR Consolidated Loss Weights
        self.lambda_adv = 0.01         # Hinge adversarial loss
        self.lambda_th = 1.0           # Thermal saliency preservation
        self.lambda_vis = 1.0          # Visible detail preservation (gradient)
        self.lambda_str = 1.0          # Structural preservation (SSIM with decision reference)
        self.lambda_red = 0.2          # Redundancy suppression via Stage 1 S_c