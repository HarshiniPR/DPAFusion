import os

class Stage4Config:
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Data & Local Checkpoints (Root first)
        self.data_dir = os.path.join(self.project_root, 'LLVIP')
        self.checkpoint_dir = os.path.join(self.project_root, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Frozen Stage 1 Weights (Local first, fallback to Drive)
        local_s1 = os.path.join(self.checkpoint_dir, 'stage1_best.pth')
        drive_s1 = '/content/drive/MyDrive/DPAFusion_Checkpoints/Stage1/stage1_best.pth'
        self.stage1_ckpt_path = local_s1 if os.path.exists(local_s1) else drive_s1

        # Frozen Stage 2 Weights (Local first, fallback to Drive)
        local_s2 = os.path.join(self.checkpoint_dir, 'stage2_best.pth')
        drive_s2 = '/content/drive/MyDrive/DPAFusion_Checkpoints/Stage2/stage2_best.pth'
        self.stage2_ckpt_path = local_s2 if os.path.exists(local_s2) else drive_s2

        # Model Flags
        self.use_cem_multiplication = False
        self.in_channels = 128
        self.cond_dim = 2
        self.out_channels = 3

        # Training Parameters
        self.img_size = (256, 256)
        self.batch_size = 8
        self.epochs = 20
        self.lr_g = 1e-4
        self.lr_d = 1e-4
        self.num_workers = 2