import os

class Stage2Config:
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Paths
        self.data_dir = os.path.join(self.project_root, 'LLVIP')
        # All stage 2 outputs save strictly to root checkpoints first
        self.checkpoint_dir = os.path.join(self.project_root, 'checkpoints')
        
        # Frozen Stage 1 Checkpoint resolution: check local first, fallback to Drive
        local_stage1 = os.path.join(self.checkpoint_dir, 'stage1_best.pth')
        drive_stage1 = '/content/drive/MyDrive/DPAFusion_Checkpoints/Stage1/stage1_best.pth'
        self.stage1_ckpt_path = local_stage1 if os.path.exists(local_stage1) else drive_stage1

        # Stage 1 Architecture Match
        self.use_cem_multiplication = False

        # PPO Training Parameters
        self.img_size = (256, 256)
        self.batch_size = 8
        self.epochs = 30
        self.lr_actor = 1e-4
        self.lr_critic = 3e-4
        self.gamma = 0.99
        self.clip_eps = 0.2
        self.ppo_epochs = 4
        self.num_workers = 2