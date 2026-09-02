import os

class Stage3Config:
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Paths
        self.data_dir = os.path.join(self.project_root, 'LLVIP')
        self.checkpoint_dir = os.path.join(self.project_root, 'checkpoints')
        
        # Resolve Stage 1 Checkpoint (Local first, fallback to Drive)
        local_s1 = os.path.join(self.checkpoint_dir, 'stage1_best.pth')
        drive_s1 = '/content/drive/MyDrive/DPAFusion_Checkpoints/Stage1/stage1_best.pth'
        self.stage1_ckpt_path = local_s1 if os.path.exists(local_s1) else drive_s1

        # Resolve Stage 2 Checkpoint (Local first, fallback to Drive)
        local_s2 = os.path.join(self.checkpoint_dir, 'stage2_best.pth')
        drive_s2 = '/content/drive/MyDrive/DPAFusion_Checkpoints/Stage2/stage2_best.pth'
        self.stage2_ckpt_path = local_s2 if os.path.exists(local_s2) else drive_s2

        # Model Architecture Flags
        self.use_cem_multiplication = False
        self.in_channels = 128
        self.num_operators = 4
        self.img_size = (256, 256)
        self.batch_size = 8
        self.num_workers = 2