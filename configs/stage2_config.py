import os

class Stage2Config:
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Paths
        self.data_dir = os.path.join(self.project_root, 'LLVIP')
        
        # Direct Google Drive storage for Colab safety
        self.drive_ckpt_dir = '/content/drive/MyDrive/DPAFusion_Checkpoints/Stage2'
        self.local_ckpt_dir = os.path.join(self.project_root, 'checkpoints')
        self.checkpoint_dir = self.drive_ckpt_dir if os.path.exists('/content/drive') else self.local_ckpt_dir
        
        # Load Frozen Stage 1 Weights
        self.stage1_ckpt_path = os.path.join(self.checkpoint_dir, 'stage1_best.pth')
        if not os.path.exists(self.stage1_ckpt_path):
            self.stage1_ckpt_path = os.path.join(self.local_ckpt_dir, 'stage1_best.pth')

        # Model Flag for Stage 1 Backbone
        self.use_cem_multiplication = False   # <-- ADDED THIS LINE

        # Training Parameters
        self.img_size = (256, 256)
        self.batch_size = 8
        self.epochs = 10
        self.lr_actor = 1e-4
        self.lr_critic = 3e-4
        self.gamma = 0.99
        self.clip_eps = 0.2
        self.ppo_epochs = 4
        self.num_workers = 2