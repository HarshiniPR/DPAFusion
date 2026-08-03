import os

class Stage1Config:
    def __init__(self):
        # Base project path
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Paths
        self.data_dir = os.path.join(self.project_root, 'LLVIP')
        self.checkpoint_dir = os.path.join(self.project_root, 'checkpoints')
        
        # Training Parameters (Preserved)
        self.img_size = (256, 256)
        self.batch_size = 8
        self.epochs = 10
        self.lr = 5e-5
        self.weight_decay = 1e-4
        self.num_workers = 2
        
        # CEM Option
        self.use_cem_multiplication = False