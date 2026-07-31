import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class LLVIPDataset(Dataset):
    def __init__(self, root_dir=None, split='train', img_size=(256, 256)):
        super().__init__()
        
        if root_dir is None:
            # Resolves path to DAPFUSION/LLVIP/ relative to this file's location
            base_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            root_dir = os.path.join(base_project_dir, 'LLVIP')
            
        self.split = split
        self.ir_dir = os.path.join(root_dir, 'infrared', split)
        self.vis_dir = os.path.join(root_dir, 'visible', split)
        
        if not os.path.exists(self.vis_dir) or not os.path.exists(self.ir_dir):
            raise FileNotFoundError(
                f"LLVIP directories not found.\n"
                f"Expected RGB at: {self.vis_dir}\n"
                f"Expected IR at:  {self.ir_dir}\n"
                f"Please verify your folder structure."
            )

        self.filenames = sorted([
            f for f in os.listdir(self.vis_dir) 
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        self.vis_transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.ir_transform = T.Compose([
            T.Resize(img_size),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        vis_path = os.path.join(self.vis_dir, filename)
        ir_path = os.path.join(self.ir_dir, filename)
        
        vis_img = Image.open(vis_path).convert('RGB')
        ir_img = Image.open(ir_path).convert('L')
        
        return {
            'rgb': self.vis_transform(vis_img),
            'ir': self.ir_transform(ir_img),
            'filename': filename
        }