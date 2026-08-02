import os
import torch
from torch.utils.data import DataLoader

from datasets.dataset import LLVIPDataset
from models.feature_representation.stage1_net import Stage1LHMRM
from losses.stage1_loss import Stage1Loss

def run_stage1_training(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print(f"=== Training Stage 1 (LHMRM) on {device} ===")

    train_dataset = LLVIPDataset(root_dir=config.data_dir, split='train', img_size=config.img_size)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers
    )

    model = Stage1LHMRM(use_multiplication_in_cem=config.use_cem_multiplication).to(device)
    criterion = Stage1Loss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        running_metrics = {'loss_decorr': 0., 'loss_comp': 0., 'loss_structure': 0., 'loss_info': 0.}

        for step, batch in enumerate(train_loader):
            Ir = batch['rgb'].to(device)
            It = batch['ir'].to(device)

            optimizer.zero_grad()
            Fu, Fr, Ft, Sc = model(Ir, It)
            
            loss, loss_dict = criterion(Fu, Fr, Ft, Sc)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            for k, v in loss_dict.items():
                running_metrics[k] += v

            if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
                print(
                    f"Epoch [{epoch}/{config.epochs}] Step [{step+1}/{len(train_loader)}] | "
                    f"Total: {loss.item():.4f} | "
                    f"Decorr: {loss_dict['loss_decorr']:.4f} | "
                    f"CEM: {loss_dict['loss_comp']:.4f} | "
                    f"Structure: {loss_dict['loss_structure']:.4f} | "
                    f"Info: {loss_dict['loss_info']:.4f}"
                )

        ckpt_path = os.path.join(config.checkpoint_dir, f'stage1_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss / len(train_loader)
        }, ckpt_path)

        num_batches = len(train_loader)
        print("\n" + "="*65)
        print(f"EPOCH {epoch} SUMMARY & LOSS BREAKDOWN:")
        print(f" -> Average Total Loss     : {running_loss / num_batches:.4f}")
        print(f" -> Avg Loss Decorrelation : {running_metrics['loss_decorr'] / num_batches:.4f}")
        print(f" -> Avg Loss CEM Entropy   : {running_metrics['loss_comp'] / num_batches:.4f}")
        print(f" -> Avg Loss Structure     : {running_metrics['loss_structure'] / num_batches:.4f}")
        print(f" -> Avg Loss Info          : {running_metrics['loss_info'] / num_batches:.4f}")
        print(f" -> Checkpoint Saved       : {ckpt_path}")
        print("="*65 + "\n")