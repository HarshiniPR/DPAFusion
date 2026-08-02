import os
import torch
from torch.utils.data import DataLoader
from datasets.dataset import LLVIPDataset
from models.feature_representation.stage1_net import Stage1LHMRM
from losses.stage1_loss import Stage1Loss

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Checkpoint output directory setup
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'stage1', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"=== Training Stage 1 (LHMRM) on {device} ===")
    train_dataset = LLVIPDataset(split='train', img_size=(256, 256))
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)

    model = Stage1LHMRM(use_multiplication_in_cem=False).to(device)
    criterion = Stage1Loss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    epochs = 5
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_metrics = {'loss_contrastive': 0., 'loss_comp': 0., 'loss_cosine': 0., 'loss_structure': 0.}

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
                # print(f"Epoch [{epoch}/{epochs}] Step [{step+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")
                print(
                    f"Epoch [{epoch}/{epochs}] Step [{step+1}/{len(train_loader)}] | "
                    f"Total: {loss.item():.4f} | "
                    f"Contrastive: {loss_dict['loss_contrastive']:.4f} | "
                    f"CEM: {loss_dict['loss_comp']:.4f} | "
                    f"Cosine: {loss_dict['loss_cosine']:.4f} | "
                    f"Structure: {loss_dict['loss_structure']:.4f}"
                )

        # Save checkpoint per epoch
        ckpt_path = os.path.join(checkpoint_dir, f'stage1_epoch_{epoch}.pth')
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
        print(f" -> Avg Loss Contrastive   : {running_metrics['loss_contrastive'] / num_batches:.4f}")
        print(f" -> Avg Loss CEM           : {running_metrics['loss_comp'] / num_batches:.4f}")
        print(f" -> Avg Loss Cosine        : {running_metrics['loss_cosine'] / num_batches:.4f}")
        print(f" -> Avg Loss Structure     : {running_metrics['loss_structure'] / num_batches:.4f}")
        print(f" -> Checkpoint Saved       : {ckpt_path}")
        print(f" -> Output Feature Fu Size : {Fu.shape} (Expected: [B, 128, 64, 64])")
        print("="*65 + "\n")
        
        # print("\n" + "="*50)
        # print(f"EPOCH {epoch} SUMMARY:")
        # print(f" -> Average Total Loss: {running_loss / len(train_loader):.4f}")
        # for k, v in running_metrics.items():
        #     print(f" -> {k}: {v / len(train_loader):.4f}")
        # print(f" -> Checkpoint Saved: {ckpt_path}")
        # print(f" -> Output Feature Fu Shape: {Fu.shape} (Expected: [B, 128, 64, 64])")
        # print("="*50 + "\n")

if __name__ == '__main__':
    train()