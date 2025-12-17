import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from data_utils import MiniImageNetDataset
from vqvae import VQVAE
import time
import wandb
from tqdm import tqdm
import numpy as np
from PIL import Image

def train_vqvae():
    # Settings for config
    # TODO: Make these into parameters

    # 16 or 32
    batch_size = 16 
    # 50 or 100
    max_epochs = 50 
    learning_rate = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Disable cuDNN to avoid segfault
    if device == 'cuda':
        torch.backends.cudnn.enabled = False
        
    # Save directory
    save_dir = 'checkpoints_vqvae_miniimagenet'
    os.makedirs(save_dir, exist_ok=True)
    
    # Model Config
    num_hiddens = 1024
    num_residual_hiddens = 256
    num_residual_layers = 2
    embedding_dim = 256
    # Codebook size
    num_embeddings = 1024 
    commitment_cost = 0.25
    decay = 0.99

    # Wandb, set as luke's project id in weight and bias
    wandb.init(project="speculative-decoding-distillation", name="vqvae-mini-imagenet", config={
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "device": device,
        "model_type": "VQ-VAE",
        "num_embeddings": num_embeddings,
        "dataset": "Mini-ImageNet"
    })

    print(f"Using device: {device}")
    
    # Dataset - MiniImageNet (local)
    # Returns (3, 64, 64) normalized tensors
    dataset = MiniImageNetDataset(root_dir='dataset', split='train', return_image=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    model = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens,
                  num_embeddings, embedding_dim, commitment_cost)
    model.to(device)
    
    optimizer = Adam(model.parameters(), lr=learning_rate, amsgrad=False)
    
    model.train()
    for epoch in range(max_epochs):
        start_time = time.time()
        total_loss = 0
        total_recon_error = 0
        total_perplexity = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{max_epochs}")
        for i, images in enumerate(pbar):
            # images should be (B, 3, 64, 64) float normalized to [-1, 1] or [0, 1]
            # data_utils likely returns [0, 255] or PIL?
            # We need to ensure it returns tensors.
            
            images = images.to(device)
            
            optimizer.zero_grad()
            vq_loss, data_recon, perplexity = model(images)
            recon_error = F.mse_loss(data_recon, images)
            loss = recon_error + vq_loss
            loss.backward()
            
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_error += recon_error.item()
            total_perplexity += perplexity.item()
            
            wandb.log({
                "train_loss": loss.item(),
                "recon_error": recon_error.item(),
                "vq_loss": vq_loss.item(),
                "perplexity": perplexity.item(),
                "epoch": epoch + 1
            })
            pbar.set_postfix({"loss": loss.item(), "recon": recon_error.item()})
            
        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} finished in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(save_dir, f'vqvae_epoch_{epoch+1}.pt'))
        
        # Visualize
        # if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            # Reconstruct first batch
            # images is (B, 3, 64, 64)
            _, x_recon, _ = model(images[:8])
            
            # Convert to image
            # Assuming images are [-0.5, 0.5] or [0, 1]? 
            # If we use standard normalization, we need to denormalize.
            # Let's assume [0, 1] for now.
            
            orig_imgs = images[:8].cpu().permute(0, 2, 3, 1).numpy()
            recon_imgs = x_recon.cpu().permute(0, 2, 3, 1).numpy()
            
            # Clip
            orig_imgs = np.clip(orig_imgs, 0, 1)
            recon_imgs = np.clip(recon_imgs, 0, 1)
            
            log_images = []
            for k in range(8):
                combined = np.concatenate([orig_imgs[k], recon_imgs[k]], axis=1)
                log_images.append(wandb.Image(combined, caption=f"Orig vs Recon {k}"))
            
            wandb.log({"reconstructions": log_images})
        model.train()

    torch.save(model.state_dict(), os.path.join(save_dir, 'vqvae_final.pt'))
    print("VQ-VAE Training finished.")
    wandb.finish()

if __name__ == "__main__":
    train_vqvae()
