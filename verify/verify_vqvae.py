import torch
from vqvae import VQVAE
from data_utils import MiniImageNetDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

def verify_vqvae():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.enabled = False # Keep consistent with training
        
    # Load Model
    model = VQVAE(num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32,
                  num_embeddings=1024, embedding_dim=64, commitment_cost=0.25)
    
    # Find latest checkpoint
    ckpt_dir = 'checkpoints_vqvae_retrain'
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
    if not ckpts:
        print("No checkpoints found.")
        return
    
    # Sort by epoch
    ckpts.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if 'epoch' in x else 0)
    latest_ckpt = os.path.join(ckpt_dir, ckpts[-1])
    print(f"Loading checkpoint: {latest_ckpt}")
    
    model.load_state_dict(torch.load(latest_ckpt, map_location=device))
    model.to(device)
    model.eval()
    
    # Load Data
    dataset = MiniImageNetDataset(root_dir='dataset', split='val', build_palette=False, return_image=True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    
    # Get a batch
    images = next(iter(dataloader))
    images = images.to(device)
    
    with torch.no_grad():
        loss, reconstructed, _ = model(images)
    
    # Save comparisons
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    reconstructed = reconstructed.cpu().permute(0, 2, 3, 1).numpy()
    
    # Clamp
    reconstructed = np.clip(reconstructed, 0, 1)
    
    output_dir = 'vqvae_verification'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving comparisons to {output_dir}...")
    
    for i in range(len(images)):
        orig = (images[i] * 255).astype(np.uint8)
        recon = (reconstructed[i] * 255).astype(np.uint8)
        
        # Concatenate side by side
        combined = np.concatenate([orig, recon], axis=1)
        
        Image.fromarray(combined).save(os.path.join(output_dir, f'comparison_{i}.png'))
        
    print("Done.")

if __name__ == "__main__":
    verify_vqvae()
