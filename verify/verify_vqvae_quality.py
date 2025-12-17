
import os
import torch
import numpy as np
import csv
from PIL import Image
from vqvae import VQVAE

# Load 5 random images from CSV and verify VQ-VAE quality
# Compare original and reconstructed images side-by-side

# Check VQ-VAE quality for 5 training images
def verify_vqvae_quality():
    # 1. Config
    data_csv = 'dataset/dogs_08_17.csv'
    output_dir = 'verify_vqvae_samples'
    os.makedirs(output_dir, exist_ok=True)
    num_samples = 5
    device = 'cpu'

    print(f"Using device: {device}")

    # 2. Load VQ-VAE
    print("Loading VQ-VAE...")
    vqvae = VQVAE(num_hiddens=1024, num_residual_layers=2, num_residual_hiddens=256,
                  num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
    
    ckpt_path = 'checkpoints_vqvae_miniimagenet/vqvae_final.pt'
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} not found!")
        return

    vqvae.load_state_dict(torch.load(ckpt_path, map_location=device))
    vqvae.to(device)
    vqvae.eval()

    # 3. Load 5 Random Images from CSV
    print(f"Sampling {num_samples} images from {data_csv}...")
    image_paths = []
    with open(data_csv, 'r') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
        # Random sample
        sampled_rows = np.random.choice(all_rows, num_samples, replace=False)
        for row in sampled_rows:
            image_paths.append(os.path.join('dataset/images', row['filename']))

    # 4. Process and Save
    for i, img_path in enumerate(image_paths):
        print(f"Processing {img_path}...")
        img_name = os.path.basename(img_path)
        
        # Load and Resize
        img_pil = Image.open(img_path).convert('RGB')
        img_64 = img_pil.resize((64, 64), Image.Resampling.LANCZOS)
        
        # To Tensor (1, 3, 64, 64)
        img_tensor = torch.tensor(np.array(img_64), dtype=torch.float32).permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Reconstruct
        with torch.no_grad():
            _, x_recon, _ = vqvae(img_tensor)
            
        # Post-process
        img_recon_tensor = x_recon[0].clamp(0, 1).permute(1, 2, 0)
        img_recon_np = (img_recon_tensor.numpy() * 255).astype(np.uint8)
        img_recon = Image.fromarray(img_recon_np)
        
        # Concatenate Side-by-Side
        combined = Image.new('RGB', (128, 64))
        combined.paste(img_64, (0, 0))
        combined.paste(img_recon, (64, 0))
        
        save_path = os.path.join(output_dir, f'vqvae_comparison_{i}_{img_name}')
        combined.save(save_path)
        print(f"Saved {save_path}")

if __name__ == "__main__":
    verify_vqvae_quality()
