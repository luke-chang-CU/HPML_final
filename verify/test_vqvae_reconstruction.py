
import torch
from vqvae import VQVAE
from PIL import Image
import numpy as np
import os
import argparse

def test_reconstruction(image_path, output_prefix="vqvae_test"):
    device = 'cpu' # VQ-VAE is small enough for CPU inference for 1 image
    print(f"Using device: {device}")

    # 1. Load VQ-VAE
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

    # 2. Load and Preprocess Image
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found!")
        return

    print(f"Loading image: {image_path}")
    img = Image.open(image_path).convert('RGB')
    
    # Resize to 64x64
    img_64 = img.resize((64, 64), Image.Resampling.LANCZOS)
    img_64.save(f"{output_prefix}_original_64x64.png")
    print(f"Saved downsampled original: {output_prefix}_original_64x64.png")

    # To Tensor
    img_tensor = torch.tensor(np.array(img_64), dtype=torch.float32).permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device) # (1, 3, 64, 64)

    # 3. Reconstruct
    with torch.no_grad():
        _, x_recon, _ = vqvae(img_tensor)

    # 4. Save Reconstruction
    img_recon_tensor = x_recon[0].clamp(0, 1).permute(1, 2, 0).cpu()
    img_recon_np = (img_recon_tensor.numpy() * 255).astype(np.uint8)
    img_recon = Image.fromarray(img_recon_np)
    
    img_recon.save(f"{output_prefix}_reconstructed.png")
    print(f"Saved reconstruction: {output_prefix}_reconstructed.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    args = parser.parse_args()
    
    test_reconstruction(args.image)
