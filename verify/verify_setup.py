import os
import torch
import numpy as np
from PIL import Image
from data_utils import MiniImageNetDataset, load_palette, tokens_to_image
import matplotlib.pyplot as plt

def verify_reconstruction():
    # Load dataset and palette
    dataset = MiniImageNetDataset(root_dir='dataset', split='val', build_palette=False)
    palette = load_palette()
    
    # Get a few samples
    indices = np.random.choice(len(dataset), 5, replace=False)
    
    os.makedirs('verification_samples', exist_ok=True)
    
    for i, idx in enumerate(indices):
        # Get quantized tokens
        tokens = dataset[idx] # (1024,)
        
        # Reconstruct image from tokens
        reconstructed_img = tokens_to_image(tokens, palette)
        
        # Get original image
        # dataset.ds is the HF dataset
        original_img = dataset.ds[idx]['image'].convert('RGB').resize((64, 64))
        
        # Save comparison
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(original_img)
        ax[0].set_title("Original (64x64)")
        ax[0].axis('off')
        
        ax[1].imshow(reconstructed_img)
        ax[1].set_title("Quantized Reconstruction")
        ax[1].axis('off')
        
        plt.savefig(f'verification_samples/verify_{i}.png')
        plt.close()
        print(f"Saved verification_samples/verify_{i}.png")

if __name__ == "__main__":
    verify_reconstruction()
