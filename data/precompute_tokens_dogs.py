
import os
import torch
from torch.utils.data import DataLoader
from data_utils import MiniImageNetDataset
from vqvae import VQVAE
from tqdm import tqdm
import numpy as np

def precompute_tokens_dogs():
    # Configuration
    batch_size = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        torch.backends.cudnn.enabled = False
        
    model_path = 'checkpoints_vqvae_miniimagenet/vqvae_final.pt'
    output_dir = 'dataset_tokens'
    os.makedirs(output_dir, exist_ok=True)
    
    # Model Config (Must match training)
    num_hiddens = 1024
    num_residual_hiddens = 256
    num_residual_layers = 2
    embedding_dim = 256
    num_embeddings = 1024 
    commitment_cost = 0.25
    
    # Load Model
    print("Loading VQ-VAE...")
    model = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens,
                  num_embeddings, embedding_dim, commitment_cost)
    
    # Check if checkpoint exists
    if not os.path.exists(model_path):
        print(f"Checkpoint {model_path} not found. Please wait for training to finish.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Target our specific dog split
    split = 'dogs_08_17' 
    print(f"Processing {split} split...")
    
    # return_label=True is fine, but we will ignore it or use it for debug
    dataset = MiniImageNetDataset(root_dir='dataset', split=split, build_palette=False, return_image=True, return_label=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    all_tokens = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            # Encode
            # indices: (B, H*W)
            indices = model.encode(images)
            all_tokens.append(indices.cpu())
    
    all_tokens = torch.cat(all_tokens, dim=0)
    
    print(f"Saved {all_tokens.shape} tokens for {split}")
    
    # Save just tokens (unconditional training)
    # Or keep dictionary format to avoid breaking TokenDataset
    # The 'labels' will be dummy or original labels, but if we want Unconditional, we might not need them.
    # However, keeping the standard format is safer.
    
    data = {'tokens': all_tokens, 'labels': None} # No labels for unconditional
    torch.save(data, os.path.join(output_dir, f'{split}_tokens.pt'))

if __name__ == "__main__":
    precompute_tokens_dogs()
