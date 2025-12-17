import os
import torch
from torch.utils.data import DataLoader
from data_utils import MiniImageNetDataset
from vqvae import VQVAE
from tqdm import tqdm
import numpy as np

def precompute_tokens():
    # Configuration
    batch_size = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        torch.backends.cudnn.enabled = False
        
    model_path = 'checkpoints_vqvae_miniimagenet/vqvae_final.pt' # Will be created after training
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
    
    for split in ['train', 'val']:
        # Ensure we return labels now
        print(f"Processing {split} split...")
        dataset = MiniImageNetDataset(root_dir='dataset', split=split, build_palette=False, return_image=True, return_label=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        all_tokens = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader):
                images = images.to(device)
                # Encode
                # indices: (B, H*W)
                indices = model.encode(images)
                all_tokens.append(indices.cpu())
                all_labels.append(labels)
        
        all_tokens = torch.cat(all_tokens, dim=0)
        all_labels = torch.cat(all_labels, dim=0) # labels are tensors now
        
        print(f"Saved {all_tokens.shape} tokens and {all_labels.shape} labels for {split}")
        data = {'tokens': all_tokens, 'labels': all_labels}
        torch.save(data, os.path.join(output_dir, f'{split}_tokens.pt'))

if __name__ == "__main__":
    precompute_tokens()
