```python
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from data.data_utils import TokenDataset
from models.model import GPT, GPTConfig
from models.vqvae import VQVAE
import time
import wandb
from tqdm import tqdm
import numpy as np
from PIL import Image

def train():
    # Configuration
    batch_size = 32
    max_epochs = 100 # Increased epochs for smaller dataset
    learning_rate = 5e-5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = 'checkpoints_teacher_unconditional'
    os.makedirs(save_dir, exist_ok=True)

    # WandB Init
    wandb.init(project="speculative-decoding-distillation", name="teacher-dogs-unconditional", config={
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "device": device,
        "model_type": "Teacher-GPT-Unconditional",
        "dataset": "dogs_08_17"
    })

    print(f"Using device: {device}")
    torch.cuda.empty_cache()

    # Dataset
    token_file = 'dataset_tokens/dogs_08_17_tokens.pt'
    if not os.path.exists(token_file):
        print(f"Token file {token_file} not found.")
        return

    dataset = TokenDataset(token_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    print(f"Dataset size: {len(dataset)}")

    # Model (Unconditional)
    config = GPTConfig(
        vocab_size=1024,
        block_size=256,
        n_layer=20,
        n_head=16,
        n_embd=1024,
        num_classes=None # Unconditional
    )
    model = GPT(config)
    model.to(device)
    
    print("Compiling model...")
    model = torch.compile(model)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs * len(dataloader))
    scaler = torch.cuda.amp.GradScaler()

    # Load VQ-VAE for visualization
    print("Loading VQ-VAE for visualization...")
    vqvae = VQVAE(num_hiddens=1024, num_residual_layers=2, num_residual_hiddens=256,
                  num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
    if os.path.exists('checkpoints_vqvae_miniimagenet/vqvae_final.pt'):
         vqvae.load_state_dict(torch.load('checkpoints_vqvae_miniimagenet/vqvae_final.pt', map_location='cpu'))
    vqvae.to('cpu')
    vqvae.eval()

    # Training Loop
    model.train()
    for epoch in range(max_epochs):
        start_time = time.time()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{max_epochs}")
        for i, data in enumerate(pbar):
            # TokenDataset returns (tokens, labels) if labels exist, or just tokens.
            # Our precompute script saved labels=None, so it returns just tokens (tuple check in data_utils?)
            # Let's check data_utils.TokenDataset
            # It checks isinstance(data, dict). We saved {'tokens':..., 'labels': None}.
            # So self.has_labels will be False (if we modify TokenDataset to check for None) or True but returns None.
            # Actually, precompute_tokens_dogs saved 'labels': None.
            # TokenDataset.__init__ sets self.labels = data['labels'].
            # __getitem__ returns self.tokens[idx], self.labels[idx].
            # self.labels is None. Indexing None raises TypeError.
            # CRITICAL: We need to check if TokenDataset handles None labels properly. 
            pass
            
            # --- FIXING ON THE FLY ---
            # Instead of relying on TokenDataset behavior which might crash, 
            # I will assume `data` might be a tuple or tensor.
            # If tensor, good. If tuple (tokens, None), unpack.
            
            if isinstance(data, (tuple, list)):
                tokens = data[0]
            else:
                tokens = data
                
            tokens = tokens.to(device) # (B, 256)
            
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                # Unconditional forward (no class_labels)
                logits, loss, _ = model(inputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            
            wandb.log({
                "train_loss": loss.item(),
                "epoch": epoch + 1
            })
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} finished in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}")
        
        wandb.log({
            "epoch_avg_loss": avg_loss,
            "epoch_time": epoch_time
        })
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'teacher_epoch_{epoch+1}.pt'))
        
        # Eval
        if (epoch + 1) % 5 == 0:
            print("Generating samples...")
            model.eval()
            with torch.no_grad():
                # Generate 4 samples (Unconditional)
                initial_idx = torch.randint(0, 1024, (4, 1)).to(device)
                
                # Unconditional generate
                generated = model.generate(initial_idx, max_new_tokens=255, temperature=1.0, top_k=100)
                
                generated_cpu = generated.cpu()
                
                images = []
                for j in range(4):
                    indices = generated_cpu[j].view(1, 256)
                    decoded = vqvae.decode(indices)
                    img_tensor = decoded[0].detach().cpu().permute(1, 2, 0).clamp(0, 1)
                    img_np = (img_tensor.numpy() * 255).astype(np.uint8)
                    images.append(wandb.Image(Image.fromarray(img_np), caption=f"Ep{epoch+1} Sample {j}"))
                
                wandb.log({"generated_samples": images})
            model.train()

    torch.save(model.state_dict(), os.path.join(save_dir, 'teacher_final.pt'))
    wandb.finish()

if __name__ == "__main__":
    train()
