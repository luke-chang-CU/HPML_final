import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from data_utils import TokenDataset
from model import GPT, GPTConfig
from vqvae import VQVAE
import time
import wandb
from tqdm import tqdm
import numpy as np
from PIL import Image

def train():
    # Configuration
    batch_size = 32 # Small seq len (256) allows larger batch size
    max_epochs = 50
    learning_rate = 5e-5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = 'checkpoints_teacher_vqvae'
    os.makedirs(save_dir, exist_ok=True)

    # WandB Init
    wandb.init(project="speculative-decoding-distillation", name="teacher-training-vqvae", config={
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "device": device,
        "model_type": "Teacher-GPT-VQVAE",
        "scheduler": "CosineAnnealing"
    })

    print(f"Using device: {device}")
    torch.cuda.empty_cache()

    # Dataset
    token_file = 'dataset_tokens/train_tokens.pt'
    if not os.path.exists(token_file):
        print(f"Token file {token_file} not found. Please run precompute_tokens.py first.")
        return

    dataset = TokenDataset(token_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    print(f"Dataset size: {len(dataset)}")

    # Model
    config = GPTConfig(
        vocab_size=1024, # VQ-VAE codebook
        block_size=256, # 16x16
        n_layer=20,
        n_head=16,
        n_embd=1024,
        num_classes=64 # 64 training classes
    )
    model = GPT(config)
    model.to(device)
    
    # Compile model for speedup
    print("Compiling model...")
    model = torch.compile(model)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs * len(dataloader))
    scaler = torch.cuda.amp.GradScaler()

    # Load VQ-VAE for visualization (CPU)
    print("Loading VQ-VAE for visualization...")
    vqvae = VQVAE(num_hiddens=1024, num_residual_layers=2, num_residual_hiddens=256,
                  num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
    # Load checkpoint if available, otherwise visualization will be garbage (but that's fine for testing code)
    if os.path.exists('checkpoints_vqvae_miniimagenet/vqvae_final.pt'): # Try to load something
         # Find latest
         pass
         # We will load inside the loop or just assume it's there. 
         # For now, let's try to load epoch 1 if it exists.
         try:
             vqvae.load_state_dict(torch.load('checkpoints_vqvae_miniimagenet/vqvae_final.pt', map_location='cpu'))
         except:
             print("Could not load VQ-VAE checkpoint. Visualization will be random.")
    
    vqvae.to('cpu')
    vqvae.eval()

    # Training Loop
    model.train()
    for epoch in range(max_epochs):
        start_time = time.time()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{max_epochs}")
        for i, (tokens, labels) in enumerate(pbar):
            tokens = tokens.to(device) # (B, 256)
            labels = labels.to(device) # (B,)
            
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                logits, loss, _ = model(inputs, targets, class_labels=labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Log to WandB
            wandb.log({
                "train_loss": loss.item(),
                "epoch": epoch + 1,
                "batch": i,
                "lr": scheduler.get_last_lr()[0]
            })
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} finished in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}")
        
        wandb.log({
            "epoch_avg_loss": avg_loss,
            "epoch_time": epoch_time
        })
        
        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(save_dir, f'teacher_epoch_{epoch+1}.pt'))
        
        # Generate and log samples
        print("Generating evaluation samples...")
        model.eval()
        with torch.no_grad():
            # Generate 4 samples for random classes
            # Pick 4 random classes from 0-63
            sample_labels = torch.randint(0, 64, (4,), device=device)
            
            initial_idx = torch.randint(0, 1024, (4, 1)).to(device)
            # Generate 255 more tokens
            generated = model.generate(initial_idx, max_new_tokens=255, temperature=1.0, top_k=100, class_labels=sample_labels)
            
            # Decode with VQ-VAE (CPU)
            generated_cpu = generated.cpu() # (4, 256)
            
            # Load visualization dataset (lazy load)
            if not hasattr(train, 'vis_dataset'):
                from data_utils import MiniImageNetDataset
                # Use 'train' split because that's where the classes 0-63 come from
                print("Loading visualization dataset (train split)...")
                train.vis_dataset = MiniImageNetDataset(split='train', return_image=True)

            images = []
            for j in range(4):
                # Generated Image
                indices = generated_cpu[j].view(1, 256)
                decoded = vqvae.decode(indices)
                img_tensor_gen = decoded[0].detach().cpu() # (3, 64, 64)
                
                # Real Image for this class
                class_idx = sample_labels[j].item()
                img_tensor_real = train.vis_dataset.get_class_image(class_idx) # (3, 64, 64)
                
                # Concatenate (Side by Side)
                # (3, 64, 128)
                combined = torch.cat((img_tensor_gen, img_tensor_real), dim=2)
                
                img_np = combined.permute(1, 2, 0).clamp(0, 1).numpy()
                img = (img_np * 255).astype(np.uint8)
                img = Image.fromarray(img)
                
                images.append(wandb.Image(img, caption=f"Ep{epoch+1} Cls{class_idx}: Gen | Real"))
            
            wandb.log({"generated_samples": images})
        model.train()

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'teacher_final.pt'))
    print("Training finished.")
    wandb.finish()

if __name__ == "__main__":
    train()
