
import os
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
from PIL import Image
import numpy as np

def train_distill_unconditional():
    # Configuration
    student_id = 1  # Largest Student (Half-Size of Teacher)
    batch_size = 32
    max_epochs = 100
    learning_rate = 1e-4
    
    # Pure Imitation (V3)
    temperature = 1.0 
    alpha = 0.0 # 100% Soft Loss
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = 'checkpoints_distill_unconditional'
    os.makedirs(save_dir, exist_ok=True)
    
    # Student Configs (Half-Size Hierarchy)
    student_configs = {
        1: ("10L_1024E", GPTConfig(1024, 256, 10, 16, 1024, num_classes=None)),
        4: ("9L_384E", GPTConfig(1024, 256, 9, 6, 384, num_classes=None)),
    }
    
    student_name, student_conf = student_configs[student_id]

    # WandB Init
    wandb.init(project="speculative-decoding-distillation", name=f"distill-dog-unconditional-{student_name}", config={
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "device": device,
        "student_config": student_name,
        "temperature": temperature,
        "alpha": alpha,
        "type": "Unconditional Distillation"
    })

    print(f"Using device: {device}")
    
    # Load Dataset
    token_file = 'dataset_tokens/dogs_08_17_tokens.pt'
    dataset = TokenDataset(token_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print(f"Dataset size: {len(dataset)}")

    # 1. Load Teacher
    print("Loading Teacher...")
    teacher_conf = GPTConfig(1024, 256, 20, 16, 1024, num_classes=None)
    teacher = GPT(teacher_conf)
    
    teacher_ckpt = 'checkpoints_teacher_unconditional/teacher_final.pt'
    # Wait for teacher if running in pipeline, or just fail if manual
    if not os.path.exists(teacher_ckpt):
        print(f"Teacher checkpoint {teacher_ckpt} not found! Ensure teacher training finishes.")
        # Try finding latest epoch?
        # For now, just warn.
    else:
        # Load state dict (handle compilation prefix)
        st = torch.load(teacher_ckpt, map_location=device)
        new_st = {}
        for k, v in st.items():
            if k.startswith('_orig_mod.'):
                new_st[k[10:]] = v
            else:
                new_st[k] = v
        teacher.load_state_dict(new_st)
    
    teacher.to(device)
    teacher.eval()
    
    # 2. Initialize Student
    print(f"Initializing Student: {student_name}")
    student = GPT(student_conf)
    student.to(device)
    
    optimizer = AdamW(student.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs * len(dataloader))
    scaler = torch.cuda.amp.GradScaler()

    # Load VQ-VAE for visualization (CPU)
    print("Loading VQ-VAE for visualization...")

    
    vqvae = VQVAE(num_hiddens=1024, num_residual_layers=2, num_residual_hiddens=256,
                  num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
    if os.path.exists('checkpoints_vqvae_miniimagenet/vqvae_final.pt'):
         vqvae.load_state_dict(torch.load('checkpoints_vqvae_miniimagenet/vqvae_final.pt', map_location='cpu'))
    vqvae.to('cpu')
    vqvae.eval()

    # Training Loop
    student.train()
    for epoch in range(max_epochs):
        start_time = time.time()
        total_loss = 0
        total_soft = 0
        total_hard = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{max_epochs}")
        for i, data in enumerate(pbar):
            if isinstance(data, (tuple, list)):
                tokens = data[0]
            else:
                tokens = data
            
            tokens = tokens.to(device)
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]
            
            with torch.no_grad():
                # Teacher Forward (Get Logits)
                # all_logits=True to get full sequence
                teacher_logits, _, _ = teacher(inputs, all_logits=True)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                # Student Forward
                student_logits, hard_loss, _ = student(inputs, targets)
                
                # Distillation Loss
                B, T, V = student_logits.shape
                
                # Soft Loss (KL Divergence)
                # T=1.0 (Fixed)
                s_logits_scaled = student_logits / temperature
                t_logits_scaled = teacher_logits / temperature
                
                s_probs = F.log_softmax(s_logits_scaled, dim=-1)
                t_probs = F.softmax(t_logits_scaled, dim=-1)
                
                soft_loss = F.kl_div(s_probs, t_probs, reduction='batchmean') * (temperature ** 2)
                
                # Combined Loss
                loss = (alpha * hard_loss) + ((1.0 - alpha) * soft_loss)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            total_soft += soft_loss.item()
            if hard_loss is not None:
                total_hard += hard_loss.item()
            
            wandb.log({
                "train_loss": loss.item(),
                "soft_loss": soft_loss.item(),
                "hard_loss": (hard_loss.item() if hard_loss else 0),
                "epoch": epoch + 1
            })
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} finished in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save(student.state_dict(), os.path.join(save_dir, f'student_{student_name}_epoch_{epoch+1}.pt'))
        
        # Generation Check (Every 5 Epochs)
        if (epoch + 1) % 5 == 0:
            print("Generating student samples...")
            student.eval()
            with torch.no_grad():
                # Unconditional Generation
                initial_idx = torch.randint(0, 1024, (4, 1)).to(device)
                
                # Unconditional generate
                generated = student.generate(initial_idx, max_new_tokens=255, temperature=1.0, top_k=100)
                
                generated_cpu = generated.cpu()
                
                images = []
                for j in range(4):
                    indices = generated_cpu[j].view(1, 256)
                    decoded = vqvae.decode(indices)
                    img_tensor = decoded[0].detach().cpu().permute(1, 2, 0).clamp(0, 1)
                    img_np = (img_tensor.numpy() * 255).astype(np.uint8)
                    images.append(wandb.Image(Image.fromarray(img_np), caption=f"Stud Ep{epoch+1} Sample {j}"))
                
                wandb.log({"student_samples": images})
            student.train()

    torch.save(student.state_dict(), os.path.join(save_dir, f'student_{student_name}_final.pt'))
    wandb.finish()

if __name__ == "__main__":
    train_distill_unconditional()
