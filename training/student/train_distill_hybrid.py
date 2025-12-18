
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
import time
import wandb
from tqdm import tqdm


from models.vqvae import VQVAE
from PIL import Image
import numpy as np
import argparse

def train_distill_hybrid():
    parser = argparse.ArgumentParser()
    parser.add_argument('--student_id', type=int, default=1, help='Student ID (1-4)')
    args = parser.parse_args()
    
    # Configuration
    student_id = args.student_id
    batch_size = 32
    max_epochs = 100
    learning_rate = 1e-4
    
    # Hybrid settings
    temperature = 1.5 
    alpha_early = 0.2 # For first 32 tokens
    alpha_late = 0.0  # For rest (Pure Imitation)
    cutoff_token = 32
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = 'checkpoints_distill_hybrid'
    os.makedirs(save_dir, exist_ok=True)
    
    # Student Configs
    student_configs = {
        1: ("10L_1024E", GPTConfig(1024, 256, 10, 16, 1024, num_classes=None)),
        2: ("8L_512E",  GPTConfig(1024, 256, 8, 8, 512, num_classes=None)),
        3: ("6L_384E",  GPTConfig(1024, 256, 6, 6, 384, num_classes=None)),
        4: ("4L_256E",  GPTConfig(1024, 256, 4, 4, 256, num_classes=None)),
    }
    
    if student_id not in student_configs:
        print(f"Invalid student_id {student_id}. Available: {list(student_configs.keys())}")
        return

    student_name, student_conf = student_configs[student_id]

    # WandB Init
    wandb.init(project="speculative-decoding-distillation", name=f"distill-hybrid-{student_name}", config={
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "device": device,
        "student_config": student_name,
        "temperature": temperature,
        "alpha_early": alpha_early,
        "alpha_late": alpha_late,
        "cutoff_token": cutoff_token,
        "type": "Hybrid Distillation"
    })

    print(f"Using device: {device}")
    
    # Load Dataset (Full Sequences used by default, starting at Top-Left)
    token_file = 'dataset_tokens/dogs_08_17_tokens.pt'
    dataset = TokenDataset(token_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print(f"Dataset size: {len(dataset)}")

    # 1. Load Teacher
    print("Loading Teacher...")
    teacher_conf = GPTConfig(1024, 256, 20, 16, 1024, num_classes=None)
    teacher = GPT(teacher_conf)
    
    # Use Epoch 100 or Final? User implied overfit teacher is the challenge.
    teacher_ckpt = 'checkpoints_teacher_unconditional/teacher_final.pt'
    if not os.path.exists(teacher_ckpt):
         # Fallback to epoch 100 if final not named exactly final
         teacher_ckpt = 'checkpoints_teacher_unconditional/teacher_epoch_100.pt'

    print(f"Using Teacher Checkpoint: {teacher_ckpt}")
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

    # Create Alpha Mask
    # Shape (T,) -> (255,) since input is 255 length
    # Time steps 0..31 -> Alpha=0.2
    # Time steps 32..254 -> Alpha=0.0
    # Inputs: (B, T)
    seq_len = 255
    alpha_mask = torch.zeros(seq_len, device=device)
    alpha_mask[:cutoff_token] = alpha_early
    alpha_mask[cutoff_token:] = alpha_late
    # Reshape for broadcasting (1, T)
    alpha_mask = alpha_mask.view(1, seq_len)

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
                teacher_logits, _, _ = teacher(inputs, all_logits=True)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                # Student Forward
                # student_logits: (B, T, V)
                # hard_loss (scalar) is returned by model.py but averages over T.
                # We need per-token hard loss to apply masking correctly?
                # Actually model.py returns scalar hard_loss (mean reduction).
                # To apply hybrid alpha correctly spatially, we should compute hard loss manually here.
                
                # Student Forward
                # We need all logits for the full sequence to match targets
                student_logits, _, _ = student(inputs, targets=None, all_logits=True)
                
                # --- Manual Loss Calculation ---
                B, T, V = student_logits.shape
                
                # 1. Hard Loss (CE) per token
                # student_logits: (B, T, V) -> (B*T, V)
                # targets: (B, T) -> (B*T)
                loss_hard_per_token = F.cross_entropy(student_logits.reshape(B*T, V), targets.reshape(B*T), reduction='none')
                loss_hard_per_token = loss_hard_per_token.view(B, T)
                
                # 2. Soft Loss (KL) per token
                s_logits_scaled = student_logits / temperature
                t_logits_scaled = teacher_logits / temperature
                
                s_probs = F.log_softmax(s_logits_scaled, dim=-1)
                t_probs = F.softmax(t_logits_scaled, dim=-1)
                
                # F.kl_div with reduction='none' returns (B, T, V)
                # We sum over V to get per-token KL
                loss_soft_per_token = F.kl_div(s_probs, t_probs, reduction='none').sum(dim=-1) * (temperature ** 2)
                
                # 3. Combine with Mask
                # loss = alpha * hard + (1-alpha) * soft
                # alpha_mask is (1, T)
                
                combined_loss_per_token = (alpha_mask * loss_hard_per_token) + ((1.0 - alpha_mask) * loss_soft_per_token)
                
                loss = combined_loss_per_token.mean()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            # For logging, rough approx
            total_soft += loss_soft_per_token.mean().item()
            total_hard += loss_hard_per_token.mean().item()
            
            wandb.log({
                "train_loss": loss.item(),
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
                # Use Top-Left Token (0) for consistency with training
                token_path = 'dataset_tokens/dogs_08_17_tokens.pt'
                data = torch.load(token_path)
                if isinstance(data, dict):
                    all_tokens = data['tokens']
                else:
                    all_tokens = data
                
                # Sample 4 random images and take their first token for initialization
                rand_indices = torch.randint(0, len(all_tokens), (4,))
                initial_idx = all_tokens[rand_indices, 0:1].to(device)
                
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
    train_distill_hybrid()
