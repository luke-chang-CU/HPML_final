import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from data.data_utils import TokenDataset
from models.model import GPT, GPTConfig
from models.vqvae import VQVAE
import time
import wandb
from tqdm import tqdm
import argparse
import numpy as np
from PIL import Image

def train_distill(student_config_name, student_config):
    # Configuration
    batch_size = 32 # Small seq len
    max_epochs = 30
    learning_rate = 1e-4
    
    # Distillation V3 Hyperparameters (Pure Imitation)
    start_temp = 1.0
    end_temp = 1.0 # Constant temperature
    alpha = 0.0    # 100% Soft Loss, 0% Hard Loss
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = 'checkpoints_distill_vqvae_v2'
    os.makedirs(save_dir, exist_ok=True)

    # WandB Init
    wandb.init(project="speculative-decoding-distillation", name=f"distill-v3-{student_config_name}", config={
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "device": device,
        "student_config": student_config_name,
        "start_temp": start_temp,
        "end_temp": end_temp,
        "alpha": alpha,
        "note": "pure_imitation_T1_alpha0"
    })

    print(f"Using device: {device}")
    torch.cuda.empty_cache()

    # Dataset
    token_file = 'dataset_tokens/train_tokens.pt'
    if not os.path.exists(token_file):
        print(f"Token file {token_file} not found.")
        return

    dataset = TokenDataset(token_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Teacher Model (Frozen)
    # Must match teacher config in train_teacher.py
    teacher_config = GPTConfig(1024, 256, 20, 16, 1024, num_classes=64)
    teacher = GPT(teacher_config)
    # Load teacher checkpoint
    teacher_path = 'checkpoints_teacher_vqvae/teacher_final.pt'
    if not os.path.exists(teacher_path):
        print(f"Teacher checkpoint {teacher_path} not found. Train teacher first.")
        # For testing, we might proceed or return. Let's return.
        return
        
    state_dict = torch.load(teacher_path, map_location=device)
    # Fix for compiled model saving if needed
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
            
    teacher.load_state_dict(new_state_dict)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Student Model
    # Add num_classes to student config
    student_config.num_classes = 64
    student = GPT(student_config)
    student.to(device)
    
    # Compile student
    # student = torch.compile(student)

    optimizer = AdamW(student.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    
    # Load VQ-VAE for visualization (CPU)
    print("Loading VQ-VAE for visualization...")
    vqvae = VQVAE(num_hiddens=1024, num_residual_layers=2, num_residual_hiddens=256,
                  num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
    if os.path.exists('checkpoints_vqvae_miniimagenet/vqvae_final.pt'):
         try:
             vqvae.load_state_dict(torch.load('checkpoints_vqvae_miniimagenet/vqvae_final.pt', map_location='cpu'))
         except:
             pass
    vqvae.to('cpu')
    vqvae.eval()

    # Training Loop
    student.train()
    for epoch in range(max_epochs):
        start_time = time.time()
        total_loss = 0
        
        # Constant temperature
        current_temp = start_temp
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{max_epochs} (T={current_temp:.2f})")
        for i, (tokens, labels) in enumerate(pbar):
            tokens = tokens.to(device)
            labels = labels.to(device)
            
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    logits_T, _, _ = teacher(inputs, class_labels=labels, all_logits=True)
                
                # Student forward
                logits_S, loss_hard, _ = student(inputs, targets, class_labels=labels)
                
                # Distillation Loss
                # Soft targets
                # KLDiv equivalent: cross_entropy(log_softmax(S), softmax(T))
                # Note: We divide logits by T before softmax
                
                loss_soft = F.cross_entropy(
                    logits_S.view(-1, logits_S.size(-1)) / current_temp,
                    F.softmax(logits_T.view(-1, logits_T.size(-1)) / current_temp, dim=-1)
                ) * (current_temp ** 2)
                
                loss = alpha * loss_hard + (1 - alpha) * loss_soft
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            wandb.log({
                "train_loss": loss.item(),
                "loss_hard": loss_hard.item(),
                "loss_soft": loss_soft.item(),
                "epoch": epoch + 1,
                "batch": i,
                "temperature": current_temp
            })
            pbar.set_postfix({"loss": loss.item(), "soft": f"{loss_soft.item():.2f}"})
        
        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} finished in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}, Temp: {current_temp:.2f}")
        
        wandb.log({
            "epoch_avg_loss": avg_loss,
            "epoch_time": epoch_time
        })
        
        # Save checkpoint
        torch.save(student.state_dict(), os.path.join(save_dir, f'student_{student_config_name}_epoch_{epoch+1}.pt'))
        
        # Generate samples
        if (epoch + 1) % 5 == 0: 
            print("Generating evaluation samples...")
            student.eval()
            with torch.no_grad():
                # Random class labels
                sample_labels = torch.randint(0, 64, (4,), device=device)
                initial_idx = torch.randint(0, 1024, (4, 1)).to(device)
                generated = student.generate(initial_idx, max_new_tokens=255, temperature=1.0, top_k=100, class_labels=sample_labels)
                
                generated_cpu = generated.cpu()
                images = []
                for j in range(4):
                    indices = generated_cpu[j].view(1, 256)
                    decoded = vqvae.decode(indices)
                    img_tensor = decoded[0].permute(1, 2, 0).clamp(0, 1).numpy()
                    img = (img_tensor * 255).astype(np.uint8)
                    img = Image.fromarray(img)
                    images.append(wandb.Image(img, caption=f"Epoch {epoch+1} Class {sample_labels[j].item()}"))
                wandb.log({"generated_samples": images})
            student.train()

    torch.save(student.state_dict(), os.path.join(save_dir, f'student_{student_config_name}_final.pt'))
    print("Distillation finished.")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--student_id', type=int, required=True, help='1, 2, 3, or 4')
    args = parser.parse_args()
    
    # Scale up for better analysis against ~250M Teacher
    # Updated configs: Halving hierarchy starting from Teacher (~253M)
    configs = {
        1: ("10L_1024E", GPTConfig(1024, 256, 10, 16, 1024, num_classes=64)), # ~126M params (Half of Teacher)
        2: ("9L_768E", GPTConfig(1024, 256, 9, 12, 768, num_classes=64)),     # ~63M params (Half of S1)
        3: ("10L_512E", GPTConfig(1024, 256, 10, 8, 512, num_classes=64)),    # ~31M params (Half of S2)
        4: ("9L_384E", GPTConfig(1024, 256, 9, 6, 384, num_classes=64)),      # ~15M params (Half of S3)
    }
    
    name, config = configs[args.student_id]
    train_distill(name, config)
