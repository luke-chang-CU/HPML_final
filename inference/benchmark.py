import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
from vqvae import VQVAE
from data_utils import load_palette, tokens_to_image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from PIL import Image
from tqdm import tqdm
import numpy as np
def load_images_as_tensor(image_dir, num_samples=None):
    files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]
    if num_samples:
        files = files[:num_samples]
    
    images = []
    for f in files:
        img = Image.open(f).convert('RGB')
        img = img.resize((299, 299)) 
        images.append(np.array(img))
    
    if len(images) == 0:
        return torch.empty(0, 3, 299, 299)
        
    return torch.tensor(np.array(images), dtype=torch.uint8).permute(0, 3, 1, 2)

import argparse
import os
import time
import wandb
from inference_speculative import speculative_sampling

def fix_state_dict(ckpt):
    new_ckpt = {}
    for k, v in ckpt.items():
        if k.startswith('_orig_mod.'):
            new_ckpt[k[10:]] = v
        else:
            new_ckpt[k] = v
    return new_ckpt

def benchmark():
    # Configuration
    num_samples_fid = 0  # Set to 0 to skip FID and measuring speed only
    batch_size = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    wandb.init(project="speculative-decoding-distillation", name="benchmark-all-models")
    
    print(f"Using device: {device}")
    
    # Load VQ-VAE (CPU for safety)
    vqvae = VQVAE(num_hiddens=1024, num_residual_layers=2, num_residual_hiddens=256,
                  num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
    if os.path.exists('checkpoints_vqvae_miniimagenet/vqvae_final.pt'):
        vqvae.load_state_dict(torch.load('checkpoints_vqvae_miniimagenet/vqvae_final.pt', map_location='cpu'))
    vqvae.to('cpu')
    vqvae.eval()
    
    # Load Teacher (Common)
    print("Loading Teacher...")
    teacher_config = GPTConfig(1024, 256, 20, 16, 1024, num_classes=64)
    teacher = GPT(teacher_config)
    teacher.load_state_dict(fix_state_dict(torch.load('checkpoints_teacher_vqvae/teacher_final.pt', map_location=device)))
    teacher.to(device)
    teacher.eval()
    
    # Prepare Real Images for FID
    print("Preparing real images for FID...")
    real_images_dir = 'benchmark_real_samples'
    os.makedirs(real_images_dir, exist_ok=True)
    if len(os.listdir(real_images_dir)) < num_samples_fid:
        from data_utils import MiniImageNetDataset
        dataset = MiniImageNetDataset(root_dir='dataset', split='val', return_image=True)
        for i in range(num_samples_fid):
            if i >= len(dataset): break
            img_tensor = dataset[i]
            img = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(real_images_dir, f'real_{i}.png'))
            
    real_images = load_images_as_tensor(real_images_dir, num_samples_fid).to('cpu') # Keep on CPU
    
    # Metrics
    # Defer metric init to CPU phase to avoid Segfaults
    
    # Updated configs (Half-Size Hierarchy)
    student_configs = {
        1: ("10L_1024E", GPTConfig(1024, 256, 10, 16, 1024, num_classes=64)),
        2: ("9L_768E", GPTConfig(1024, 256, 9, 12, 768, num_classes=64)),
        3: ("10L_512E", GPTConfig(1024, 256, 10, 8, 512, num_classes=64)),
        4: ("9L_384E", GPTConfig(1024, 256, 9, 6, 384, num_classes=64)),
    }
    
    results = []
    
    # Scenarios: Teacher (Auto-regressive) + Students (Speculative)
    scenarios = [('Teacher', None)] + [('Student ' + name, idx) for idx, (name, _) in student_configs.items()]
    
    for name, student_id in scenarios:
        print(f"\n--- Benchmarking {name} ---")
        
        # Prepare Student if needed
        student = None
        if student_id is not None:
            student_name, student_conf = student_configs[student_id]
            student = GPT(student_conf)
            ckpt_path = f'checkpoints_distill_vqvae_v2/student_{student_name}_final.pt'
            if not os.path.exists(ckpt_path):
                print(f"Skipping {name}: Checkpoint not found.")
                continue
            student.load_state_dict(fix_state_dict(torch.load(ckpt_path, map_location=device)))
            student.to(device)
            student.eval()

            # Measure Standalone Speed for Student
            print(f"Measuring Standalone Speed for {student_name}...")
            start_t = time.time()
            num_standalone_speed_samples = 4
            for _ in range(num_standalone_speed_samples):
                _ = student.generate(torch.randint(0, 1024, (1, 1)).to(device), max_new_tokens=255, class_labels=torch.randint(0, 64, (1,), device=device))
            dur = time.time() - start_t
            speed_standalone = (num_standalone_speed_samples * 255) / dur
            print(f"Standalone Speed {student_name}: {speed_standalone:.2f} tok/s")
            
        # 1. Speed Test (Tokens/sec)
        print("Measuring Speed...")
        start_time = time.time()
        num_speed_samples = 4 
        total_tokens = 0
        
        for _ in range(num_speed_samples):
             initial_idx = torch.randint(0, 1024, (1, 1)).to(device)
             cls = torch.randint(0, 64, (1,), device=device)
             
             if student_id is None:
                 out = teacher.generate(initial_idx, max_new_tokens=255, temperature=1.0, top_k=100, class_labels=cls)
             else:
                 out = speculative_sampling(teacher, student, initial_idx, max_new_tokens=255, gamma=4, temperature=1.0, class_labels=cls)
             total_tokens += (out.shape[1] - 1)
             
        duration = time.time() - start_time
        tokens_per_sec = total_tokens / duration
        print(f"Speed: {tokens_per_sec:.2f} tokens/s")
        
        # 2. FID/IS Generation
        print(f"Generating {num_samples_fid} samples for Metrics...")
        output_dir = f'benchmark_samples_{name.replace(" ", "_")}'
        os.makedirs(output_dir, exist_ok=True)
        
        generated_count = 0
        pbar = tqdm(total=num_samples_fid)
        
        print("Generating...")
        while generated_count < num_samples_fid:
            curr_bs = 1 # Force batch size 1 for Speculative Decoding (divergence handling limitation)
            
            initial_idx = torch.randint(0, 1024, (curr_bs, 1)).to(device)
            labels = torch.randint(0, 64, (curr_bs,), device=device)
            
            with torch.no_grad():
                if student_id is None:
                     # Teacher generate (now uses KV cache internally)
                     out = teacher.generate(initial_idx, max_new_tokens=255, temperature=1.0, top_k=100, class_labels=labels)
                else:
                     out = speculative_sampling(teacher, student, initial_idx, max_new_tokens=255, gamma=4, temperature=1.0, class_labels=labels)
                
                # Decode and Save (CPU)
                indices = out.cpu()
                for b in range(curr_bs):
                    seq = indices[b].view(1, -1)
                    # Enforce strictly 256 length
                    if seq.shape[1] > 256: 
                        seq = seq[:, :256]
                    elif seq.shape[1] < 256:
                        # Pad with 0 if too short (shouldn't happen)
                        pad = torch.zeros((1, 256 - seq.shape[1]), dtype=torch.long)
                        seq = torch.cat((seq, pad), dim=1)
                    
                    # Clamp indices
                    seq = torch.clamp(seq, 0, 1023)
                    
                    decoded = vqvae.decode(seq)
                    img_tensor = decoded[0].detach().cpu().permute(1, 2, 0).clamp(0, 1)
                    
                    try:
                        img_np = (img_tensor.numpy() * 255).astype(np.uint8)
                        Image.fromarray(img_np).save(os.path.join(output_dir, f'img_{generated_count+b}.png'))
                    except Exception as e:
                        print(f"Save error: {e}")
                
            generated_count += curr_bs
            pbar.update(curr_bs)
        
        pbar.close()
        
        # Compute Metrics (on CPU to avoid Segfaults)
        print("Computing Metrics on CPU...")
        try:
            # Re-init metrics on CPU
            fid_metric_cpu = FrechetInceptionDistance(feature=2048).to('cpu')
            inception_metric_cpu = InceptionScore().to('cpu')
            
            # Load images
            fake_images_cpu = load_images_as_tensor(output_dir, num_samples_fid).to('cpu')
            real_images_cpu = load_images_as_tensor(real_images_dir, num_samples_fid).to('cpu')
            
            fid_metric_cpu.update(real_images_cpu, real=True)
            fid_metric_cpu.update(fake_images_cpu, real=False)
            fid_score = fid_metric_cpu.compute().item()
            
            inception_metric_cpu.update(fake_images_cpu)
            is_score_t, _ = inception_metric_cpu.compute()
            is_score = is_score_t.item()
        except Exception as e:
            print(f"Metric calculation failed: {e}")
            fid_score = 0.0
            is_score = 0.0
        
        print(f"Results {name}: FID={fid_score:.2f}, IS={is_score:.2f}, Speed={tokens_per_sec:.2f}")
        results.append({
            "Model": name,
            "FID": fid_score,
            "IS": is_score,
            "Speed (tok/s)": tokens_per_sec
        })
        
        wandb.log({
            f"{name}_FID": fid_score,
            f"{name}_IS": is_score,
            f"{name}_Speed": tokens_per_sec
        })
        
    print("\n\n====== FINAL REPORT ======")
    print(f"{'Model':<25} | {'FID':<10} | {'IS':<10} | {'Speed':<10}")
    print("-" * 65)
    for r in results:
        print(f"{r['Model']:<25} | {r['FID']:<10.2f} | {r['IS']:<10.2f} | {r['Speed (tok/s)']:<10.2f}")
    
    # Save to file
    with open("benchmark_results.csv", "w") as f:
        f.write("Model,FID,IS,Speed\n")
        for r in results:
             f.write(f"{r['Model']},{r['FID']},{r['IS']},{r['Speed (tok/s)']}\n")

if __name__ == "__main__":
    benchmark()
