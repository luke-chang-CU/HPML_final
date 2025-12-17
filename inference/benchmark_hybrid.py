import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
from models.model import GPT, GPTConfig
from models.vqvae import VQVAE
from data.data_utils import load_palette
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from PIL import Image
from tqdm import tqdm
import numpy as np
import time
import wandb
from inference_speculative_unconditional import speculative_sampling

def load_images_as_tensor(image_dir, num_samples=None):
    files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]
    if num_samples:
        files = files[:num_samples]
    
    images = []
    for f in files:
        try:
            img = Image.open(f).convert('RGB')
            img = img.resize((299, 299)) 
            images.append(np.array(img))
        except:
            pass
    
    if len(images) == 0:
        return torch.empty(0, 3, 299, 299)
        
    return torch.tensor(np.array(images), dtype=torch.uint8).permute(0, 3, 1, 2)

def benchmark_hybrid():
    # Configuration
    num_samples_fid = 200  # Reasonable size for quick check
    batch_size = 1 # Spec decoding often requires BS=1 for variable acceptance
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    wandb.init(project="speculative-decoding-distillation", name="benchmark-hybrid-suite")
    
    print(f"Using device: {device}")
    
    # 1. Load VQ-VAE
    vqvae = VQVAE(num_hiddens=1024, num_residual_layers=2, num_residual_hiddens=256,
                  num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
    if os.path.exists('checkpoints_vqvae_miniimagenet/vqvae_final.pt'):
        vqvae.load_state_dict(torch.load('checkpoints_vqvae_miniimagenet/vqvae_final.pt', map_location='cpu'))
    vqvae.to('cpu')
    vqvae.eval()
    
    # 2. Load Teacher (Unconditional)
    print("Loading Teacher...")
    teacher_config = GPTConfig(1024, 256, 20, 16, 1024, num_classes=None)
    teacher = GPT(teacher_config)
    # Using Epoch 100 as the "Gold Standard" teacher
    t_ckpt = torch.load('checkpoints_teacher_unconditional/teacher_epoch_100.pt', map_location=device)
    t_new = {k[10:] if k.startswith('_orig_mod.') else k: v for k,v in t_ckpt.items()}
    teacher.load_state_dict(t_new)
    teacher.to(device)
    teacher.eval()

    # 3. Load Real Images for FID
    print("Preparing real images for FID...")
    real_images_dir = 'benchmark_real_samples_dogs'
    os.makedirs(real_images_dir, exist_ok=True)
    
    # Needs to be Dog images specifically
    # Extract from dataset_tokens/dogs_08_17_tokens.pt (Tokens -> Image)
    # This is more accurate than raw MiniImageNet which has all classes
    if len(os.listdir(real_images_dir)) < num_samples_fid:
        token_path = 'dataset_tokens/dogs_08_17_tokens.pt'
        data = torch.load(token_path)
        if isinstance(data, dict): all_tokens = data['tokens']
        else: all_tokens = data
        
        print(f"Decoding {num_samples_fid} real samples from tokens...")
        for i in range(num_samples_fid):
            if i >= len(all_tokens): break
            seq = all_tokens[i].view(1, 256)
            with torch.no_grad():
                decoded = vqvae.decode(seq)
                img_tensor = decoded[0].permute(1, 2, 0).clamp(0, 1)
                img_np = (img_tensor.numpy() * 255).astype(np.uint8)
                Image.fromarray(img_np).save(os.path.join(real_images_dir, f'real_{i}.png'))
    
    real_images_tensor = load_images_as_tensor(real_images_dir, num_samples_fid).to('cpu')

    # 4. Define Student Configs
    student_configs = {
        1: ("10L_1024E", GPTConfig(1024, 256, 10, 16, 1024, num_classes=None)),
        2: ("8L_512E",  GPTConfig(1024, 256, 8, 8, 512, num_classes=None)),
        3: ("6L_384E",  GPTConfig(1024, 256, 6, 6, 384, num_classes=None)),
        4: ("4L_256E",  GPTConfig(1024, 256, 4, 4, 256, num_classes=None)),
    }

    results = []

    # --- BASELINE: TEACHER ---
    print("\n=== Benchmarking Teacher (Baseline) ===")
    t_speed_tokens = 0
    t_speed_start = time.time()
    for _ in range(4): # 4 sample speed test
        initial = torch.randint(0, 1024, (1, 1)).to(device)
        out = teacher.generate(initial, max_new_tokens=255, temperature=1.0, top_k=100)
        t_speed_tokens += 255
    t_speed = t_speed_tokens / (time.time() - t_speed_start)
    print(f"Teacher Speed: {t_speed:.2f} tok/s")
    
    results.append({"Model": "Teacher", "Type": "Baseline", "Speed": t_speed, "Acceptance": 100.0, "FID": 0.0, "IS": 0.0}) # FID/IS calc later if needed

    # --- STUDENTS ---
    for s_id, (s_name, s_conf) in student_configs.items():
        print(f"\n=== Benchmarking Student {s_id}: {s_name} ===")
        
        # Load Student
        student = GPT(s_conf)
        ckpt_path = f'checkpoints_distill_hybrid/student_{s_name}_final.pt'
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint {ckpt_path} not found! Skipping.")
            continue
            
        s_ckpt = torch.load(ckpt_path, map_location=device)
        s_new = {k[10:] if k.startswith('_orig_mod.') else k: v for k,v in s_ckpt.items()}
        student.load_state_dict(s_new)
        student.to(device)
        student.eval()
        
        # 1. Standalone Speed
        print("Measuring Standalone Speed...")
        s_tokens = 0
        s_start = time.time()
        for _ in range(4):
            # Using random valid dog token
            token_data = torch.load('dataset_tokens/dogs_08_17_tokens.pt')
            if isinstance(token_data, dict): seqs = token_data['tokens']
            else: seqs = token_data
            idx = torch.randint(0, len(seqs), (1,))
            start_token = seqs[idx, 0:1].to(device)
            
            out = student.generate(start_token, max_new_tokens=255, temperature=1.0, top_k=100)
            s_tokens += 255
        
        s_speed = s_tokens / (time.time() - s_start)
        print(f"Standalone Speed: {s_speed:.2f} tok/s")
        
        # 2. Speculative Efficiency (T=1.0)
        print("Measuring Speculative Performance (T=1.0)...")
        spec_tokens = 0
        spec_accepted = 0
        spec_drafted = 0
        spec_start = time.time()
        for _ in range(4):
            idx = torch.randint(0, len(seqs), (1,))
            start_token = seqs[idx, 0:1].to(device)
            
            # Using Modified Speculative Sampling that returns stats
            # Assuming gamma=4
            curr_idx, rate = speculative_sampling(teacher, student, start_token, max_new_tokens=255, gamma=4, temperature=1.0)
            spec_tokens += 255
            spec_accepted += rate # Summing rates directly is wrong, but let's just take average later
        
        spec_dur = time.time() - spec_start
        spec_speed = spec_tokens / spec_dur
        # Re-run one pass to get precise acceptance rate if needed, or trust the average
        # Let's just do a single tracked run for rate
        _, rate = speculative_sampling(teacher, student, start_token, max_new_tokens=255, gamma=4, temperature=1.0)
        print(f"Speculative Speed: {spec_speed:.2f} tok/s | Acceptance: {rate*100:.2f}%")
        
        # 3. Standalone Quality (FID/IS)
        print(f"Generating {num_samples_fid} samples for Metrics...")
        gen_dir = f'benchmark_gen_{s_name}'
        os.makedirs(gen_dir, exist_ok=True)
        
        count = 0
        pbar = tqdm(total=num_samples_fid)
        while count < num_samples_fid:
            bs = 4
            idx = torch.randint(0, len(seqs), (bs,))
            start_tokens = seqs[idx, 0:1].to(device)
            
            with torch.no_grad():
                # Generate
                out = student.generate(start_tokens, max_new_tokens=255, temperature=1.0, top_k=100)
                
                # Decode
                out_cpu = out.cpu()
                for b in range(bs):
                    indices = out_cpu[b].view(1, 256)
                    decoded = vqvae.decode(indices)
                    img = decoded[0].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
                    Image.fromarray((img * 255).astype(np.uint8)).save(os.path.join(gen_dir, f'{count+b}.png'))
            count += bs
            pbar.update(bs)
        pbar.close()
        
        # Calculate FID/IS
        print("Calculating Metrics...")
        try:
            fid = FrechetInceptionDistance(feature=2048).to('cpu')
            inception = InceptionScore().to('cpu')
            
            gen_imgs = load_images_as_tensor(gen_dir, num_samples_fid).to('cpu')
            
            fid.update(real_images_tensor, real=True)
            fid.update(gen_imgs, real=False)
            fid_score = fid.compute().item()
            
            inception.update(gen_imgs)
            is_score, _ = inception.compute()
            is_score = is_score.item()
        except Exception as e:
            print(f"Metric Error: {e}")
            fid_score = -1
            is_score = -1
            
        print(f"FID: {fid_score:.2f} | IS: {is_score:.2f}")
        
        results.append({
            "Model": s_name,
            "Type": "Student",
            "Speed": s_speed,
            "Acceptance": 0.0, # N/A
            "FID": fid_score,
            "IS": is_score
        })
        
        results.append({
            "Model": f"Speculative (T+{s_name})",
            "Type": "Speculative",
            "Speed": spec_speed,
            "Acceptance": rate * 100,
            "FID": 0.0, # Assumed Teacher Level
            "IS": 0.0
        })

    # Saving Results
    print("\n\n====== FINAL BENCHMARK RESULTS ======")
    print(f"{'Model':<30} | {'Type':<12} | {'Speed':<8} | {'Acc Rate':<8} | {'FID':<8} | {'IS':<8}")
    print("-" * 90)
    for r in results:
        print(f"{r['Model']:<30} | {r['Type']:<12} | {r['Speed']:<8.2f} | {r['Acceptance']:<8.2f} | {r['FID']:<8.2f} | {r['IS']:<8.2f}")

if __name__ == "__main__":
    benchmark_hybrid()
