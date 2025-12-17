import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
from models.model import GPT, GPTConfig
from models.vqvae import VQVAE
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from PIL import Image
from tqdm import tqdm
import numpy as np
import time

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

def benchmark_teacher_quality():
    num_samples = 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Load VQ-VAE
    vqvae = VQVAE(num_hiddens=1024, num_residual_layers=2, num_residual_hiddens=256,
                  num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
    if os.path.exists('checkpoints_vqvae_miniimagenet/vqvae_final.pt'):
        vqvae.load_state_dict(torch.load('checkpoints_vqvae_miniimagenet/vqvae_final.pt', map_location='cpu'))
    vqvae.to('cpu')
    vqvae.eval()

    # 2. Load Teacher
    print("Loading Teacher (Epoch 100)...")
    teacher_config = GPTConfig(1024, 256, 20, 16, 1024, num_classes=None)
    teacher = GPT(teacher_config)
    t_ckpt = torch.load('checkpoints_teacher_unconditional/teacher_epoch_100.pt', map_location=device)
    t_new = {k[10:] if k.startswith('_orig_mod.') else k: v for k,v in t_ckpt.items()}
    teacher.load_state_dict(t_new)
    teacher.to(device)
    teacher.eval()

    # 3. Generate Samples
    gen_dir = 'benchmark_gen_teacher'
    os.makedirs(gen_dir, exist_ok=True)
    
    print(f"Generating {num_samples} samples...")
    count = 0
    pbar = tqdm(total=num_samples)
    
    # Load token references just for random start token logic if we want consistency, 
    # but Teacher can generate from scratch too.
    # Let's use valid start tokens to be fair to students (though unconditional should deal with anything)
    token_path = 'dataset_tokens/dogs_08_17_tokens.pt'
    data = torch.load(token_path)
    if isinstance(data, dict): seqs = data['tokens']
    else: seqs = data
    
    while count < num_samples:
        bs = 4
        # Random valid start token
        idx = torch.randint(0, len(seqs), (bs,))
        start_tokens = seqs[idx, 0:1].to(device)
        
        with torch.no_grad():
            # Generate
            out = teacher.generate(start_tokens, max_new_tokens=255, temperature=1.0, top_k=100)
            
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

    # 4. Compute Metrics
    print("Computing metrics...")
    real_images_dir = 'benchmark_real_samples_dogs'
    if not os.path.exists(real_images_dir):
        print("Real samples directory not found! Run benchmark_hybrid.py first.")
        return

    fid = FrechetInceptionDistance(feature=2048).to('cpu')
    inception = InceptionScore().to('cpu')
    
    print("Loading real images...")
    real_imgs = load_images_as_tensor(real_images_dir, num_samples).to('cpu')
    print("Loading generated images...")
    gen_imgs = load_images_as_tensor(gen_dir, num_samples).to('cpu')
    
    print("Updating FID...")
    fid.update(real_imgs, real=True)
    fid.update(gen_imgs, real=False)
    fid_score = fid.compute().item()
    
    print("Updating IS...")
    inception.update(gen_imgs)
    is_score, _ = inception.compute()
    is_score = is_score.item()
    
    print(f"\nTeacher Results:\nFID: {fid_score:.2f}\nIS: {is_score:.2f}")

if __name__ == "__main__":
    benchmark_teacher_quality()
