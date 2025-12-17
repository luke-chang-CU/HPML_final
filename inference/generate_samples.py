
import torch
from model import GPT, GPTConfig
from vqvae import VQVAE
import os
from PIL import Image
import numpy as np

def generate_samples():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Load Teacher
    print("Loading Teacher...")
    config = GPTConfig(
        vocab_size=1024,
        block_size=256,
        n_layer=20,
        n_head=16,
        n_embd=1024,
        num_classes=64
    )
    model = GPT(config)

    ckpt_path = 'checkpoints_teacher_vqvae/teacher_epoch_49.pt'
    if not os.path.exists(ckpt_path):
        print("Checkpoint not found!")
        return

    state_dict = torch.load(ckpt_path, map_location=device)
    # Fix for compiled model saving
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # 2. Load VQ-VAE
    print("Loading VQ-VAE...")
    vqvae = VQVAE(num_hiddens=1024, num_residual_layers=2, num_residual_hiddens=256,
                  num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)

    vqvae_path = 'checkpoints_vqvae_miniimagenet/vqvae_final.pt'
    if os.path.exists(vqvae_path):
        vqvae.load_state_dict(torch.load(vqvae_path, map_location='cpu'))

    vqvae.to('cpu')
    vqvae.eval()

    # 3. Generate 20 Samples for Class 5
    target_class = 5
    num_samples = 20
    output_dir = f'class_{target_class}_samples'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {num_samples} samples for Class {target_class}...")

    # We can batch this somewhat
    batch_size = 5
    total_generated = 0

    while total_generated < num_samples:
        curr_bs = min(batch_size, num_samples - total_generated)
        
        initial_idx = torch.randint(0, 1024, (curr_bs, 1)).to(device)
        labels = torch.full((curr_bs,), target_class, device=device)

        with torch.no_grad():
            generated = model.generate(initial_idx, max_new_tokens=255, temperature=1.0, top_k=100, class_labels=labels)

        # Decode
        generated_cpu = generated.cpu()
        
        for i in range(curr_bs):
            indices = generated_cpu[i].view(1, 256)
            decoded = vqvae.decode(indices)
            img_tensor = decoded[0].detach().cpu().permute(1, 2, 0).clamp(0, 1)
            img_np = (img_tensor.numpy() * 255).astype(np.uint8)
            
            save_path = os.path.join(output_dir, f'sample_{total_generated + i + 1}.png')
            Image.fromarray(img_np).save(save_path)
            print(f"Saved {save_path}")
            
        total_generated += curr_bs

    print(f"Done! Saved to {output_dir}/")

if __name__ == "__main__":
    generate_samples()
