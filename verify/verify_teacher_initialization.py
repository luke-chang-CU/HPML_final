
import torch
import os
from model import GPT, GPTConfig
from vqvae import VQVAE
from PIL import Image
import numpy as np

def verify_initialization():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Load Tokens
    token_path = 'dataset_tokens/dogs_08_17_tokens.pt'
    if not os.path.exists(token_path):
        print("Token file not found.")
        return
    
    data = torch.load(token_path)
    if isinstance(data, dict):
        tokens = data['tokens'] # (N, 256)
    else:
        tokens = data
        
    print(f"Loaded {len(tokens)} sequences.")
    
    # Pick 5 random indices
    indices = torch.randint(0, len(tokens), (5,))
    selected_tokens = tokens[indices] # (5, 256)
    
    # Extract just the first token for initialization
    start_tokens = selected_tokens[:, 0:1].to(device) # (5, 1)
    
    # 2. Load Teacher
    print("Loading Teacher...")
    config = GPTConfig(
        vocab_size=1024,
        block_size=256,
        n_layer=20,
        n_head=16,
        n_embd=1024,
        num_classes=None
    )
    model = GPT(config)
    
    ckpt_path = 'checkpoints_teacher_unconditional/teacher_final.pt'
    state_dict = torch.load(ckpt_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # 3. Load VQ-VAE for Decoding
    print("Loading VQ-VAE...")
    vqvae = VQVAE(num_hiddens=1024, num_residual_layers=2, num_residual_hiddens=256,
                  num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
    vqvae.load_state_dict(torch.load('checkpoints_vqvae_miniimagenet/vqvae_final.pt', map_location='cpu'))
    vqvae.to('cpu')
    vqvae.eval()

    # 4. Generate
    print("Generating...")
    with torch.no_grad():
        # Generate full 256 tokens (start_tokens is 1, so generates 255 more)
        # Note: top_k=100, temp=1.0. With 0.02 loss, prob dist should be sharp.
        generated_indices = model.generate(start_tokens, max_new_tokens=255, temperature=1.0, top_k=100)
    
    # 5. Decode and Save
    output_dir = 'verify_init_samples'
    os.makedirs(output_dir, exist_ok=True)
    
    gen_cpu = generated_indices.cpu()
    
    for i in range(5):
        # A. Decode Generated
        idx_gen = gen_cpu[i].view(1, 256)
        decoded_gen = vqvae.decode(idx_gen)
        img_gen = decoded_gen[0].detach().cpu().permute(1, 2, 0).clamp(0, 1)
        
        # B. Decode Original (Ground Truth from Token File)
        idx_real = selected_tokens[i].view(1, 256) # (1, 256)
        decoded_real = vqvae.decode(idx_real)
        img_real = decoded_real[0].detach().cpu().permute(1, 2, 0).clamp(0, 1)
        
        # Concatenate: Left (Generated), Right (Original)
        combined = torch.cat((img_gen, img_real), dim=1)
        
        img_np = (combined.numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        save_path = os.path.join(output_dir, f'comparison_{i}.png')
        img_pil.save(save_path)
        print(f"Saved {save_path} (Left: Gen, Right: Real)")

if __name__ == "__main__":
    verify_initialization()
